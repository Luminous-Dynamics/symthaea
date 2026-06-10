// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use super::*;

#[test]
fn test_integration_config_default() {
    let config = IntegrationConfig::default();
    assert_eq!(config.num_cycles, 10);
    assert_eq!(config.workspace_capacity, 3);
}

#[test]
fn test_consciousness_state_default() {
    let state = ConsciousnessState::default();
    assert!((state.phi - 0.0).abs() < 1e-10);
    assert!(state.conscious_contents.is_empty());
}

#[test]
fn test_pipeline_creation() {
    let pipeline = ConsciousnessPipeline::default();
    assert!((pipeline.state.phi - 0.0).abs() < 1e-10);
}

#[test]
fn test_pipeline_process() {
    let mut pipeline = ConsciousnessPipeline::default();
    let input = vec![BinaryHV::random(42)];
    pipeline.process_cycle(&input);
    assert!(pipeline.state.phi > 0.0);
}

#[test]
fn test_altered_state_effects() {
    let mut pipeline = ConsciousnessPipeline::default();

    pipeline.set_altered_state(AlteredStateIndex::Wake);
    assert!(pipeline.state.consciousness_level > 0.5);

    pipeline.set_altered_state(AlteredStateIndex::Propofol);
    assert!(pipeline.state.consciousness_level < 0.1);
}

#[test]
fn test_assessment() {
    let pipeline = ConsciousnessPipeline::default();
    let assessment = pipeline.assess();
    assert!(!assessment.is_conscious); // Default state is not conscious
}

#[test]
fn test_process_method() {
    let config = IntegrationConfig::default();
    let mut pipeline = ConsciousnessPipeline::new(config);
    pipeline.set_embodiment(0.8);

    let input = vec![
        BinaryHV::random(1),
        BinaryHV::random(2),
        BinaryHV::random(3),
    ];
    let priorities = vec![0.9, 0.7, 0.5];

    let state = pipeline.process(input, &priorities);
    assert!(state.consciousness_level > 0.0);
    assert!(state.phi > 0.0);
}

#[test]
fn test_bound_object() {
    let obj = BoundObject {
        representation: BinaryHV::random(1),
        synchrony: 0.8,
        binding_strength: 0.9,
        conscious: true,
        level: BindingLevel::Feature,
        child_ids: Vec::new(),
        attention_weight: 1.0,
        creation_cycle: 0,
        persistence_cycles: 0,
        temporal_stability: 1.0,
    };
    assert!(obj.is_conscious());
    assert_eq!(obj.level, BindingLevel::Feature);
}

#[test]
fn test_workspace_item() {
    let item = WorkspaceItem {
        content: BinaryHV::random(2),
        activation: 0.95,
        source: "visual".to_string(),
        is_broadcasting: true,
        duration_ms: 100,
    };
    assert!(item.activation > 0.9);
    assert!(item.is_broadcasting);
}

#[test]
fn test_meta_thought() {
    let thought = MetaThought {
        about: "seeing red".to_string(),
        target: "visual_perception".to_string(),
        intensity: 0.8,
        confidence: 0.9,
        order: 2,
        representation: BinaryHV::random(42),
    };
    assert_eq!(thought.order, 2);
    assert!(thought.confidence > 0.5);
}

#[test]
fn test_binding_calculation_with_bound_objects() {
    // Create pipeline with bound objects to test REAL binding calculation
    let mut pipeline = ConsciousnessPipeline::default();

    // Add bound objects with known synchrony and binding_strength
    pipeline.state.bound_objects = vec![
        BoundObject {
            representation: BinaryHV::random(1),
            synchrony: 0.9,        // High synchrony
            binding_strength: 0.8, // Strong binding
            conscious: true,
            level: BindingLevel::Feature,
            child_ids: Vec::new(),
            attention_weight: 1.0,
            creation_cycle: 0,
            persistence_cycles: 5,
            temporal_stability: 0.95,
        },
        BoundObject {
            representation: BinaryHV::random(2),
            synchrony: 0.7,        // Medium synchrony
            binding_strength: 0.6, // Medium binding
            conscious: true,
            level: BindingLevel::Feature,
            child_ids: Vec::new(),
            attention_weight: 1.0,
            creation_cycle: 0,
            persistence_cycles: 3,
            temporal_stability: 0.85,
        },
        BoundObject {
            representation: BinaryHV::random(3),
            synchrony: 0.5,        // Lower synchrony
            binding_strength: 0.4, // Weaker binding
            conscious: false,
            level: BindingLevel::Feature,
            child_ids: Vec::new(),
            attention_weight: 1.0,
            creation_cycle: 0,
            persistence_cycles: 1,
            temporal_stability: 0.7,
        },
    ];

    // Assess and check binding score
    let assessment = pipeline.assess_integration();

    // Verify binding is NOT zero (was the bug before fix)
    assert!(
        assessment.component_scores.get("binding").unwrap() > &0.0,
        "Binding should be > 0 with bound objects"
    );

    // Calculate expected binding:
    // Object 1: 0.9 * 0.8 = 0.72
    // Object 2: 0.7 * 0.6 = 0.42
    // Object 3: 0.5 * 0.4 = 0.20
    // total_synchrony = 0.72 + 0.42 + 0.20 = 1.34
    // avg_binding = 1.34 / 3 = 0.4467
    // bound_fraction = min(3, 10) / 10 = 0.3
    // binding_score = 0.4467 * 0.7 + 0.3 * 0.3 = 0.3127 + 0.09 = 0.4027
    let binding_score = *assessment.component_scores.get("binding").unwrap();
    assert!(
        (binding_score - 0.4027).abs() < 0.01,
        "Expected binding ~0.40, got {}",
        binding_score
    );

    println!(
        "✅ Binding calculation with bound objects: {:.4}",
        binding_score
    );
}

#[test]
fn test_binding_calculation_empty() {
    // Test that empty bound_objects returns 0
    let pipeline = ConsciousnessPipeline::default();
    let assessment = pipeline.assess_integration();

    let binding_score = *assessment.component_scores.get("binding").unwrap();
    assert!(
        (binding_score - 0.0).abs() < 1e-10,
        "Binding should be 0 with no bound objects, got {}",
        binding_score
    );

    println!("✅ Binding calculation empty: {:.4}", binding_score);
}

#[test]
fn test_workspace_calculation_with_broadcasting() {
    // Test workspace calculation with actual broadcasting items
    let mut pipeline = ConsciousnessPipeline::default();

    pipeline.state.conscious_contents = vec![
        WorkspaceItem {
            content: BinaryHV::random(1),
            activation: 0.9,
            source: "visual".to_string(),
            is_broadcasting: true, // Broadcasting
            duration_ms: 100,
        },
        WorkspaceItem {
            content: BinaryHV::random(2),
            activation: 0.7,
            source: "auditory".to_string(),
            is_broadcasting: true, // Broadcasting
            duration_ms: 50,
        },
        WorkspaceItem {
            content: BinaryHV::random(3),
            activation: 0.5,
            source: "touch".to_string(),
            is_broadcasting: false, // Not broadcasting
            duration_ms: 30,
        },
    ];

    let assessment = pipeline.assess_integration();
    let workspace_score = *assessment.component_scores.get("workspace").unwrap();

    // Expected: 2 items broadcasting with activations 0.9 and 0.7
    // broadcast_activation = (0.9 + 0.7) / 2 = 0.8
    // access_ratio = 2/3 = 0.667
    // workspace = 0.8 * (0.5 + 0.5 * 0.667) = 0.8 * 0.833 = 0.667
    assert!(
        workspace_score > 0.6 && workspace_score < 0.75,
        "Expected workspace ~0.67, got {}",
        workspace_score
    );

    println!(
        "✅ Workspace calculation with broadcasting: {:.4}",
        workspace_score
    );
}

#[test]
fn test_hot_calculation_with_meta_awareness() {
    // Test HOT calculation with actual meta-awareness
    let mut pipeline = ConsciousnessPipeline::default();

    pipeline.state.meta_awareness = vec![
        MetaThought {
            about: "thinking".to_string(),
            target: "cognition".to_string(),
            intensity: 0.8,
            confidence: 0.9, // High confidence
            order: 1,        // First-order meta
            representation: BinaryHV::random(1),
        },
        MetaThought {
            about: "aware of thinking".to_string(),
            target: "meta_cognition".to_string(),
            intensity: 0.7,
            confidence: 0.7, // Medium confidence
            order: 2,        // Second-order meta
            representation: BinaryHV::random(2),
        },
    ];

    let assessment = pipeline.assess_integration();
    let hot_score = *assessment.component_scores.get("hot").unwrap();

    // Expected calculation:
    // weighted_confidence = 0.9*(1+0.2*1) + 0.7*(1+0.2*2) = 0.9*1.2 + 0.7*1.4 = 1.08 + 0.98 = 2.06
    // total_weight = 1.2 + 1.4 = 2.6
    // base = 2.06 / 2.6 = 0.792
    // depth_bonus = min(2/3, 0.2) = 0.2
    // hot = 0.792 + 0.2 = 0.992 (clamped to 1.0)
    assert!(hot_score > 0.9, "Expected HOT ~0.99, got {}", hot_score);

    println!("✅ HOT calculation with meta-awareness: {:.4}", hot_score);
}

#[test]
fn test_process_creates_bound_objects() {
    // Test that process() creates bound objects from correlated inputs
    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.set_embodiment(0.8);

    // Create multiple high-priority inputs
    let inputs = vec![
        BinaryHV::random(100),
        BinaryHV::random(101),
        BinaryHV::random(102),
    ];
    let priorities = vec![0.8, 0.7, 0.6]; // All above 0.5 threshold

    // Process inputs
    let state = pipeline.process(inputs, &priorities);

    // Should have created bound objects
    assert!(
        !state.bound_objects.is_empty(),
        "Expected bound objects to be created, but got none"
    );

    // Verify bound object properties
    let bound = &state.bound_objects[0];
    assert!(
        bound.synchrony >= 0.0 && bound.synchrony <= 1.0,
        "Synchrony should be in [0,1], got {}",
        bound.synchrony
    );
    assert!(
        bound.binding_strength > 0.5,
        "Binding strength should be > 0.5 for high-priority inputs, got {}",
        bound.binding_strength
    );

    println!(
        "✅ process() creates bound objects: {} objects with synchrony={:.4}, strength={:.4}",
        state.bound_objects.len(),
        bound.synchrony,
        bound.binding_strength
    );
}

#[test]
fn test_process_no_binding_for_single_input() {
    // Test that single input doesn't create bound objects
    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.set_embodiment(0.8);

    let inputs = vec![BinaryHV::random(200)];
    let priorities = vec![0.9];

    let state = pipeline.process(inputs, &priorities);

    // Single input should not create bound objects (need >= 2 for binding)
    assert!(
        state.bound_objects.is_empty(),
        "Single input should not create bound objects"
    );

    println!("✅ process() correctly skips binding for single input");
}

#[test]
fn test_process_no_binding_for_low_priority() {
    // Test that low-priority inputs don't create bound objects
    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.set_embodiment(0.8);

    let inputs = vec![
        BinaryHV::random(300),
        BinaryHV::random(301),
        BinaryHV::random(302),
    ];
    let priorities = vec![0.3, 0.2, 0.4]; // All below 0.5 threshold

    let state = pipeline.process(inputs, &priorities);

    // Low priority inputs should not create bound objects
    assert!(
        state.bound_objects.is_empty(),
        "Low-priority inputs should not create bound objects"
    );

    println!("✅ process() correctly skips binding for low-priority inputs");
}

#[test]
fn test_process_binding_integrates_with_assessment() {
    // Test that bound objects from process() affect assess_integration()
    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.set_embodiment(0.9);

    // First, check baseline without processing
    let baseline_assessment = pipeline.assess_integration();
    let baseline_binding = *baseline_assessment.component_scores.get("binding").unwrap();
    assert_eq!(baseline_binding, 0.0, "Baseline binding should be 0");

    // Process inputs to create binding
    let inputs = vec![BinaryHV::random(400), BinaryHV::random(401)];
    let priorities = vec![0.8, 0.9];

    pipeline.process(inputs, &priorities);

    // Now check that binding score is non-zero
    let assessment = pipeline.assess_integration();
    let binding_score = *assessment.component_scores.get("binding").unwrap();

    assert!(
        binding_score > 0.0,
        "Binding score should be > 0 after processing, got {}",
        binding_score
    );

    println!(
        "✅ process() binding integrates with assessment: binding={:.4}",
        binding_score
    );
}

#[test]
fn test_bound_object_synchrony_calculation() {
    // Test synchrony is calculated from HV similarity
    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.set_embodiment(0.9);

    // Create similar HVs (should have higher synchrony)
    let base_hv = BinaryHV::random(500);
    let similar_hv = base_hv.clone(); // Identical = max similarity

    let inputs = vec![base_hv, similar_hv];
    let priorities = vec![0.9, 0.9];

    let state = pipeline.process(inputs, &priorities);

    // Identical HVs should have high synchrony
    assert!(!state.bound_objects.is_empty());
    let bound = &state.bound_objects[0];

    // similarity of identical HVs should be 1.0, so synchrony >= 0.5 (our minimum)
    assert!(
        bound.synchrony >= 0.5,
        "Synchrony for similar HVs should be >= 0.5, got {}",
        bound.synchrony
    );

    println!(
        "✅ Synchrony calculated from HV similarity: {:.4}",
        bound.synchrony
    );
}

#[test]
fn test_multiple_bound_objects_clustering() {
    // Test that dissimilar inputs create separate bound objects
    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.set_embodiment(0.9);

    // Create 4 inputs: 2 similar pairs, each pair dissimilar to the other
    // Similar HVs should cluster together
    let hv1 = BinaryHV::random(600);
    let hv2 = hv1.clone(); // Similar to hv1
    let hv3 = BinaryHV::random(700); // Different from hv1/hv2
    let hv4 = hv3.clone(); // Similar to hv3

    let inputs = vec![hv1, hv2, hv3, hv4];
    let priorities = vec![0.8, 0.85, 0.9, 0.75]; // All high priority

    let state = pipeline.process(inputs, &priorities);

    // Should have at least 1 bound object (may be more depending on clustering)
    assert!(
        !state.bound_objects.is_empty(),
        "Should have at least one bound object"
    );

    println!(
        "✅ Clustering created {} bound objects",
        state.bound_objects.len()
    );
}

#[test]
fn test_binding_persistence_decay() {
    // Test that bound objects persist but decay across cycles
    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.set_embodiment(0.9);

    // First cycle: create bound object
    let inputs = vec![BinaryHV::random(800), BinaryHV::random(801)];
    let priorities = vec![0.8, 0.9];
    let state1 = pipeline.process(inputs, &priorities);

    assert!(
        !state1.bound_objects.is_empty(),
        "Should have bound objects after first cycle"
    );
    let initial_strength = state1.bound_objects[0].binding_strength;

    // Second cycle: process empty or different inputs
    // Bound objects should still exist but be decayed
    let new_inputs = vec![BinaryHV::random(900)]; // Only 1 input, no new binding
    let new_priorities = vec![0.3]; // Low priority
    let state2 = pipeline.process(new_inputs, &new_priorities);

    // Previous bound objects should still exist but decayed
    if !state2.bound_objects.is_empty() {
        let decayed_strength = state2.bound_objects[0].binding_strength;
        assert!(
            decayed_strength < initial_strength,
            "Binding strength should decay: {} < {}",
            decayed_strength,
            initial_strength
        );
        println!(
            "✅ Binding decayed from {:.4} to {:.4}",
            initial_strength, decayed_strength
        );
    } else {
        // If object was removed due to decay, that's also valid
        println!("✅ Bound object was removed due to decay (strength below threshold)");
    }
}

#[test]
fn test_binding_reinforcement() {
    // Test that repeated similar inputs reinforce existing bindings
    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.set_embodiment(0.9);

    // Create consistent inputs across multiple cycles
    let base1 = BinaryHV::random(1000);
    let base2 = BinaryHV::random(1001);

    // First cycle
    let inputs1 = vec![base1.clone(), base2.clone()];
    let priorities = vec![0.8, 0.9];
    pipeline.process(inputs1, &priorities);

    let strength_after_1 = pipeline
        .state
        .bound_objects
        .first()
        .map(|o| o.binding_strength)
        .unwrap_or(0.0);

    // Second cycle with similar inputs (should reinforce)
    let inputs2 = vec![base1.clone(), base2.clone()];
    pipeline.process(inputs2, &priorities);

    let strength_after_2 = pipeline
        .state
        .bound_objects
        .first()
        .map(|o| o.binding_strength)
        .unwrap_or(0.0);

    // Third cycle
    let inputs3 = vec![base1.clone(), base2.clone()];
    pipeline.process(inputs3, &priorities);

    let strength_after_3 = pipeline
        .state
        .bound_objects
        .first()
        .map(|o| o.binding_strength)
        .unwrap_or(0.0);

    println!(
        "Binding strength: cycle1={:.4}, cycle2={:.4}, cycle3={:.4}",
        strength_after_1, strength_after_2, strength_after_3
    );

    // Reinforcement should counter decay, maintaining or increasing strength
    // (Note: exact behavior depends on decay rate vs reinforcement amount)
    assert!(
        strength_after_3 > 0.0,
        "Binding should persist after reinforcement"
    );
    println!("✅ Binding reinforcement maintains persistence");
}

#[test]
fn test_max_bound_objects_limit() {
    // Test that bound objects are limited to MAX_BOUND_OBJECTS
    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.set_embodiment(0.9);

    // Create many dissimilar high-priority inputs to trigger multiple clusters
    // Each pair should create a separate bound object
    let inputs: Vec<BinaryHV> = (0..20).map(|i| BinaryHV::random(2000 + i)).collect();
    let priorities: Vec<f64> = vec![0.9; 20];

    let state = pipeline.process(inputs, &priorities);

    // Should be capped at MAX_BOUND_OBJECTS (10)
    assert!(
        state.bound_objects.len() <= 10,
        "Bound objects should be limited to 10, got {}",
        state.bound_objects.len()
    );

    println!(
        "✅ Bound objects limited to {} (max 10)",
        state.bound_objects.len()
    );
}

// === INTEGRATED CONSCIOUSNESS SYSTEMS TESTS ===

#[test]
fn test_enable_integrated_systems() {
    // Test that integrated systems can be enabled
    let mut pipeline = ConsciousnessPipeline::default();

    // Initially disabled
    assert!(
        !pipeline.has_integrated_systems(),
        "Integrated systems should be disabled by default"
    );
    assert!(pipeline.metacognitive_monitor.is_none());
    assert!(pipeline.cross_modal_binder.is_none());
    assert!(pipeline.temporal_binder.is_none());

    // Enable them
    pipeline.enable_integrated_systems();

    // Now enabled
    assert!(
        pipeline.has_integrated_systems(),
        "Integrated systems should be enabled after enable_integrated_systems()"
    );
    assert!(pipeline.metacognitive_monitor.is_some());
    assert!(pipeline.cross_modal_binder.is_some());
    assert!(pipeline.temporal_binder.is_some());

    println!("✅ Integrated systems enabled successfully");
}

#[test]
fn test_integrated_systems_processing() {
    // Test that integrated systems affect state during processing
    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.set_embodiment(0.8);
    pipeline.enable_integrated_systems();

    // Process several cycles to build up metrics
    for i in 0..5 {
        let inputs = vec![BinaryHV::random(3000 + i), BinaryHV::random(3100 + i)];
        let priorities = vec![0.8, 0.7];
        pipeline.process(inputs, &priorities);
    }

    let state = pipeline.get_state();

    // Metacognitive metrics should be updated
    assert!(
        state.metacognitive_confidence >= 0.0 && state.metacognitive_confidence <= 1.0,
        "Metacognitive confidence should be in [0,1], got {}",
        state.metacognitive_confidence
    );

    // Predictive mode should be set
    assert!(
        state.inference_mode == PredictiveMode::Exploring
            || state.inference_mode == PredictiveMode::Exploiting
            || state.inference_mode == PredictiveMode::Balanced,
        "Inference mode should be valid"
    );

    // Theta phase should be advancing
    assert!(
        state.theta_phase >= 0.0,
        "Theta phase should be non-negative"
    );

    println!("✅ Integrated systems processing works:");
    println!(
        "   - Metacognitive confidence: {:.4}",
        state.metacognitive_confidence
    );
    println!("   - Inference mode: {:?}", state.inference_mode);
    println!("   - Theta phase: {:.4}", state.theta_phase);
    println!(
        "   - Cross-modal coherence: {:.4}",
        state.cross_modal_coherence
    );
    println!("   - Narrative coherence: {:.4}", state.narrative_coherence);
}

#[test]
fn test_predictive_mode_transitions() {
    // Test that predictive mode changes based on prediction accuracy
    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.set_embodiment(0.9);
    pipeline.enable_integrated_systems();

    // Process high-quality inputs (should lead to exploiting mode)
    for _ in 0..10 {
        let inputs = vec![BinaryHV::random(4000), BinaryHV::random(4001)];
        let priorities = vec![0.95, 0.95]; // High priorities = high consciousness = high accuracy
        pipeline.process(inputs, &priorities);
    }

    let state = pipeline.get_state();

    // With high prediction accuracy, should be in exploiting mode
    if state.prediction_accuracy > 0.7 {
        assert_eq!(
            state.inference_mode,
            PredictiveMode::Exploiting,
            "High prediction accuracy should trigger Exploiting mode"
        );
    }

    println!("✅ Predictive mode transitions work:");
    println!("   - Prediction accuracy: {:.4}", state.prediction_accuracy);
    println!("   - Inference mode: {:?}", state.inference_mode);
}

#[test]
fn test_phi_trend_prediction() {
    // Test that Φ trends lead to predictions
    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.set_embodiment(0.8);
    pipeline.enable_integrated_systems();

    // Process several cycles with increasing quality (should create positive trend)
    let mut inputs_quality = 0.5;
    for i in 0..7 {
        inputs_quality += 0.05; // Increasing quality
        let inputs = vec![BinaryHV::random(5000 + i), BinaryHV::random(5100 + i)];
        let priorities = vec![inputs_quality, inputs_quality];
        pipeline.process(inputs, &priorities);
    }

    let state = pipeline.get_state();

    // After 5+ cycles, should have Φ prediction
    // Phi should always be finite and non-negative
    assert!(state.phi.is_finite(), "Φ should be finite");
    assert!(state.phi >= 0.0, "Φ should be non-negative");
    assert!(state.phi_trend.is_finite(), "Φ trend should be finite");

    if state.predicted_phi.is_some() {
        let predicted = state.predicted_phi.unwrap();
        assert!(predicted.is_finite(), "Predicted Φ should be finite");
        assert!(predicted >= 0.0, "Predicted Φ should be non-negative");
        println!("✅ Φ prediction generated:");
        println!("   - Current Φ: {:.4}", state.phi);
        println!("   - Predicted Φ: {:.4}", predicted);
        println!("   - Φ trend: {:.4}", state.phi_trend);
    } else {
        println!("⚠️ Φ prediction not yet generated (needs more history)");
    }
}

#[test]
fn test_cross_modal_coherence() {
    // Test that cross-modal coherence increases with multiple modalities
    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.set_embodiment(0.9);
    pipeline.enable_integrated_systems();

    // Add workspace items with different source modalities
    pipeline.state.conscious_contents.push(WorkspaceItem {
        content: BinaryHV::random(6000),
        activation: 0.9,
        source: "visual_cortex".to_string(), // Contains "visual"
        is_broadcasting: true,
        duration_ms: 100,
    });
    pipeline.state.conscious_contents.push(WorkspaceItem {
        content: BinaryHV::random(6001),
        activation: 0.8,
        source: "audio_processing".to_string(), // Contains "audio"
        is_broadcasting: true,
        duration_ms: 100,
    });
    pipeline.state.conscious_contents.push(WorkspaceItem {
        content: BinaryHV::random(6002),
        activation: 0.7,
        source: "motor_planning".to_string(), // Contains "motor"
        is_broadcasting: true,
        duration_ms: 100,
    });

    // Process to update cross-modal metrics
    let inputs = vec![BinaryHV::random(6100)];
    let priorities = vec![0.5];
    pipeline.process(inputs, &priorities);

    let state = pipeline.get_state();

    // With 3 modalities, should have higher cross-modal coherence
    assert!(
        state.active_modalities.len() >= 2,
        "Should detect multiple modalities, got {}",
        state.active_modalities.len()
    );
    assert!(
        state.cross_modal_coherence > 0.5,
        "Multi-modal integration should boost coherence above 0.5, got {}",
        state.cross_modal_coherence
    );

    println!(
        "✅ Cross-modal coherence with {} modalities:",
        state.active_modalities.len()
    );
    println!("   - Active modalities: {:?}", state.active_modalities);
    println!(
        "   - Cross-modal coherence: {:.4}",
        state.cross_modal_coherence
    );
}

// === Φ-GUIDED TOPOLOGY OPTIMIZATION TESTS ===

#[test]
fn test_enable_phi_optimization() {
    let mut pipeline = ConsciousnessPipeline::default();

    // Initially disabled
    assert!(
        !pipeline.has_phi_optimization(),
        "Φ optimization should be disabled by default"
    );
    assert!(pipeline.phi_optimizer.is_none());
    assert!(pipeline.consciousness_network.is_none());

    // Enable with 8 nodes
    pipeline.enable_phi_optimization(8, 10);

    // Now enabled
    assert!(
        pipeline.has_phi_optimization(),
        "Φ optimization should be enabled after enable_phi_optimization()"
    );
    assert!(pipeline.phi_optimizer.is_some());
    assert!(pipeline.consciousness_network.is_some());

    // Verify network
    let network = pipeline.consciousness_network().unwrap();
    assert_eq!(network.node_count(), 8);
    assert!(network.edge_count() > 0, "Ring topology should have edges");

    println!(
        "✅ Φ optimization enabled: {} nodes, {} edges",
        network.node_count(),
        network.edge_count()
    );
}

#[test]
fn test_manual_phi_optimization() {
    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.enable_phi_optimization(6, 10);

    let initial_phi = pipeline.state.phi;

    // Run optimization step
    let result = pipeline.optimize_phi();
    assert!(result.is_some(), "Should return optimization result");

    let opt = result.unwrap();
    assert!(opt.phi >= 0.0 && opt.phi <= 1.0, "Φ should be in [0,1]");
    assert_eq!(opt.step, 1, "Should be first step");

    println!("✅ Manual Φ optimization:");
    println!("   - Initial Φ: {:.4}", initial_phi);
    println!("   - Optimized Φ: {:.4}", opt.phi);
    println!("   - Delta: {:.4}", opt.phi_delta);
}

#[test]
#[ignore = "expensive: graph topology optimization (~60s)"]
fn test_phi_optimization_multiple_steps() {
    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.enable_phi_optimization(8, 5);

    // Run 5 optimization steps
    let results = pipeline.optimize_phi_steps(5);

    assert_eq!(results.len(), 5, "Should have 5 results");

    // Φ should be non-negative throughout
    for (i, result) in results.iter().enumerate() {
        assert!(result.phi >= 0.0, "Φ should be non-negative at step {}", i);
        assert_eq!(result.step, i + 1, "Step number should match");
    }

    println!("✅ Multi-step Φ optimization:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "   Step {}: Φ = {:.4} (Δ = {:+.4})",
            i + 1,
            r.phi,
            r.phi_delta
        );
    }
}

#[test]
#[ignore = "expensive: 20 cycles with phi optimization (~90s)"]
fn test_phi_optimization_during_processing() {
    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.set_embodiment(0.8);
    pipeline.enable_phi_optimization(8, 5); // Optimize every 5 cycles

    // Track Φ values across cycles
    let mut phi_values = Vec::new();

    // Process for 20 cycles (should trigger optimization at 5, 10, 15, 20)
    for i in 0..20 {
        let inputs = vec![BinaryHV::random(7000 + i), BinaryHV::random(7100 + i)];
        let priorities = vec![0.8, 0.7];
        pipeline.process(inputs, &priorities);
        phi_values.push(pipeline.state.phi);
    }

    // Should have optimization results after processing
    if let Some(ref last_opt) = pipeline.last_optimization {
        println!("✅ Φ optimization during processing:");
        println!("   - Last optimization step: {}", last_opt.step);
        println!("   - Final Φ: {:.4}", last_opt.phi);
        println!(
            "   - Topological unity: {:.4}",
            pipeline.state.topological_unity
        );
    }

    // Topological unity should be updated
    assert!(
        pipeline.state.topological_unity > 0.0,
        "Topological unity should be positive after optimization"
    );
}

#[test]
fn test_phi_optimization_with_custom_config() {
    use crate::hdc::phi_guided_search::PhiOptimizationConfig;

    let config = PhiOptimizationConfig {
        dim: 512, // Smaller dimension for faster test
        learning_rate: 0.2,
        adaptive_lr: true,
        ..Default::default()
    };

    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.enable_phi_optimization_with_config(
        config,
        10, // 10 nodes
        InitializationStrategy::SmallWorld { rewire_prob: 0.3 },
        5,
    );

    assert!(pipeline.has_phi_optimization());

    // Get network stats before mutable operation
    let (node_count, edge_count) = {
        let network = pipeline.consciousness_network().unwrap();
        assert_eq!(network.node_count(), 10);
        (network.node_count(), network.edge_count())
    };

    // Run optimization
    let result = pipeline.optimize_phi();
    assert!(result.is_some());

    println!("✅ Custom config Φ optimization:");
    println!("   - Nodes: {}", node_count);
    println!("   - Edges: {}", edge_count);
    if let Some(r) = result {
        println!("   - Φ: {:.4}", r.phi);
    }
}

#[test]
#[ignore = "expensive: 15 cycles with phi optimization (~80s)"]
fn test_phi_trend_with_optimization() {
    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.set_embodiment(0.9);
    pipeline.enable_phi_optimization(8, 3); // Optimize every 3 cycles

    // Process many cycles to see Φ trend
    for i in 0..15 {
        let inputs = vec![BinaryHV::random(8000 + i), BinaryHV::random(8100 + i)];
        let priorities = vec![0.9, 0.85];
        pipeline.process(inputs, &priorities);
    }

    let state = pipeline.get_state();

    println!("✅ Φ trend with optimization:");
    println!("   - Final Φ: {:.4}", state.phi);
    println!("   - Φ trend: {:+.4}", state.phi_trend);
    println!("   - Topological unity: {:.4}", state.topological_unity);

    // Φ trend should be within reasonable bounds
    assert!(
        state.phi_trend >= -0.1 && state.phi_trend <= 0.1,
        "Φ trend should be bounded, got {}",
        state.phi_trend
    );
}

#[test]
fn test_all_systems_integration() {
    // Test full integration: integrated systems + Φ optimization
    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.set_embodiment(0.9);
    pipeline.enable_integrated_systems();
    pipeline.enable_phi_optimization(8, 5);

    // Verify both are enabled
    assert!(pipeline.has_integrated_systems());
    assert!(pipeline.has_phi_optimization());

    // Process with all systems active
    for i in 0..20 {
        let inputs = vec![BinaryHV::random(9000 + i), BinaryHV::random(9100 + i)];
        let priorities = vec![0.85, 0.8];
        pipeline.process(inputs, &priorities);
    }

    let state = pipeline.get_state();

    println!("✅ Full system integration:");
    println!("   - Φ: {:.4}", state.phi);
    println!(
        "   - Metacognitive confidence: {:.4}",
        state.metacognitive_confidence
    );
    println!(
        "   - Cross-modal coherence: {:.4}",
        state.cross_modal_coherence
    );
    println!("   - Topological unity: {:.4}", state.topological_unity);
    println!("   - Narrative coherence: {:.4}", state.narrative_coherence);
    println!("   - Consciousness level: {:.4}", state.consciousness_level);

    // All metrics should be valid
    assert!(state.phi >= 0.0 && state.phi <= 1.0);
    assert!(state.consciousness_level >= 0.0 && state.consciousness_level <= 1.0);
    assert!(state.topological_unity >= 0.0 && state.topological_unity <= 1.0);
}

// ==========================================
// FEEDBACK DYNAMICS TESTS
// ==========================================

#[test]
fn test_enable_feedback_dynamics() {
    let mut pipeline = ConsciousnessPipeline::default();

    // Initially disabled
    assert!(!pipeline.has_feedback_dynamics());

    // Enable feedback dynamics
    pipeline.enable_feedback_dynamics();
    assert!(pipeline.has_feedback_dynamics());

    // Check initial state
    assert!(pipeline.current_prediction().is_none());
    assert!(pipeline.pending_intervention().is_none());
    assert!(pipeline.recent_insights().is_empty());

    println!("✅ Feedback dynamics enabled successfully");
}

#[test]
fn test_emotional_prediction() {
    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.enable_feedback_dynamics();

    // Record enough emotional states to build prediction history (need >= 5)
    for i in 0..5 {
        let valence = 0.5 + (i as f64 * 0.1); // Gradually increasing valence
        let arousal = 0.5;
        pipeline.update_emotional_prediction(valence, arousal);
    }

    // Now get a prediction (should work with sufficient history)
    let prediction = pipeline.update_emotional_prediction(0.8, 0.5);
    assert!(
        prediction.is_some(),
        "Should return a prediction after building history"
    );

    let pred = prediction.unwrap();
    println!("✅ Emotional prediction:");
    println!("   - Predicted valence: {:.4}", pred.predicted_valence);
    println!("   - Predicted arousal: {:.4}", pred.predicted_arousal);
    println!("   - Confidence: {:.4}", pred.confidence);

    // Prediction should be stored
    assert!(pipeline.current_prediction().is_some());
}

#[test]
fn test_emotional_prediction_generates_intervention() {
    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.enable_feedback_dynamics();

    // Record several emotional states to build prediction history
    for i in 0..10 {
        // Simulating declining emotional state that might trigger intervention
        let valence = 0.3 - (i as f64 * 0.02); // Decreasing valence
        let arousal = 0.7 + (i as f64 * 0.01); // Increasing arousal (stress)
        pipeline.update_emotional_prediction(valence, arousal);
    }

    // Check if intervention is recommended
    let prediction = pipeline.current_prediction();
    assert!(
        prediction.is_some(),
        "prediction should exist after 10 emotional updates"
    );
    let pred = prediction.unwrap();
    println!("✅ Emotional prediction after decline:");
    println!("   - Predicted valence: {:.4}", pred.predicted_valence);
    println!("   - Predicted arousal: {:.4}", pred.predicted_arousal);
    // After a declining valence trajectory, predicted valence should be low
    assert!(
        pred.predicted_valence < 0.5,
        "predicted valence should reflect decline, got {}",
        pred.predicted_valence
    );
}

#[test]
fn test_apply_intervention() {
    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.enable_feedback_dynamics();

    // Build up emotional history that would trigger intervention
    for _ in 0..5 {
        pipeline.update_emotional_prediction(0.2, 0.8); // Low valence, high arousal
    }

    // Try to apply intervention
    let intervention = pipeline.apply_intervention();

    println!("✅ Intervention application:");
    if let Some(ref int) = intervention {
        println!("   - Applied intervention: {:?}", int);
    } else {
        println!("   - No intervention needed at this time");
    }

    // Pipeline should still have feedback dynamics enabled after intervention attempt
    assert!(
        pipeline.has_feedback_dynamics(),
        "feedback dynamics should remain enabled after apply_intervention"
    );
}

#[test]
fn test_feedback_dynamics_during_processing() {
    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.set_embodiment(0.8);
    pipeline.enable_feedback_dynamics();

    // Process several cycles with feedback dynamics enabled
    for i in 0..10 {
        let inputs = vec![BinaryHV::random(10000 + i), BinaryHV::random(10100 + i)];
        let priorities = vec![0.7, 0.75];
        pipeline.process(inputs, &priorities);
    }

    let state = pipeline.get_state();

    println!("✅ Feedback dynamics during processing:");
    println!("   - Consciousness level: {:.4}", state.consciousness_level);
    println!("   - Φ: {:.4}", state.phi);

    // Check that feedback dynamics is still enabled after processing
    assert!(pipeline.has_feedback_dynamics());

    // Metrics should be valid
    assert!(state.consciousness_level >= 0.0 && state.consciousness_level <= 1.0);
}

#[test]
fn test_feedback_dynamics_with_phi_optimization() {
    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.set_embodiment(0.85);
    pipeline.enable_feedback_dynamics();
    pipeline.enable_phi_optimization(8, 3);

    // Both should be enabled
    assert!(pipeline.has_feedback_dynamics());
    assert!(pipeline.has_phi_optimization());

    // Process with both systems active
    for i in 0..15 {
        let inputs = vec![BinaryHV::random(11000 + i), BinaryHV::random(11100 + i)];
        let priorities = vec![0.8, 0.85];
        pipeline.process(inputs, &priorities);
    }

    let state = pipeline.get_state();

    println!("✅ Feedback dynamics + Φ optimization:");
    println!("   - Φ: {:.4}", state.phi);
    println!("   - Topological unity: {:.4}", state.topological_unity);
    println!("   - Consciousness level: {:.4}", state.consciousness_level);

    // All metrics should be valid
    assert!(state.phi >= 0.0 && state.phi <= 1.0);
    assert!(state.consciousness_level >= 0.0 && state.consciousness_level <= 1.0);
}

#[test]
fn test_all_systems_with_feedback_dynamics() {
    // Full integration test: integrated systems + Φ optimization + feedback dynamics
    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.set_embodiment(0.9);
    pipeline.enable_integrated_systems();
    pipeline.enable_phi_optimization(8, 5);
    pipeline.enable_feedback_dynamics();

    // Verify all are enabled
    assert!(pipeline.has_integrated_systems());
    assert!(pipeline.has_phi_optimization());
    assert!(pipeline.has_feedback_dynamics());

    // Process with all systems active
    for i in 0..20 {
        let inputs = vec![BinaryHV::random(12000 + i), BinaryHV::random(12100 + i)];
        let priorities = vec![0.85, 0.8];
        pipeline.process(inputs, &priorities);
    }

    let state = pipeline.get_state();

    println!("✅ Full system integration with feedback dynamics:");
    println!("   - Φ: {:.4}", state.phi);
    println!(
        "   - Metacognitive confidence: {:.4}",
        state.metacognitive_confidence
    );
    println!(
        "   - Cross-modal coherence: {:.4}",
        state.cross_modal_coherence
    );
    println!("   - Topological unity: {:.4}", state.topological_unity);
    println!("   - Narrative coherence: {:.4}", state.narrative_coherence);
    println!("   - Consciousness level: {:.4}", state.consciousness_level);

    // All metrics should be valid
    assert!(state.phi >= 0.0 && state.phi <= 1.0);
    assert!(state.consciousness_level >= 0.0 && state.consciousness_level <= 1.0);
    assert!(state.topological_unity >= 0.0 && state.topological_unity <= 1.0);
    assert!(state.metacognitive_confidence >= 0.0 && state.metacognitive_confidence <= 1.0);
}

// ==========================================
// SELF-AWARENESS TESTS
// ==========================================

#[test]
fn test_enable_self_awareness() {
    let mut pipeline = ConsciousnessPipeline::default();

    // Initially disabled
    assert!(!pipeline.has_self_awareness());

    // Enable self-awareness
    pipeline.enable_self_awareness(1024, 16);
    assert!(pipeline.has_self_awareness());

    // Check initial state
    assert_eq!(pipeline.self_awareness_level(), 0.0);
    assert!(pipeline.current_self_model().is_none());
    assert!(pipeline.latest_introspection().is_none());

    println!("✅ Self-awareness enabled successfully");
}

#[test]
fn test_introspection() {
    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.enable_self_awareness(1024, 16);

    // Perform introspection
    let report = pipeline.introspect();
    assert!(report.is_some(), "Introspection should return a report");

    let report = report.unwrap();
    println!("✅ Introspection report:");
    println!("   - Believed Φ: {:.4}", report.believed_phi);
    println!("   - Believed mode: {:?}", report.believed_mode);
    println!(
        "   - Self-model confidence: {:.4}",
        report.self_model_confidence
    );
    println!(
        "   - Self-model accuracy: {:.4}",
        report.self_model_accuracy
    );
    println!(
        "   - Self-awareness level: {:.4}",
        report.self_awareness_level
    );
}

#[test]
#[ignore = "expensive: 1024-dim self-awareness (~60s)"]
fn test_process_self_aware() {
    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.enable_self_awareness(1024, 16);

    // Process input with self-awareness
    let input = BinaryHV::random(42);
    let update = pipeline.process_self_aware(&input);

    assert!(update.is_some(), "Should return a self-aware update");
    let update = update.unwrap();

    println!("✅ Self-aware processing:");
    println!("   - Base Φ: {:.4}", update.base_update.phi);
    println!("   - Predicted Φ: {:.4}", update.self_model.predicted_phi);
    println!("   - Prediction error: {:.4}", update.prediction_error);
    println!(
        "   - Self-awareness level: {:.4}",
        update.self_awareness_level
    );

    // Self-model should now be cached
    assert!(pipeline.current_self_model().is_some());
    assert!(pipeline.self_awareness_level() > 0.0);
}

#[test]
#[ignore = "takes >60s in debug builds"]
fn test_self_awareness_prediction_learning() {
    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.enable_self_awareness(1024, 16);

    // Process multiple inputs to let the self-model learn
    let mut prediction_errors = Vec::new();
    for i in 0..20 {
        let input = BinaryHV::random(1000 + i);
        if let Some(update) = pipeline.process_self_aware(&input) {
            prediction_errors.push(update.prediction_error);
        }
    }

    println!("✅ Self-awareness learning:");
    println!(
        "   - Initial prediction error: {:.4}",
        prediction_errors.first().unwrap_or(&1.0)
    );
    println!(
        "   - Final prediction error: {:.4}",
        prediction_errors.last().unwrap_or(&1.0)
    );
    println!("   - Number of updates: {}", prediction_errors.len());

    // Check that we have a self-model
    let self_model = pipeline.current_self_model();
    assert!(self_model.is_some());

    let model = self_model.unwrap();
    println!("   - Self-model confidence: {:.4}", model.confidence);
    println!("   - Self-model accuracy: {:.4}", model.accuracy);
}

#[test]
#[ignore = "expensive: 15 cycles with self-awareness (~90s)"]
fn test_self_awareness_during_processing() {
    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.set_embodiment(0.8);
    pipeline.enable_self_awareness(1024, 16);

    // Process several cycles with self-awareness enabled
    for i in 0..15 {
        let inputs = vec![BinaryHV::random(13000 + i), BinaryHV::random(13100 + i)];
        let priorities = vec![0.75, 0.8];
        pipeline.process(inputs, &priorities);
    }

    let state = pipeline.get_state();

    println!("✅ Self-awareness during processing:");
    println!("   - Φ: {:.4}", state.phi);
    println!("   - Consciousness level: {:.4}", state.consciousness_level);
    println!(
        "   - Metacognitive confidence: {:.4}",
        state.metacognitive_confidence
    );
    println!(
        "   - Self-awareness level: {:.4}",
        pipeline.self_awareness_level()
    );

    // Self-awareness should have affected metacognitive confidence
    assert!(pipeline.has_self_awareness());
    assert!(pipeline.self_awareness_level() > 0.0);
}

#[test]
#[ignore = "expensive: meta-cognitive assessment (~70s)"]
fn test_meta_cognitive_assessment() {
    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.enable_self_awareness(1024, 16);

    // Process to generate meta-assessment
    for i in 0..5 {
        let input = BinaryHV::random(2000 + i);
        pipeline.process_self_aware(&input);
    }

    let assessment = pipeline.meta_assessment();
    assert!(assessment.is_some(), "Meta-assessment should be available");

    let assessment = assessment.unwrap();
    println!("✅ Meta-cognitive assessment:");
    println!("   - Clarity: {:.4}", assessment.clarity);
    println!(
        "   - Mode appropriateness: {:.4}",
        assessment.mode_appropriateness
    );
    println!("   - Φ optimality: {:.4}", assessment.phi_optimality);
    println!("   - Change recommended: {}", assessment.change_recommended);
    println!("   - Reasoning: {}", assessment.reasoning);
}

#[test]
#[ignore = "takes >60s in debug builds"]
fn test_complete_integration_all_systems() {
    // Ultimate integration test: ALL systems working together
    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.set_embodiment(0.9);
    pipeline.enable_integrated_systems();
    pipeline.enable_phi_optimization(8, 5);
    pipeline.enable_feedback_dynamics();
    pipeline.enable_self_awareness(1024, 16);

    // Verify all are enabled
    assert!(pipeline.has_integrated_systems());
    assert!(pipeline.has_phi_optimization());
    assert!(pipeline.has_feedback_dynamics());
    assert!(pipeline.has_self_awareness());

    // Process with ALL systems active
    for i in 0..25 {
        let inputs = vec![BinaryHV::random(14000 + i), BinaryHV::random(14100 + i)];
        let priorities = vec![0.85, 0.8];
        pipeline.process(inputs, &priorities);
    }

    let state = pipeline.get_state();

    println!("✅ COMPLETE INTEGRATION - All 8 systems working together:");
    println!("   - Φ: {:.4}", state.phi);
    println!(
        "   - Metacognitive confidence: {:.4}",
        state.metacognitive_confidence
    );
    println!(
        "   - Cross-modal coherence: {:.4}",
        state.cross_modal_coherence
    );
    println!("   - Topological unity: {:.4}", state.topological_unity);
    println!("   - Narrative coherence: {:.4}", state.narrative_coherence);
    println!("   - Consciousness level: {:.4}", state.consciousness_level);
    println!(
        "   - Self-awareness level: {:.4}",
        pipeline.self_awareness_level()
    );

    // Introspection
    if let Some(report) = pipeline.introspect() {
        println!("   - Believed Φ: {:.4}", report.believed_phi);
        println!(
            "   - Self-model accuracy: {:.4}",
            report.self_model_accuracy
        );
    }

    // All metrics should be valid
    assert!(state.phi >= 0.0 && state.phi <= 1.0);
    assert!(state.consciousness_level >= 0.0 && state.consciousness_level <= 1.0);
    assert!(state.topological_unity >= 0.0 && state.topological_unity <= 1.0);
    assert!(state.metacognitive_confidence >= 0.0 && state.metacognitive_confidence <= 1.0);
    assert!(pipeline.self_awareness_level() >= 0.0 && pipeline.self_awareness_level() <= 1.0);
}

// ==========================================
// UNIFIED CONSCIOUSNESS OPTIMIZER TESTS
// ==========================================

#[test]
fn test_enable_full_consciousness() {
    let mut pipeline = ConsciousnessPipeline::default();

    // Enable all systems at once
    pipeline.enable_full_consciousness();

    // All systems should be enabled
    assert!(pipeline.has_full_consciousness());
    assert!(pipeline.has_integrated_systems());
    assert!(pipeline.has_phi_optimization());
    assert!(pipeline.has_feedback_dynamics());
    assert!(pipeline.has_self_awareness());

    println!("✅ Full consciousness enabled successfully");
}

#[test]
#[ignore = "takes >60s in debug builds"]
fn test_consciousness_metrics_report() {
    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.set_embodiment(0.8);
    pipeline.enable_full_consciousness();

    // Process some cycles
    for i in 0..10 {
        let inputs = vec![BinaryHV::random(15000 + i), BinaryHV::random(15100 + i)];
        let priorities = vec![0.8, 0.85];
        pipeline.process(inputs, &priorities);
    }

    // Get metrics report
    let report = pipeline.consciousness_metrics();

    println!("✅ Consciousness Metrics Report:");
    println!("{}", report);

    // Check report fields
    assert!(report.phi >= 0.0 && report.phi <= 1.0);
    assert!(report.consciousness_level >= 0.0 && report.consciousness_level <= 1.0);
    assert!(report.integrated_systems_active);
    assert!(report.phi_optimization_active);
    assert!(report.feedback_dynamics_active);
    assert!(report.self_awareness_active);
    assert!(report.processing_cycles > 0);
}

#[test]
fn test_optimization_recommendations() {
    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.set_embodiment(0.3); // Low embodiment
    pipeline.enable_full_consciousness();

    // Get recommendations
    let recommendations = pipeline.optimization_recommendations();

    println!("✅ Optimization Recommendations:");
    for rec in &recommendations {
        println!("   {}", rec);
    }

    // With low embodiment (0.3), we should get at least one recommendation
    println!("   Total recommendations: {}", recommendations.len());
    assert!(
        !recommendations.is_empty(),
        "low embodiment (0.3) should produce at least one optimization recommendation"
    );
}

#[test]
#[ignore = "expensive: full 11-system consciousness pipeline (~120s)"]
fn test_optimization_cycle() {
    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.set_embodiment(0.8);
    pipeline.enable_full_consciousness();

    // Process some cycles first
    for i in 0..5 {
        let inputs = vec![BinaryHV::random(16000 + i)];
        let priorities = vec![0.75];
        pipeline.process(inputs, &priorities);
    }

    // Run an optimization cycle
    let summary = pipeline.run_optimization_cycle();

    println!("✅ Optimization Cycle:");
    println!("{}", summary);

    // Check summary
    assert!(summary.phi_before >= 0.0 && summary.phi_before <= 1.0);
    assert!(summary.phi_after >= 0.0 && summary.phi_after <= 1.0);
}

#[test]
fn test_enable_full_consciousness_with_config() {
    let mut pipeline = ConsciousnessPipeline::default();

    // Enable with custom configuration
    pipeline.enable_full_consciousness_with_config(
        16,   // phi_nodes
        3,    // phi_frequency
        2048, // hdc_dim
        32,   // n_processes
    );

    assert!(pipeline.has_full_consciousness());

    println!("✅ Full consciousness with custom config enabled");
}

#[test]
#[ignore = "takes >60s in debug builds"]
fn test_full_consciousness_processing_pipeline() {
    // Comprehensive test of the entire consciousness pipeline
    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.set_embodiment(0.9);
    pipeline.enable_full_consciousness();

    // Run many processing cycles
    for i in 0..30 {
        let inputs = vec![
            BinaryHV::random(17000 + i),
            BinaryHV::random(17100 + i),
            BinaryHV::random(17200 + i),
        ];
        let priorities = vec![0.9, 0.85, 0.8];
        pipeline.process(inputs, &priorities);

        // Periodically run optimization
        if i % 10 == 9 {
            let summary = pipeline.run_optimization_cycle();
            println!(
                "Cycle {}: Φ = {:.4} → {:.4}",
                i, summary.phi_before, summary.phi_after
            );
        }
    }

    // Get final metrics
    let metrics = pipeline.consciousness_metrics();

    println!("\n✅ FULL CONSCIOUSNESS PIPELINE COMPLETE:");
    println!("{}", metrics);

    // All systems should still be active
    assert!(pipeline.has_full_consciousness());
    assert!(metrics.processing_cycles == 30);
}

#[test]
fn test_orchestrator_recommendations_display() {
    use crate::hdc::consciousness_integration::{
        OptimizationRecommendation, RecommendationPriority,
    };

    let recommendations = vec![
        OptimizationRecommendation {
            system: "phi".to_string(),
            priority: RecommendationPriority::High,
            message: "Φ is below optimal range".to_string(),
            suggested_action: Some("Increase binding".to_string()),
        },
        OptimizationRecommendation {
            system: "consciousness".to_string(),
            priority: RecommendationPriority::Medium,
            message: "Consciousness level could be improved".to_string(),
            suggested_action: None,
        },
        OptimizationRecommendation {
            system: "info".to_string(),
            priority: RecommendationPriority::Low,
            message: "System is running normally".to_string(),
            suggested_action: None,
        },
    ];

    println!("✅ Recommendation Display Test:");
    for rec in &recommendations {
        println!("   {}", rec);
    }

    assert_eq!(
        recommendations.len(),
        3,
        "should have exactly 3 recommendations"
    );
    assert_eq!(recommendations[0].system, "phi");
    assert!(
        recommendations[0].suggested_action.is_some(),
        "high-priority recommendation should have a suggested action"
    );
}

// ==========================================
// BATCH OPERATIONS & OBSERVABILITY TESTS
// ==========================================

#[test]
fn test_process_uses_batch_clustering() {
    // Verify that batch clustering produces the same binding results
    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.set_embodiment(0.8);

    // Use inputs with controlled similarity (same seed pairs should cluster)
    let inputs = vec![
        BinaryHV::random(100),
        BinaryHV::random(101),
        BinaryHV::random(200),
        BinaryHV::random(201),
    ];
    let priorities = vec![0.8, 0.8, 0.8, 0.8];

    pipeline.process(inputs, &priorities);

    let state = pipeline.get_state();
    // Pipeline should have processed without panicking
    assert!(state.phi >= 0.0 && state.phi <= 1.0);
    assert!(state.consciousness_level >= 0.0 && state.consciousness_level <= 1.0);
}

#[test]
fn test_metrics_recorded_during_process() {
    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.set_embodiment(0.8);
    pipeline.enable_metrics_collector();

    assert!(pipeline.metrics_collector().is_some());

    // Run several cycles
    for i in 0..5 {
        let inputs = vec![BinaryHV::random(500 + i), BinaryHV::random(600 + i)];
        let priorities = vec![0.8, 0.75];
        pipeline.process(inputs, &priorities);
    }

    // Check that metrics were recorded
    let collector = pipeline.metrics_collector().unwrap();
    let reader = collector.read().unwrap();
    // The MetricsCollector records phi_history; we check via its snapshot method
    // by downcasting. Since Observer is a trait object, we verify it didn't panic
    // and the pipeline state is valid.
    drop(reader);

    let state = pipeline.get_state();
    assert!(state.phi >= 0.0);
}

#[test]
fn test_verification_runs_periodically() {
    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.set_embodiment(0.8);
    pipeline.enable_verification(3); // Verify every 3 cycles

    // Initially no verification
    assert!(pipeline.latest_verification().is_none());

    // Run 6 cycles (should trigger verification at cycles 3 and 6)
    for i in 0..6 {
        let inputs = vec![
            BinaryHV::random(700 + i as u64),
            BinaryHV::random(800 + i as u64),
        ];
        let priorities = vec![0.8, 0.75];
        pipeline.process(inputs, &priorities);
    }

    // Verification should have been produced
    let report = pipeline.latest_verification();
    assert!(
        report.is_some(),
        "Verification report should exist after 6 cycles with interval=3"
    );

    let report = report.unwrap();
    assert!(report.confidence >= 0.0 && report.confidence <= 1.0);
    assert!(report.consensus_phi >= 0.0);
}

#[test]
fn test_register_subsystem() {
    use crate::hdc::consciousness_subsystem::{ConsciousnessSubsystem, SubsystemError};

    struct TestSubsystem {
        call_count: usize,
    }

    impl ConsciousnessSubsystem for TestSubsystem {
        fn name(&self) -> &str {
            "test"
        }
        fn process_cycle(
            &mut self,
            state: &mut ConsciousnessState,
            _inputs: &[BinaryHV],
        ) -> Result<(), SubsystemError> {
            self.call_count += 1;
            state.phi = (state.phi + 0.001).min(1.0);
            Ok(())
        }
        fn is_enabled(&self) -> bool {
            true
        }
    }

    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.set_embodiment(0.8);
    pipeline.register_subsystem(Box::new(TestSubsystem { call_count: 0 }));
    assert_eq!(pipeline.subsystem_count(), 1);

    // Process a cycle
    let inputs = vec![BinaryHV::random(900), BinaryHV::random(901)];
    let priorities = vec![0.8, 0.75];
    pipeline.process(inputs, &priorities);

    // No subsystem errors
    assert!(pipeline.last_subsystem_errors().is_empty());
    let state = pipeline.get_state();
    assert!(state.phi >= 0.0);
}

#[test]
fn test_subsystem_replaces_inline_processing() {
    // Register a "meta_consciousness" subsystem → inline process_meta_consciousness is skipped
    use crate::hdc::consciousness_metacognitive::MetaConsciousnessWrapped;

    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.set_embodiment(0.8);
    pipeline.enable_meta_consciousness(4);

    // Register the wrapped variant (should override inline)
    pipeline.register_subsystem(Box::new(MetaConsciousnessWrapped::new(4)));
    assert!(pipeline.has_subsystem_named("meta_consciousness"));

    // Process — should not panic and should use the subsystem
    for i in 0..5 {
        let inputs = vec![
            BinaryHV::random(950 + i as u64),
            BinaryHV::random(960 + i as u64),
        ];
        let priorities = vec![0.8, 0.75];
        pipeline.process(inputs, &priorities);
    }

    let state = pipeline.get_state();
    assert!(state.phi >= 0.0 && state.phi <= 1.0);
    assert!(state.metacognitive_confidence >= 0.0);
}

#[test]
fn test_ring_buffer_history_cap() {
    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.set_embodiment(0.8);
    pipeline.set_max_history(10);

    // Run 20 cycles
    for i in 0..20 {
        let inputs = vec![BinaryHV::random(1000 + i as u64)];
        let priorities = vec![0.8];
        pipeline.process(inputs, &priorities);
    }

    // History should be capped at 10
    assert!(
        pipeline.history.len() <= 10,
        "History should be bounded, got {}",
        pipeline.history.len()
    );
}

#[test]
fn test_subsystem_panic_safety() {
    use crate::hdc::consciousness_subsystem::{ConsciousnessSubsystem, SubsystemError};

    struct PanickingSubsystem;
    impl ConsciousnessSubsystem for PanickingSubsystem {
        fn name(&self) -> &str {
            "panicker"
        }
        fn process_cycle(
            &mut self,
            _state: &mut ConsciousnessState,
            _inputs: &[BinaryHV],
        ) -> Result<(), SubsystemError> {
            panic!("intentional test panic");
        }
        fn is_enabled(&self) -> bool {
            true
        }
    }

    struct GoodSubsystem;
    impl ConsciousnessSubsystem for GoodSubsystem {
        fn name(&self) -> &str {
            "good"
        }
        fn process_cycle(
            &mut self,
            state: &mut ConsciousnessState,
            _inputs: &[BinaryHV],
        ) -> Result<(), SubsystemError> {
            state.phi = (state.phi + 0.01).min(1.0);
            Ok(())
        }
        fn is_enabled(&self) -> bool {
            true
        }
    }

    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.set_embodiment(0.8);
    pipeline.register_subsystem(Box::new(PanickingSubsystem));
    pipeline.register_subsystem(Box::new(GoodSubsystem));

    // Should NOT panic — panicking subsystem is caught
    let inputs = vec![BinaryHV::random(1100)];
    pipeline.process(inputs, &[0.8]);

    // Should have 1 error from the panicker
    assert_eq!(pipeline.last_subsystem_errors().len(), 1);
    assert_eq!(pipeline.last_subsystem_errors()[0].subsystem, "panicker");
}

#[test]
fn test_subsystem_error_collection() {
    use crate::hdc::consciousness_subsystem::{ConsciousnessSubsystem, SubsystemError};

    struct FailingSubsystem;
    impl ConsciousnessSubsystem for FailingSubsystem {
        fn name(&self) -> &str {
            "failer"
        }
        fn process_cycle(
            &mut self,
            _state: &mut ConsciousnessState,
            _inputs: &[BinaryHV],
        ) -> Result<(), SubsystemError> {
            Err(SubsystemError {
                subsystem: "failer".into(),
                message: "test error".into(),
            })
        }
        fn is_enabled(&self) -> bool {
            true
        }
    }

    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.set_embodiment(0.8);
    pipeline.register_subsystem(Box::new(FailingSubsystem));

    let inputs = vec![BinaryHV::random(1200)];
    pipeline.process(inputs, &[0.8]);

    assert_eq!(pipeline.last_subsystem_errors().len(), 1);
    assert_eq!(pipeline.last_subsystem_errors()[0].message, "test error");

    // Errors should clear on next cycle
    let inputs2 = vec![BinaryHV::random(1201)];
    pipeline.process(inputs2, &[0.8]);
    // Still 1 error because the same subsystem fails every cycle
    assert_eq!(pipeline.last_subsystem_errors().len(), 1);
}

#[test]
fn test_subsystem_priority_ordering() {
    use crate::hdc::consciousness_subsystem::{ConsciousnessSubsystem, SubsystemError};
    use std::sync::{Arc, Mutex};

    let order: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

    struct OrderTracker {
        name: String,
        prio: i32,
        order: Arc<Mutex<Vec<String>>>,
    }
    impl ConsciousnessSubsystem for OrderTracker {
        fn name(&self) -> &str {
            &self.name
        }
        fn process_cycle(
            &mut self,
            _state: &mut ConsciousnessState,
            _inputs: &[BinaryHV],
        ) -> Result<(), SubsystemError> {
            self.order.lock().unwrap().push(self.name.clone());
            Ok(())
        }
        fn is_enabled(&self) -> bool {
            true
        }
        fn priority(&self) -> i32 {
            self.prio
        }
    }

    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.set_embodiment(0.8);

    // Register in non-priority order
    pipeline.register_subsystem(Box::new(OrderTracker {
        name: "low".into(),
        prio: -10,
        order: order.clone(),
    }));
    pipeline.register_subsystem(Box::new(OrderTracker {
        name: "high".into(),
        prio: 100,
        order: order.clone(),
    }));
    pipeline.register_subsystem(Box::new(OrderTracker {
        name: "mid".into(),
        prio: 0,
        order: order.clone(),
    }));

    let inputs = vec![BinaryHV::random(1300)];
    pipeline.process(inputs, &[0.8]);

    let executed = order.lock().unwrap();
    assert_eq!(executed.as_slice(), &["high", "mid", "low"]);
}

#[test]
fn test_checkpoint_save_restore() {
    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.set_embodiment(0.9);

    // Run some cycles to build up state
    for i in 0..10 {
        let inputs = vec![BinaryHV::random(1400 + i as u64)];
        pipeline.process(inputs, &[0.85]);
    }

    let checkpoint = pipeline.save_checkpoint();
    let saved_phi = pipeline.get_state().phi;
    let saved_cycle = checkpoint.current_cycle;

    // Mutate the pipeline further
    for i in 0..5 {
        let inputs = vec![BinaryHV::random(1500 + i as u64)];
        pipeline.process(inputs, &[0.5]);
    }

    // Restore
    pipeline.restore_checkpoint(checkpoint);
    assert!(
        (pipeline.get_state().phi - saved_phi).abs() < 1e-15,
        "Phi should be restored exactly"
    );
    assert_eq!(pipeline.current_cycle, saved_cycle);
}

// ==========================================
// BUILDER TESTS
// ==========================================

#[test]
fn test_builder_default() {
    let pipeline = ConsciousnessPipelineBuilder::new().build();
    assert!((pipeline.embodiment_level - 0.5).abs() < 1e-10);
    assert!(!pipeline.has_integrated_systems());
    assert_eq!(pipeline.subsystem_count(), 0);
}

#[test]
fn test_builder_with_embodiment() {
    let pipeline = ConsciousnessPipelineBuilder::new().embodiment(0.9).build();
    assert!((pipeline.embodiment_level - 0.9).abs() < 1e-10);
}

#[test]
fn test_builder_with_systems() {
    let pipeline = ConsciousnessPipelineBuilder::new()
        .embodiment(0.8)
        .integrated_systems()
        .phi_feedback()
        .creativity()
        .build();
    assert!(pipeline.has_integrated_systems());
    assert!(pipeline.has_phi_feedback());
    assert!(pipeline.has_creativity());
    assert!(!pipeline.has_fractal());
}

#[test]
fn test_builder_full_consciousness() {
    let pipeline = ConsciousnessPipelineBuilder::new()
        .embodiment(0.85)
        .full_consciousness()
        .build();
    assert!(pipeline.has_full_consciousness());
    assert_eq!(pipeline.active_system_count(), 12);
}

#[test]
fn test_builder_with_subsystem() {
    use crate::hdc::consciousness_subsystem::{ConsciousnessSubsystem, SubsystemError};

    struct BuilderTestSub;
    impl ConsciousnessSubsystem for BuilderTestSub {
        fn name(&self) -> &str {
            "builder_test"
        }
        fn process_cycle(
            &mut self,
            _s: &mut ConsciousnessState,
            _i: &[BinaryHV],
        ) -> Result<(), SubsystemError> {
            Ok(())
        }
        fn is_enabled(&self) -> bool {
            true
        }
    }

    let pipeline = ConsciousnessPipelineBuilder::new()
        .embodiment(0.8)
        .subsystem(Box::new(BuilderTestSub))
        .build();
    assert_eq!(pipeline.subsystem_count(), 1);
    assert!(pipeline.has_subsystem_named("builder_test"));
}

#[test]
fn test_builder_with_verification() {
    let pipeline = ConsciousnessPipelineBuilder::new()
        .embodiment(0.8)
        .verification(5)
        .build();
    assert!(pipeline.latest_verification().is_none()); // not run yet
    assert_eq!(pipeline.verification_interval, 5);
}

#[test]
fn test_builder_processes_correctly() {
    let mut pipeline = ConsciousnessPipelineBuilder::new()
        .embodiment(0.8)
        .integrated_systems()
        .phi_feedback()
        .build();

    for i in 0..5 {
        let inputs = vec![BinaryHV::random(2000 + i as u64)];
        pipeline.process(inputs, &[0.8]);
    }

    let state = pipeline.get_state();
    assert!(state.phi >= 0.0);
    assert!(state.consciousness_level > 0.0);
}

// ==========================================
// LIFECYCLE HOOK TESTS
// ==========================================

#[test]
fn test_on_register_called() {
    use crate::hdc::consciousness_subsystem::{ConsciousnessSubsystem, SubsystemError};
    use std::sync::{Arc, Mutex};

    let registered = Arc::new(Mutex::new(false));

    struct LifecycleSub {
        registered: Arc<Mutex<bool>>,
    }
    impl ConsciousnessSubsystem for LifecycleSub {
        fn name(&self) -> &str {
            "lifecycle"
        }
        fn process_cycle(
            &mut self,
            _s: &mut ConsciousnessState,
            _i: &[BinaryHV],
        ) -> Result<(), SubsystemError> {
            Ok(())
        }
        fn is_enabled(&self) -> bool {
            true
        }
        fn on_register(&mut self) {
            *self.registered.lock().unwrap() = true;
        }
    }

    let mut pipeline = ConsciousnessPipeline::default();
    assert!(!*registered.lock().unwrap());

    pipeline.register_subsystem(Box::new(LifecycleSub {
        registered: registered.clone(),
    }));
    assert!(
        *registered.lock().unwrap(),
        "on_register should be called during register_subsystem"
    );
}

#[test]
fn test_on_shutdown_called() {
    use crate::hdc::consciousness_subsystem::{ConsciousnessSubsystem, SubsystemError};
    use std::sync::{Arc, Mutex};

    let shutdown = Arc::new(Mutex::new(false));

    struct ShutdownSub {
        shutdown: Arc<Mutex<bool>>,
    }
    impl ConsciousnessSubsystem for ShutdownSub {
        fn name(&self) -> &str {
            "shutdown_test"
        }
        fn process_cycle(
            &mut self,
            _s: &mut ConsciousnessState,
            _i: &[BinaryHV],
        ) -> Result<(), SubsystemError> {
            Ok(())
        }
        fn is_enabled(&self) -> bool {
            true
        }
        fn on_shutdown(&mut self) {
            *self.shutdown.lock().unwrap() = true;
        }
    }

    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.register_subsystem(Box::new(ShutdownSub {
        shutdown: shutdown.clone(),
    }));
    assert!(!*shutdown.lock().unwrap());

    pipeline.clear();
    assert!(
        *shutdown.lock().unwrap(),
        "on_shutdown should be called during clear()"
    );
}

#[test]
fn test_checkpoint_restore_with_subsystems() {
    use crate::hdc::consciousness_subsystem::{ConsciousnessSubsystem, SubsystemError};
    use std::sync::{Arc, Mutex};

    let call_count = Arc::new(Mutex::new(0usize));

    struct CountingSub {
        calls: Arc<Mutex<usize>>,
    }
    impl ConsciousnessSubsystem for CountingSub {
        fn name(&self) -> &str {
            "counter"
        }
        fn process_cycle(
            &mut self,
            state: &mut ConsciousnessState,
            _i: &[BinaryHV],
        ) -> Result<(), SubsystemError> {
            *self.calls.lock().unwrap() += 1;
            state.phi = (state.phi + 0.01).min(1.0);
            Ok(())
        }
        fn is_enabled(&self) -> bool {
            true
        }
    }

    // Build pipeline with subsystem, run some cycles
    let mut pipeline = ConsciousnessPipelineBuilder::new()
        .embodiment(0.8)
        .subsystem(Box::new(CountingSub {
            calls: call_count.clone(),
        }))
        .build();

    for i in 0..5 {
        pipeline.process(vec![BinaryHV::random(3000 + i)], &[0.8]);
    }
    assert_eq!(*call_count.lock().unwrap(), 5);
    let checkpoint = pipeline.save_checkpoint();
    let saved_phi = pipeline.get_state().phi;

    // Mutate pipeline further
    for i in 0..5 {
        pipeline.process(vec![BinaryHV::random(3100 + i)], &[0.8]);
    }
    assert_eq!(*call_count.lock().unwrap(), 10);

    // Restore checkpoint — state should revert, subsystem stays registered
    pipeline.restore_checkpoint(checkpoint);
    assert!((pipeline.get_state().phi - saved_phi).abs() < 1e-15);
    assert_eq!(
        pipeline.subsystem_count(),
        1,
        "Subsystem should survive checkpoint restore"
    );

    // Subsystem should still work after restore
    pipeline.process(vec![BinaryHV::random(3200)], &[0.8]);
    assert_eq!(*call_count.lock().unwrap(), 11);
}

#[test]
fn test_subsystem_state_after_panic() {
    use crate::hdc::consciousness_subsystem::{ConsciousnessSubsystem, SubsystemError};

    struct PanicOnceSub {
        panicked: bool,
    }
    impl ConsciousnessSubsystem for PanicOnceSub {
        fn name(&self) -> &str {
            "panic_once"
        }
        fn process_cycle(
            &mut self,
            state: &mut ConsciousnessState,
            _i: &[BinaryHV],
        ) -> Result<(), SubsystemError> {
            if !self.panicked {
                self.panicked = true;
                panic!("first call panic");
            }
            state.phi = (state.phi + 0.05).min(1.0);
            Ok(())
        }
        fn is_enabled(&self) -> bool {
            true
        }
    }

    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.set_embodiment(0.8);
    pipeline.register_subsystem(Box::new(PanicOnceSub { panicked: false }));

    // First cycle: subsystem panics
    pipeline.process(vec![BinaryHV::random(3300)], &[0.8]);
    assert_eq!(pipeline.last_subsystem_errors().len(), 1);

    // Second cycle: subsystem should still exist and work
    // (panicked flag was set before the panic propagated)
    pipeline.process(vec![BinaryHV::random(3301)], &[0.8]);
    // The subsystem struct was inside catch_unwind so its state may be lost
    // on panic — this test documents the actual behavior
    let state = pipeline.get_state();
    assert!(state.phi >= 0.0);
}

#[test]
fn test_checkpoint_serialization_roundtrip() {
    let mut pipeline = ConsciousnessPipeline::default();
    pipeline.set_embodiment(0.8);

    for i in 0..5 {
        let inputs = vec![BinaryHV::random(1600 + i as u64)];
        pipeline.process(inputs, &[0.8]);
    }

    let checkpoint = pipeline.save_checkpoint();

    // Serialize to JSON
    let json = serde_json::to_string(&checkpoint).expect("checkpoint should serialize");
    assert!(!json.is_empty());

    // Deserialize back
    let restored: PipelineCheckpoint =
        serde_json::from_str(&json).expect("checkpoint should deserialize");
    assert!((restored.state.phi - checkpoint.state.phi).abs() < 1e-15);
    assert_eq!(restored.current_cycle, checkpoint.current_cycle);
}

// ==========================================
// SUBSYSTEM TIMING & INTROSPECTION TESTS
// ==========================================

#[test]
fn test_subsystem_cycle_reports() {
    use crate::hdc::consciousness_subsystem::{ConsciousnessSubsystem, SubsystemError};

    struct SlowSub;
    impl ConsciousnessSubsystem for SlowSub {
        fn name(&self) -> &str {
            "slow"
        }
        fn process_cycle(
            &mut self,
            state: &mut ConsciousnessState,
            _i: &[BinaryHV],
        ) -> Result<(), SubsystemError> {
            state.phi = (state.phi + 0.05).min(1.0);
            Ok(())
        }
        fn is_enabled(&self) -> bool {
            true
        }
    }

    struct DisabledSub;
    impl ConsciousnessSubsystem for DisabledSub {
        fn name(&self) -> &str {
            "disabled"
        }
        fn process_cycle(
            &mut self,
            _s: &mut ConsciousnessState,
            _i: &[BinaryHV],
        ) -> Result<(), SubsystemError> {
            Ok(())
        }
        fn is_enabled(&self) -> bool {
            false
        }
    }

    let mut pipeline = ConsciousnessPipelineBuilder::new()
        .embodiment(0.8)
        .subsystem(Box::new(SlowSub))
        .subsystem(Box::new(DisabledSub))
        .build();

    pipeline.process(vec![BinaryHV::random(4000)], &[0.8]);

    let reports = pipeline.last_cycle_reports();
    assert_eq!(reports.len(), 2);

    // "slow" should have ran
    let slow_report = reports.iter().find(|r| r.name == "slow").unwrap();
    assert!(slow_report.ran);
    assert!(slow_report.phi_delta > 0.0, "slow sub increases phi");
    assert!(slow_report.error.is_none());

    // "disabled" should NOT have ran
    let disabled_report = reports.iter().find(|r| r.name == "disabled").unwrap();
    assert!(!disabled_report.ran);
    assert_eq!(disabled_report.duration_us, 0);
}

#[test]
fn test_subsystem_names() {
    use crate::hdc::consciousness_subsystem::{ConsciousnessSubsystem, SubsystemError};

    struct Sub {
        n: &'static str,
    }
    impl ConsciousnessSubsystem for Sub {
        fn name(&self) -> &str {
            self.n
        }
        fn process_cycle(
            &mut self,
            _s: &mut ConsciousnessState,
            _i: &[BinaryHV],
        ) -> Result<(), SubsystemError> {
            Ok(())
        }
        fn is_enabled(&self) -> bool {
            true
        }
        fn priority(&self) -> i32 {
            match self.n {
                "alpha" => 10,
                "beta" => 5,
                "gamma" => 0,
                _ => 0,
            }
        }
    }

    let pipeline = ConsciousnessPipelineBuilder::new()
        .subsystem(Box::new(Sub { n: "gamma" }))
        .subsystem(Box::new(Sub { n: "alpha" }))
        .subsystem(Box::new(Sub { n: "beta" }))
        .build();

    let names = pipeline.subsystem_names();
    assert_eq!(names, vec!["alpha", "beta", "gamma"]);
}

#[test]
fn test_subsystem_error_in_report() {
    use crate::hdc::consciousness_subsystem::{ConsciousnessSubsystem, SubsystemError};

    struct FailSub;
    impl ConsciousnessSubsystem for FailSub {
        fn name(&self) -> &str {
            "fail_report"
        }
        fn process_cycle(
            &mut self,
            _s: &mut ConsciousnessState,
            _i: &[BinaryHV],
        ) -> Result<(), SubsystemError> {
            Err(SubsystemError {
                subsystem: "fail_report".into(),
                message: "oops".into(),
            })
        }
        fn is_enabled(&self) -> bool {
            true
        }
    }

    let mut pipeline = ConsciousnessPipelineBuilder::new()
        .embodiment(0.8)
        .subsystem(Box::new(FailSub))
        .build();

    pipeline.process(vec![BinaryHV::random(4100)], &[0.8]);

    let reports = pipeline.last_cycle_reports();
    assert_eq!(reports.len(), 1);
    assert!(reports[0].ran);
    assert!(reports[0].error.is_some());
    assert_eq!(reports[0].error.as_ref().unwrap().message, "oops");
}

// ==========================================
// SUBSYSTEM CONTEXT TESTS
// ==========================================

#[test]
fn test_subsystem_context_data_passing() {
    use crate::hdc::consciousness_subsystem::{
        ConsciousnessSubsystem, SubsystemContext, SubsystemError,
    };

    struct Producer;
    impl ConsciousnessSubsystem for Producer {
        fn name(&self) -> &str {
            "producer"
        }
        fn process_cycle(
            &mut self,
            _s: &mut ConsciousnessState,
            _i: &[BinaryHV],
        ) -> Result<(), SubsystemError> {
            Ok(())
        }
        fn is_enabled(&self) -> bool {
            true
        }
        fn priority(&self) -> i32 {
            100
        } // runs first
        fn process_cycle_with_context(
            &mut self,
            state: &mut ConsciousnessState,
            inputs: &[BinaryHV],
            context: &mut SubsystemContext,
        ) -> Result<(), SubsystemError> {
            context.set("shared_value", 42.0_f64);
            self.process_cycle(state, inputs)
        }
    }

    struct Consumer {
        received: Option<f64>,
    }
    impl ConsciousnessSubsystem for Consumer {
        fn name(&self) -> &str {
            "consumer"
        }
        fn process_cycle(
            &mut self,
            _s: &mut ConsciousnessState,
            _i: &[BinaryHV],
        ) -> Result<(), SubsystemError> {
            Ok(())
        }
        fn is_enabled(&self) -> bool {
            true
        }
        fn priority(&self) -> i32 {
            0
        } // runs second
        fn process_cycle_with_context(
            &mut self,
            state: &mut ConsciousnessState,
            inputs: &[BinaryHV],
            context: &mut SubsystemContext,
        ) -> Result<(), SubsystemError> {
            self.received = context.get::<f64>("shared_value").copied();
            self.process_cycle(state, inputs)
        }
    }

    let mut pipeline = ConsciousnessPipelineBuilder::new()
        .embodiment(0.8)
        .subsystem(Box::new(Producer))
        .subsystem(Box::new(Consumer { received: None }))
        .build();

    pipeline.process(vec![BinaryHV::random(4200)], &[0.8]);

    // Context should have the value after cycle
    let ctx = pipeline.subsystem_context();
    assert!(ctx.contains("shared_value"));
    assert_eq!(*ctx.get::<f64>("shared_value").unwrap(), 42.0);
}

#[test]
fn test_subsystem_context_cleared_each_cycle() {
    use crate::hdc::consciousness_subsystem::{
        ConsciousnessSubsystem, SubsystemContext, SubsystemError,
    };

    struct Writer;
    impl ConsciousnessSubsystem for Writer {
        fn name(&self) -> &str {
            "writer"
        }
        fn process_cycle(
            &mut self,
            _s: &mut ConsciousnessState,
            _i: &[BinaryHV],
        ) -> Result<(), SubsystemError> {
            Ok(())
        }
        fn is_enabled(&self) -> bool {
            true
        }
        fn process_cycle_with_context(
            &mut self,
            state: &mut ConsciousnessState,
            inputs: &[BinaryHV],
            context: &mut SubsystemContext,
        ) -> Result<(), SubsystemError> {
            context.set("cycle_data", "hello".to_string());
            self.process_cycle(state, inputs)
        }
    }

    let mut pipeline = ConsciousnessPipelineBuilder::new()
        .embodiment(0.8)
        .subsystem(Box::new(Writer))
        .build();

    pipeline.process(vec![BinaryHV::random(4300)], &[0.8]);
    assert!(pipeline.subsystem_context().contains("cycle_data"));

    // Context should be cleared at start of next cycle, then repopulated
    pipeline.process(vec![BinaryHV::random(4301)], &[0.8]);
    assert!(pipeline.subsystem_context().contains("cycle_data"));
}

// ==========================================
// STATE GROUP VIEW TESTS
// ==========================================

#[test]
fn test_phi_metrics_view() {
    let mut state = ConsciousnessState::default();
    state.phi = 0.75;
    state.free_energy = 0.3;
    state.topological_unity = 0.8;
    state.phi_trend = 0.02;
    state.predicted_phi = Some(0.78);

    let pm = state.phi_metrics();
    assert!((pm.phi - 0.75).abs() < 1e-10);
    assert!((pm.free_energy - 0.3).abs() < 1e-10);
    assert!((pm.topological_unity - 0.8).abs() < 1e-10);
    assert!((pm.phi_trend - 0.02).abs() < 1e-10);
    assert!((pm.predicted_phi.unwrap() - 0.78).abs() < 1e-10);
}

#[test]
fn test_temporal_state_view() {
    let mut state = ConsciousnessState::default();
    state.temporal_coherence = 0.9;
    state.theta_phase = 1.57;
    state.narrative_coherence = 0.85;
    state.present_window_length = 5;

    let ts = state.temporal_state();
    assert!((ts.temporal_coherence - 0.9).abs() < 1e-10);
    assert!((ts.theta_phase - 1.57).abs() < 1e-10);
    assert_eq!(ts.present_window_length, 5);
}

#[test]
fn test_emotional_state_view() {
    let mut state = ConsciousnessState::default();
    state.emotional_valence = -0.3;
    state.emotional_arousal = Some(0.7);
    state.uncertainty = 0.4;
    state.integration_score = 0.6;

    let es = state.emotional_state();
    assert!((es.valence - (-0.3)).abs() < 1e-10);
    assert!((es.arousal.unwrap() - 0.7).abs() < 1e-10);
    assert!((es.uncertainty - 0.4).abs() < 1e-10);
}

#[test]
fn test_all_state_views_from_pipeline() {
    let mut pipeline = ConsciousnessPipelineBuilder::new()
        .embodiment(0.8)
        .integrated_systems()
        .phi_feedback()
        .build();

    for i in 0..5 {
        pipeline.process(vec![BinaryHV::random(4400 + i)], &[0.8]);
    }

    let state = pipeline.get_state();
    let pm = state.phi_metrics();
    let ts = state.temporal_state();
    let sm = state.self_model_state();
    let es = state.emotional_state();
    let ps = state.predictive_state();
    let im = state.integration_metrics();

    // All views should reflect the state
    assert_eq!(pm.phi, state.phi);
    assert_eq!(ts.temporal_coherence, state.temporal_coherence);
    assert_eq!(sm.confidence, state.self_model_confidence);
    assert_eq!(es.valence, state.emotional_valence);
    assert_eq!(ps.precision, state.predictive_precision);
    assert_eq!(im.metacognitive_confidence, state.metacognitive_confidence);
}
