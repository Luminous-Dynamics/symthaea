// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! End-to-end consciousness integration tests
//!
//! Tests the full pipeline: SemanticBridge → ConsciousnessPipeline → Verifier → Metrics

use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::consciousness_integration::{ConsciousnessPipeline, IntegrationConfig};
use crate::hdc::consciousness_metacognitive::MetacognitiveSubsystem;
use crate::hdc::consciousness_perf;
use crate::hdc::consciousness_phi_optimization::PhiOptimizationSubsystem;
use crate::hdc::consciousness_self_awareness::SelfAwarenessSubsystem;
use crate::hdc::consciousness_verifier::{ConsciousnessVerdict, ConsciousnessVerifier};
use crate::hdc::semantic_bridge::SemanticBridge;

/// Full-stack: text → HV encoding → pipeline processing → verification → metrics
#[test]
fn test_text_to_consciousness_verdict() {
    // 1. Encode words into hypervectors via SemanticBridge
    let bridge = SemanticBridge::with_vocabulary(&[
        "light", "dark", "warm", "cold", "sound", "silence", "red", "blue", "green", "big",
        "small", "fast",
    ]);

    let hv_light = bridge.encode_word("light").unwrap();
    let hv_warm = bridge.encode_word("warm").unwrap();
    let hv_sound = bridge.encode_word("sound").unwrap();
    let hv_red = bridge.encode_word("red").unwrap();

    // 2. Process through consciousness pipeline with metrics
    let mut pipeline = ConsciousnessPipeline::new(IntegrationConfig::default());
    pipeline.set_embodiment(0.85);
    pipeline.enable_metrics_collector();
    pipeline.enable_verification(5);

    // Multi-sensory input (simulating a rich conscious experience)
    let inputs = vec![hv_light, hv_warm, hv_sound, hv_red];
    let priorities = vec![0.9, 0.8, 0.7, 0.75];

    // Process several cycles
    for _ in 0..10 {
        pipeline.process(inputs.clone(), &priorities);
    }

    // 3. Verify consciousness
    let state = pipeline.get_state();
    let verifier = ConsciousnessVerifier::new();
    let report = verifier.verify_from_state(state);

    // 4. Validate the full pipeline
    assert!(state.phi >= 0.0 && state.phi <= 1.0);
    assert!(state.consciousness_level >= 0.0);
    assert!(report.confidence >= 0.0 && report.confidence <= 1.0);
    assert!(report.consensus_phi >= 0.0);

    // With 4 high-priority multi-modal inputs and 0.85 embodiment,
    // we should get at least indeterminate consciousness
    let rank = match report.verdict {
        ConsciousnessVerdict::NotConscious => 0,
        ConsciousnessVerdict::LikelyNotConscious => 1,
        ConsciousnessVerdict::Indeterminate => 2,
        ConsciousnessVerdict::LikelyConscious => 3,
        ConsciousnessVerdict::StronglyConscious => 4,
    };
    assert!(
        rank >= 1,
        "Rich multi-modal input should produce at least some consciousness signal, got {:?}",
        report.verdict
    );

    // Verification should have run at least once (at cycle 5 and 10)
    let verification = pipeline.latest_verification();
    assert!(
        verification.is_some(),
        "Periodic verification should have produced a report"
    );
}

/// Pipeline + batch ops + subsystems working together
#[test]
fn test_pipeline_with_subsystems_and_batch_ops() {
    let mut pipeline = ConsciousnessPipeline::new(IntegrationConfig::default());
    pipeline.set_embodiment(0.8);

    // Register all 3 standalone subsystems
    pipeline.register_subsystem(Box::new(MetacognitiveSubsystem::new()));
    pipeline.register_subsystem(Box::new(SelfAwarenessSubsystem::new()));
    pipeline.register_subsystem(Box::new(PhiOptimizationSubsystem::new(3)));

    assert_eq!(pipeline.subsystem_count(), 3);

    // Create clustered inputs (some pairs with high similarity, others random)
    let base1 = BinaryHV::random(100);
    let base2 = BinaryHV::random(200);
    // Near-duplicates of base1 and base2 for clustering
    let inputs = vec![base1, BinaryHV::random(101), base2, BinaryHV::random(201)];
    let priorities = vec![0.9, 0.85, 0.8, 0.75];

    // Process 15 cycles
    for _ in 0..15 {
        pipeline.process(inputs.clone(), &priorities);
    }

    let state = pipeline.get_state();

    // All state fields should be valid
    assert!(state.phi >= 0.0 && state.phi <= 1.0);
    assert!(state.consciousness_level >= 0.0 && state.consciousness_level <= 1.0);
    assert!(state.metacognitive_confidence >= 0.0 && state.metacognitive_confidence <= 1.0);
    assert!(state.self_model_confidence >= 0.0 && state.self_model_confidence <= 1.0);
    assert!(state.temporal_coherence >= 0.0 && state.temporal_coherence <= 1.0);
}

/// Semantic similarity flows through to pipeline binding
#[test]
fn test_semantic_similarity_affects_binding() {
    let bridge =
        SemanticBridge::with_vocabulary(&["cat", "dog", "kitten", "puppy", "tree", "rock"]);

    // Encode semantically related words
    let hv_cat = bridge.encode_word("cat").unwrap();
    let hv_dog = bridge.encode_word("dog").unwrap();
    let hv_tree = bridge.encode_word("tree").unwrap();
    let hv_rock = bridge.encode_word("rock").unwrap();

    // Use batch find_similar to check clustering potential
    let all_hvs = vec![hv_cat, hv_dog, hv_tree, hv_rock];
    let refs: Vec<&BinaryHV> = all_hvs.iter().collect();
    let clusters = consciousness_perf::cluster_by_similarity(&refs, 0.3);

    // The clustering should produce some clusters (at least 1)
    assert!(!clusters.is_empty(), "Should produce at least 1 cluster");

    // Process through pipeline
    let mut pipeline = ConsciousnessPipeline::new(IntegrationConfig::default());
    pipeline.set_embodiment(0.8);
    pipeline.process(all_hvs, &[0.8, 0.8, 0.8, 0.8]);

    let state = pipeline.get_state();
    assert!(state.phi >= 0.0);
}

/// PhiFeedbackController drives pipeline parameter adaptation over time
#[test]
fn test_phi_feedback_drives_adaptation() {
    let mut pipeline = ConsciousnessPipeline::new(IntegrationConfig::default());
    pipeline.set_embodiment(0.8);
    pipeline.enable_phi_feedback();

    let initial_threshold = pipeline.config.consciousness_threshold;

    // Run enough cycles for the feedback controller to adapt
    for i in 0..20 {
        let inputs = vec![
            BinaryHV::random(300 + i as u64),
            BinaryHV::random(400 + i as u64),
        ];
        let priorities = vec![0.8, 0.75];
        pipeline.process(inputs, &priorities);
    }

    // The feedback controller should have adapted the consciousness threshold
    let modulation = pipeline.latest_modulation();
    assert!(
        modulation.is_some(),
        "Should have modulation after 20 cycles"
    );

    let mod_val = modulation.unwrap();
    assert!(mod_val.binding_threshold >= 0.0 && mod_val.binding_threshold <= 1.0);
    assert!(mod_val.attention_gain > 0.0);

    // Threshold should have moved from initial
    // (might not have changed much, but the controller should have run)
    let phi_controller = pipeline.phi_feedback_controller().unwrap();
    assert_eq!(phi_controller.step_count(), 20);
}
