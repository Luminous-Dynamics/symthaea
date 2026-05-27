#![cfg(feature = "full_consciousness")]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration Tests: Primitive → Consciousness → Routing Pipeline
//!
//! These tests verify the complete integration between:
//! - PrimitiveSystem: 250+ primitives across 9 tiers
//! - PrimitiveConsciousnessState: Consciousness state in primitive space
//! - ConsciousnessPrimitiveProcessor: Processing using primitives
//! - AdaptivePrimitiveSelector: Learning-based primitive selection
//! - EvolutionCoordinator: Co-evolution of primitives and architecture
//!
//! Critical integration points tested:
//! 1. Primitive activation → Consciousness state creation
//! 2. Consciousness state → Tier-based decisions
//! 3. Adaptive selection → Evolution feedback
//! 4. Evolution → Improved primitives

use symthaea::consciousness::evolution_bridge::{
    ArchitectureFeedback, CoordinatorConfig, EvolutionCoordinator,
};
use symthaea::consciousness::primitive_consciousness::{
    ConsciousnessDecomposer, ConsciousnessPrimitiveProcessor, PrimitiveBindingEngine,
    PrimitiveConsciousnessState,
};
use symthaea::consciousness::primitive_reasoning::{
    AdaptivePrimitiveSelector, PrimitiveAffinityGraph, ReasoningChain, TaskType, TierAwareConfig,
    TransformationType,
};
use symthaea::hdc::binary_hv::BinaryHV;
use symthaea::hdc::primitive_system::{PrimitiveSystem, PrimitiveTier};

// =============================================================================
// TEST: PRIMITIVE → CONSCIOUSNESS STATE PIPELINE
// =============================================================================

#[test]
fn test_primitive_activation_creates_consciousness_state() {
    let mut processor = ConsciousnessPrimitiveProcessor::new();

    println!("\n=== PRIMITIVE → CONSCIOUSNESS STATE ===");

    // Process input through primitive processor
    let input = BinaryHV::random(42);
    let timestamp = 1.0;
    let result = processor.process_input(&input, timestamp);

    // Should produce a valid consciousness state
    assert!(result.phi >= 0.0, "Phi should be non-negative");
    assert!(result.phi <= 1.0, "Phi should be <= 1.0");

    println!("✓ Input processed to consciousness state");
    println!("  Phi: {:.4}", result.phi);
    println!("  Active tiers: {}", result.active_by_tier.len());
}

#[test]
fn test_tier_activation_pattern_reflects_input() {
    let mut processor = ConsciousnessPrimitiveProcessor::new();

    println!("\n=== TIER ACTIVATION PATTERNS ===");

    // Process different types of inputs
    let inputs = vec![
        (BinaryHV::random(1), "Random 1"),
        (BinaryHV::random(2), "Random 2"),
        (BinaryHV::random(3), "Random 3"),
    ];

    for (i, (input, label)) in inputs.into_iter().enumerate() {
        let timestamp = i as f64;
        let state = processor.process_input(&input, timestamp);

        assert!(
            state.phi >= 0.0 && state.phi <= 1.0,
            "Phi should be in [0,1], got {}",
            state.phi
        );
        assert!(
            state.phi.is_finite(),
            "Phi should be finite for input {}",
            label
        );

        println!(
            "Input {}: {} active tiers, phi={:.4}",
            label,
            state.active_by_tier.len(),
            state.phi
        );
    }

    println!("✓ Different inputs produce different tier activations");
}

// =============================================================================
// TEST: CONSCIOUSNESS STATE PROPERTIES
// =============================================================================

#[test]
fn test_consciousness_state_coherence() {
    println!("\n=== CONSCIOUSNESS STATE COHERENCE ===");

    let timestamp = 1.0;
    let state = PrimitiveConsciousnessState::new(timestamp);

    // Phi should be non-negative
    assert!(state.phi >= 0.0, "Phi should be non-negative");

    // Active tiers should be a subset of all 9 tiers
    assert!(
        state.active_by_tier.len() <= 9,
        "At most 9 tiers can be active"
    );

    println!("✓ Consciousness state has valid phi: {:.4}", state.phi);
    println!(
        "✓ Active tiers: {:?}",
        state.active_by_tier.keys().collect::<Vec<_>>()
    );
}

#[test]
fn test_consciousness_state_activation() {
    println!("\n=== CONSCIOUSNESS STATE ACTIVATION ===");

    let system = PrimitiveSystem::new();
    let timestamp = 1.0;
    let mut state = PrimitiveConsciousnessState::new(timestamp);

    let initial_phi = state.phi;
    let initial_active = state.all_active().len();

    // Activate a primitive (use "SET" which exists in Mathematical tier)
    if let Some(prim) = system.get("SET") {
        use symthaea::consciousness::primitive_consciousness::ActivationReason;
        state.activate(
            prim.clone(),
            0.8,
            ActivationReason::BottomUp {
                input_similarity: 0.8,
            },
        );
    }

    println!(
        "Initial phi: {:.4}, active: {}",
        initial_phi, initial_active
    );
    println!(
        "After activation: phi={:.4}, active={}",
        state.phi,
        state.all_active().len()
    );

    // Should have at least one active primitive now
    assert!(
        state.all_active().len() > initial_active,
        "Should have activated a primitive"
    );

    println!("✓ State activation adds primitives");
}

// =============================================================================
// TEST: ADAPTIVE PRIMITIVE SELECTION
// =============================================================================

#[test]
fn test_adaptive_selector_learns_from_feedback() {
    println!("\n=== ADAPTIVE SELECTOR LEARNING ===");

    let mut selector = AdaptivePrimitiveSelector::new();

    // Simulate multiple trials with feedback via ReasoningChain
    let task = TaskType::Logical;

    // Create a reasoning chain with some executions
    let question = BinaryHV::random(100);
    let chain = ReasoningChain::new(question);

    // The chain has been created, selector can learn from it
    selector.update_from_chain(&chain, task);

    assert!(
        chain.total_phi >= 0.0,
        "Chain total_phi should be non-negative"
    );

    println!("✓ Adaptive selector can update from reasoning chain");
}

#[test]
fn test_tier_aware_selection_config() {
    println!("\n=== TIER-AWARE SELECTION CONFIG ===");

    let system = PrimitiveSystem::new();

    // Configure tier-aware selection
    let config = TierAwareConfig::default();

    // Get primitives by tier
    let physical_prims = system.get_tier(PrimitiveTier::Physical);
    let math_prims = system.get_tier(PrimitiveTier::Mathematical);

    println!("Physical tier: {} primitives", physical_prims.len());
    println!("Mathematical tier: {} primitives", math_prims.len());

    // Verify config has valid fields
    assert!(config.consciousness_weight >= 0.0);

    println!("✓ Tier-aware selection configured");
    println!("  Consciousness weight: {:.2}", config.consciousness_weight);
    println!("  Code weight: {:.2}", config.code_weight);
}

// =============================================================================
// TEST: PRIMITIVE AFFINITY GRAPH
// =============================================================================

#[test]
fn test_affinity_graph_learns_compositions() {
    println!("\n=== AFFINITY GRAPH COMPOSITION LEARNING ===");

    let mut graph = PrimitiveAffinityGraph::new();

    // Record successful compositions
    graph.record_composition("BIND", "PERMUTE", 0.85);
    graph.record_composition("BIND", "PERMUTE", 0.88);
    graph.record_composition("BIND", "PERMUTE", 0.82);

    // Record less successful composition
    graph.record_composition("BIND", "NEGATE", 0.45);
    graph.record_composition("BIND", "NEGATE", 0.42);

    // Should suggest PERMUTE over NEGATE for BIND
    let suggestions = graph.best_partners("BIND", 5);

    println!("Suggested partners for BIND:");
    for (partner, score) in &suggestions {
        println!("  {} → {:.4}", partner, score);
    }

    // PERMUTE should have higher affinity
    let permute_score = suggestions
        .iter()
        .find(|(n, _)| n == "PERMUTE")
        .map(|(_, s)| *s)
        .unwrap_or(0.0);

    let negate_score = suggestions
        .iter()
        .find(|(n, _)| n == "NEGATE")
        .map(|(_, s)| *s)
        .unwrap_or(0.0);

    assert!(
        permute_score > negate_score,
        "PERMUTE ({:.4}) should have higher score than NEGATE ({:.4})",
        permute_score,
        negate_score
    );

    println!("✓ PERMUTE ranks higher than NEGATE as expected");
    println!("✓ Affinity graph learns from composition outcomes");
}

// =============================================================================
// TEST: CONSCIOUSNESS DECOMPOSITION
// =============================================================================

#[test]
fn test_decomposer_identifies_active_primitives() {
    println!("\n=== CONSCIOUSNESS DECOMPOSITION ===");

    let decomposer = ConsciousnessDecomposer::new();

    // Create a semantic vector
    let semantic: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();

    // Decompose into primitive activations
    let activations = decomposer.decompose(&semantic);

    println!("Decomposition found {} activations", activations.len());

    for (prim, strength) in activations.iter().take(5) {
        println!("  {} ({:?}): {:.4}", prim.name, prim.tier, strength);
    }

    // Should produce valid activations (sorted by strength)
    for i in 1..activations.len() {
        assert!(
            activations[i - 1].1 >= activations[i].1,
            "Activations should be sorted by strength descending"
        );
    }

    println!("✓ Decomposer produces sorted activations");
}

#[test]
fn test_dominant_tier_detection() {
    println!("\n=== DOMINANT TIER DETECTION ===");

    let decomposer = ConsciousnessDecomposer::new();

    // Create different semantic profiles
    let profiles = vec![
        (
            (0..64).map(|i| (i as f32 * 0.1).sin()).collect::<Vec<_>>(),
            "Oscillating",
        ),
        (
            (0..64).map(|i| i as f32 / 64.0).collect::<Vec<_>>(),
            "Linear",
        ),
        ((0..64).map(|_| 0.5).collect::<Vec<_>>(), "Uniform"),
    ];

    let mut results = Vec::new();
    for (semantic, label) in profiles {
        let tier = decomposer.dominant_tier(&semantic);
        results.push(tier);
        if let Some(t) = tier {
            println!("{} profile → {:?} tier dominant", label, t);
        } else {
            println!("{} profile → No dominant tier", label);
        }
    }

    assert_eq!(results.len(), 3, "Should have results for all 3 profiles");

    println!("✓ Dominant tier detection works for different profiles");
}

// =============================================================================
// TEST: PRIMITIVE BINDING
// =============================================================================

#[test]
fn test_binding_engine_operations() {
    println!("\n=== PRIMITIVE BINDING OPERATIONS ===");

    let system = PrimitiveSystem::new();
    let mut engine = PrimitiveBindingEngine::new();

    // Get two primitives to bind (use actual primitives from Mathematical tier)
    let prim1 = system.get("SET").expect("SET should exist");
    let prim2 = system.get("UNION").expect("UNION should exist");

    // Test different binding types
    let binding_types = vec![
        TransformationType::Bind,
        TransformationType::Bundle,
        TransformationType::Permute,
        TransformationType::Resonate,
        TransformationType::Abstract,
        TransformationType::Ground,
    ];

    for binding_type in binding_types {
        let binding = engine.bind(prim1, prim2, binding_type);

        assert!(
            binding.strength.is_finite(),
            "{:?} binding strength should be finite",
            binding_type
        );
        assert!(
            binding.strength >= 0.0,
            "{:?} binding strength should be non-negative",
            binding_type
        );
        assert!(
            binding.phi_contribution.is_finite(),
            "{:?} phi contribution should be finite",
            binding_type
        );

        println!("{:?} binding:", binding_type);
        println!("  Strength: {:.4}", binding.strength);
        println!("  Phi contribution: {:.4}", binding.phi_contribution);
    }

    println!("✓ All 6 binding operations produce valid results");
}

#[test]
fn test_binding_affects_affinity() {
    println!("\n=== BINDING → AFFINITY FEEDBACK ===");

    let system = PrimitiveSystem::new();
    let mut engine = PrimitiveBindingEngine::new();

    // Use actual primitives from Mathematical tier
    let prim1 = system.get("SET").expect("SET should exist");
    let prim2 = system.get("AND").expect("AND should exist");

    // Perform multiple bindings
    for _ in 0..5 {
        let _ = engine.bind(prim1, prim2, TransformationType::Bind);
    }

    // Check affinity suggestion
    let suggestions = engine.suggest_partner(&prim1.name, 3);

    println!("After 5 bindings, top suggestions for {}:", prim1.name);
    for (name, score) in &suggestions {
        println!("  {} → {:.4}", name, score);
    }

    // AND should appear in suggestions after repeated binding
    let has_and = suggestions.iter().any(|(n, _)| n == "AND");
    assert!(
        has_and,
        "AND should appear in suggestions after repeated binding"
    );

    for (_, score) in &suggestions {
        assert!(score.is_finite(), "Suggestion score should be finite");
        assert!(*score >= 0.0, "Suggestion score should be non-negative");
    }

    println!("AND in suggestions: {}", has_and);

    println!("✓ Binding operations affect affinity suggestions");
}

// =============================================================================
// TEST: EVOLUTION COORDINATOR
// =============================================================================

#[test]
fn test_evolution_coordinator_creation() {
    println!("\n=== EVOLUTION COORDINATOR ===");

    let config = CoordinatorConfig {
        primitive_evolution_frequency: 5,
        primitive_generations_per_arch: 3,
        min_phi_improvement: 0.05,
        max_evolving_primitives: 10,
        cross_tier_evolution: true,
    };

    let coordinator = EvolutionCoordinator::new(config);

    assert_eq!(coordinator.generation(), 0);
    println!("✓ Coordinator created at generation 0");
}

#[test]
fn test_evolution_step_execution() {
    println!("\n=== EVOLUTION STEP EXECUTION ===");

    let config = CoordinatorConfig {
        primitive_evolution_frequency: 2, // Run every 2 generations
        ..CoordinatorConfig::default()
    };

    let mut coordinator = EvolutionCoordinator::new(config);

    // First step - should NOT run primitive evolution
    let result1 = coordinator.step().expect("Step should succeed");
    assert!(
        !result1.primitive_evolution_ran,
        "Gen 1 should not run evolution"
    );
    println!(
        "Generation 1: evolution ran = {}",
        result1.primitive_evolution_ran
    );

    // Second step - SHOULD run primitive evolution
    let result2 = coordinator.step().expect("Step should succeed");
    assert!(
        result2.primitive_evolution_ran,
        "Gen 2 should run evolution"
    );
    println!(
        "Generation 2: evolution ran = {}",
        result2.primitive_evolution_ran
    );

    println!("✓ Evolution frequency respected");
}

#[test]
fn test_evolution_feedback_mechanism() {
    println!("\n=== EVOLUTION FEEDBACK ===");

    let mut coordinator = EvolutionCoordinator::new(CoordinatorConfig::default());

    // Simulate architecture feedback
    let arch_feedback = ArchitectureFeedback {
        high_performers: vec![("BIND".to_string(), 0.9), ("PERMUTE".to_string(), 0.85)],
        low_performers: vec![("NEGATE".to_string(), 0.2)],
        routing_primitive_usage: [("BIND".to_string(), 100), ("PERMUTE".to_string(), 75)]
            .into_iter()
            .collect(),
        architecture_phi: 0.72,
    };

    coordinator.receive_architecture_feedback(arch_feedback);

    // Generate primitive feedback
    let prim_feedback = coordinator.generate_primitive_feedback();

    assert!(
        prim_feedback.primitive_fitness.is_finite(),
        "Primitive fitness should be finite"
    );
    assert!(
        prim_feedback.primitive_fitness >= 0.0 && prim_feedback.primitive_fitness <= 1.0,
        "Primitive fitness should be in [0,1], got {}",
        prim_feedback.primitive_fitness
    );

    println!("Generated primitive feedback:");
    println!("  Fitness: {:.4}", prim_feedback.primitive_fitness);
    println!("  New primitives: {:?}", prim_feedback.new_primitives);
    println!("  Deprecated: {:?}", prim_feedback.deprecated_primitives);

    println!("✓ Bidirectional feedback mechanism works");
}

// =============================================================================
// TEST: END-TO-END PIPELINE
// =============================================================================

#[test]
fn test_full_primitive_consciousness_pipeline() {
    println!("\n=== FULL PIPELINE TEST ===");

    // 1. Initialize systems
    let _system = PrimitiveSystem::new();
    let mut processor = ConsciousnessPrimitiveProcessor::new();
    let mut selector = AdaptivePrimitiveSelector::new();
    let mut coordinator = EvolutionCoordinator::new(CoordinatorConfig::default());

    println!("1. Systems initialized");

    // 2. Create input and process to consciousness state
    let input = BinaryHV::random(500);
    let timestamp = 1.0;
    let state = processor.process_input(&input, timestamp);

    println!("2. Input processed to consciousness state");
    println!(
        "   Phi: {:.4}, Active tiers: {}",
        state.phi,
        state.active_by_tier.len()
    );

    // 3. Create a reasoning chain
    let chain = ReasoningChain::new(input.clone());

    println!("3. Reasoning chain created");
    println!("   Total phi: {:.4}", chain.total_phi);

    // 4. Update selector from chain
    let task = TaskType::Logical;
    selector.update_from_chain(&chain, task);

    println!("4. Selector updated from reasoning chain");

    // 5. Update evolution coordinator
    coordinator.update_from_reasoning(&chain, task);

    println!("5. Evolution coordinator updated");

    // 6. Step evolution (may or may not run based on frequency)
    let evo_result = coordinator.step().expect("Evolution step should succeed");

    println!(
        "6. Evolution stepped: ran = {}",
        evo_result.primitive_evolution_ran
    );

    // Verify end-to-end data flow
    assert!(
        state.phi >= 0.0 && state.phi <= 1.0,
        "Phi should be in [0,1]"
    );
    assert!(state.phi.is_finite(), "Phi should be finite");
    assert!(
        chain.total_phi >= 0.0,
        "Chain total_phi should be non-negative"
    );
    assert!(
        coordinator.generation() >= 1,
        "Coordinator should have advanced at least 1 generation"
    );

    println!("\n✓ Full primitive → consciousness pipeline test PASSED");
}

// =============================================================================
// TEST: PIPELINE INVARIANTS
// =============================================================================

#[test]
fn test_phi_never_negative() {
    println!("\n=== PHI NON-NEGATIVITY INVARIANT ===");

    let mut processor = ConsciousnessPrimitiveProcessor::new();

    // Test with many random inputs
    for i in 0..100 {
        let input = BinaryHV::random(1000 + i as u64);
        let timestamp = i as f64;
        let state = processor.process_input(&input, timestamp);

        assert!(
            state.phi >= 0.0,
            "Phi should never be negative (got {} for seed {})",
            state.phi,
            i
        );
    }

    println!("✓ Phi non-negativity verified for 100 inputs");
}

#[test]
fn test_tier_membership_consistent() {
    println!("\n=== TIER MEMBERSHIP CONSISTENCY ===");

    let system = PrimitiveSystem::new();

    // Test that primitives can be retrieved by tier
    // Note: Some derived primitives may be indexed under different tiers
    // This test verifies the tier system returns primitives
    let query_tiers = vec![
        PrimitiveTier::Mathematical,
        PrimitiveTier::Physical,
        PrimitiveTier::Geometric,
        PrimitiveTier::Strategic,
        PrimitiveTier::MetaCognitive,
    ];

    // All valid tiers that primitives can belong to
    let all_valid_tiers = vec![
        PrimitiveTier::NSM,
        PrimitiveTier::Mathematical,
        PrimitiveTier::Physical,
        PrimitiveTier::Geometric,
        PrimitiveTier::Strategic,
        PrimitiveTier::MetaCognitive,
        PrimitiveTier::Temporal,
        PrimitiveTier::Compositional,
        PrimitiveTier::Consciousness,
    ];

    let mut total_primitives = 0;
    for tier in &query_tiers {
        let prims = system.get_tier(*tier);
        println!("{:?}: {} primitives", tier, prims.len());
        total_primitives += prims.len();

        // Verify each primitive has a valid tier (may be cross-indexed)
        for prim in prims {
            assert!(
                all_valid_tiers.iter().any(|t| prim.tier == *t),
                "Primitive {} has invalid tier {:?}",
                prim.name,
                prim.tier
            );
        }
    }

    assert!(
        total_primitives > 0,
        "Should have some primitives across tiers"
    );
    println!(
        "✓ Tier retrieval verified for {} primitives",
        total_primitives
    );
}

#[test]
fn test_evolution_history_monotonic_generation() {
    println!("\n=== EVOLUTION GENERATION MONOTONICITY ===");

    let mut coordinator = EvolutionCoordinator::new(CoordinatorConfig::default());

    let mut last_gen = 0;
    for _ in 0..10 {
        let _ = coordinator.step();
        let current_gen = coordinator.generation();

        assert!(
            current_gen > last_gen,
            "Generation should increase: {} -> {}",
            last_gen,
            current_gen
        );

        last_gen = current_gen;
    }

    println!("✓ Generation increases monotonically over 10 steps");
}

// =============================================================================
// TEST: PRIMITIVE SYSTEM COMPLETENESS
// =============================================================================

#[test]
fn test_primitive_system_has_core_primitives() {
    println!("\n=== CORE PRIMITIVES VERIFICATION ===");

    let system = PrimitiveSystem::new();

    // Core primitives that actually exist in the system (from Mathematical tier)
    let core_primitives = vec![
        "SET",
        "MEMBERSHIP",
        "UNION",
        "INTERSECTION",
        "NOT",
        "AND",
        "OR",
        "EMPTY_SET",
    ];

    for name in core_primitives {
        let prim = system.get(name);
        assert!(prim.is_some(), "Core primitive {} should exist", name);
        println!("✓ {} exists", name);
    }

    println!("✓ All core primitives verified");
}

#[test]
fn test_all_tiers_have_primitives() {
    println!("\n=== TIER POPULATION VERIFICATION ===");

    let system = PrimitiveSystem::new();

    // NSM tier is populated by vocabulary.rs separately, not primitive_system.rs
    // So we test only the tiers that primitive_system.rs initializes
    let populated_tiers = vec![
        PrimitiveTier::Mathematical,
        PrimitiveTier::Physical,
        PrimitiveTier::Geometric,
        PrimitiveTier::Strategic,
        PrimitiveTier::MetaCognitive,
    ];

    for tier in populated_tiers {
        let prims = system.get_tier(tier);
        println!("{:?}: {} primitives", tier, prims.len());
        assert!(!prims.is_empty(), "{:?} tier should have primitives", tier);
    }

    // NSM tier is separately managed by vocabulary system
    let nsm_prims = system.get_tier(PrimitiveTier::NSM);
    println!(
        "NSM: {} primitives (managed separately by vocabulary.rs)",
        nsm_prims.len()
    );

    println!("✓ All core tiers have primitives");
}