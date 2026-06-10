// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Tests for integrated conscious agent

use super::super::unified_hv::ContinuousHV;
use super::*;

#[test]
fn test_integrated_agent_creation() {
    let agent = IntegratedConsciousAgent::new(AgentConfig::default());
    let intro = agent.introspect();
    println!("{}", intro);
    assert!(
        intro.believed_phi >= 0.0,
        "initial phi should be non-negative, got {}",
        intro.believed_phi
    );
    assert!(
        intro.self_awareness_level >= 0.0 && intro.self_awareness_level <= 1.0,
        "self_awareness_level should be in [0,1], got {}",
        intro.self_awareness_level
    );
}

#[test]
fn test_integrated_processing() {
    let config = AgentConfig {
        dim: 1024,
        n_processes: 16,
        ..Default::default()
    };

    let mut agent = IntegratedConsciousAgent::new(config);

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("        INTEGRATED CONSCIOUS AGENT - PROCESSING TEST");
    println!("═══════════════════════════════════════════════════════════════\n");

    for i in 0..20 {
        let input = ContinuousHV::random(1024, i * 100);
        let update = agent.process(&input);

        if i % 4 == 0 {
            println!("{}", update);
        }
    }

    let intro = agent.introspect();
    println!("\n{}", intro);
    assert!(
        intro.believed_phi >= 0.0,
        "phi should be non-negative after processing, got {}",
        intro.believed_phi
    );
    assert!(
        intro.integration_quality >= 0.0,
        "integration_quality should be non-negative, got {}",
        intro.integration_quality
    );
}

#[test]
fn test_goal_directed_attention() {
    let config = AgentConfig {
        dim: 1024,
        n_processes: 16,
        self_directed_attention: true,
        ..Default::default()
    };

    let mut agent = IntegratedConsciousAgent::new(config);

    // Add a goal
    let goal_target = ContinuousHV::random(1024, 999);
    agent.add_goal("find_pattern", goal_target.clone(), 0.9);

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("        GOAL-DIRECTED ATTENTION TEST");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Process inputs, some similar to goal, some not
    for i in 0..15 {
        let input = if i % 3 == 0 {
            // Similar to goal
            goal_target
                .add(&ContinuousHV::random(1024, i * 100).scale(0.2))
                .normalize()
        } else {
            // Random
            ContinuousHV::random(1024, i * 100)
        };

        let update = agent.process(&input);

        let goal_match = if i % 3 == 0 { "[GOAL MATCH]" } else { "" };
        println!(
            "Step {}: Φ={:.4}, attention={:?}, self_directed={} {}",
            update.step,
            update.phi,
            update.attention.mode,
            update.attention.self_directed,
            goal_match
        );
    }

    let intro = agent.introspect();
    println!("\n{}", intro);
    assert!(
        intro.num_active_goals >= 1,
        "should have at least 1 active goal, got {}",
        intro.num_active_goals
    );
}

#[test]
fn test_stream_continuity() {
    let config = AgentConfig {
        dim: 1024,
        n_processes: 16,
        attention_binding_coupling: 0.8,
        ..Default::default()
    };

    let mut agent = IntegratedConsciousAgent::new(config);

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("        STREAM OF CONSCIOUSNESS CONTINUITY TEST");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Create continuous experience (similar inputs)
    let base = ContinuousHV::random(1024, 42);

    for i in 0..20 {
        let noise = ContinuousHV::random(1024, i * 100).scale(0.15);
        let input = base.add(&noise).normalize();

        let update = agent.process(&input);

        println!(
            "Step {}: stream_coherence={:.3}, continuity={:.3}, flowing={}",
            update.step,
            update.temporal.stream_coherence,
            update.temporal.continuity,
            update.temporal.is_flowing
        );
    }

    let health = agent.stream_health();
    println!(
        "\nFinal stream health: coherence={:.3}, flowing={}",
        health.coherence, health.is_flowing
    );
    assert!(
        health.coherence >= 0.0,
        "stream coherence should be non-negative, got {}",
        health.coherence
    );
    assert!(
        health.narrative_length > 0,
        "narrative should have accumulated entries, got {}",
        health.narrative_length
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// NEW TESTS FOR ENHANCED FEATURES
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_working_memory_capacity() {
    let mut wm = WorkingMemory::new(7); // Miller's number

    // Add items up to capacity
    for i in 0..7 {
        let content = ContinuousHV::random(1024, i as u64);
        wm.add_to_episodic(content, MemorySource::Perception, 0.5, i);
    }

    assert_eq!(wm.episodic_buffer.len(), 7);
    assert!(!wm.is_overloaded());

    // Add one more - should trigger removal of lowest activation item
    let overflow = ContinuousHV::random(1024, 100);
    wm.add_to_episodic(overflow, MemorySource::Perception, 0.5, 7);

    assert_eq!(wm.episodic_buffer.len(), 7); // Still at capacity
    println!(
        "Working memory capacity test passed: maintains {} items",
        wm.episodic_buffer.len()
    );
}

#[test]
fn test_working_memory_decay() {
    let mut wm = WorkingMemory::new(7);

    let content = ContinuousHV::random(1024, 42);
    wm.add_to_episodic(content.clone(), MemorySource::Perception, 0.8, 0);

    let initial_activation = wm.episodic_buffer.back().unwrap().activation;
    assert_eq!(initial_activation, 1.0);

    // Update without focus - should decay
    for _ in 0..5 {
        wm.update(None);
    }

    let decayed_activation = wm.episodic_buffer.back().unwrap().activation;
    assert!(decayed_activation < initial_activation);
    println!(
        "Decay test: activation {} -> {}",
        initial_activation, decayed_activation
    );
}

#[test]
fn test_working_memory_rehearsal() {
    let mut wm = WorkingMemory::new(7);

    let content = ContinuousHV::random(1024, 42);
    wm.add_to_episodic(content.clone(), MemorySource::Perception, 0.8, 0);

    // Let it decay first
    for _ in 0..3 {
        wm.update(None);
    }
    let before_rehearsal = wm.episodic_buffer.back().unwrap().activation;

    // Now rehearse by focusing on similar content
    wm.update(Some(&content));
    let after_rehearsal = wm.episodic_buffer.back().unwrap().activation;

    // Rehearsal should boost or maintain activation
    assert!(after_rehearsal >= before_rehearsal * 0.9); // Allow small margin
    println!(
        "Rehearsal test: {} -> {}",
        before_rehearsal, after_rehearsal
    );
}

#[test]
fn test_emotional_state_valence_arousal() {
    let mut es = EmotionalState::new();

    // Initial state should be neutral
    assert_eq!(es.valence, 0.0);
    assert!(es.arousal > 0.0 && es.arousal < 1.0);

    // High Φ, low prediction error, good goal progress = positive
    es.update(0.8, 0.1, 0.9);
    assert!(
        es.valence > 0.0,
        "High success should lead to positive valence"
    );

    // Low Φ, high prediction error, poor goal progress = negative
    let mut es2 = EmotionalState::new();
    es2.update(0.2, 0.9, 0.1);
    assert!(
        es2.valence <= 0.0,
        "Poor performance should not increase valence"
    );

    println!(
        "Emotional valence test: positive={:.2}, negative={:.2}",
        es.valence, es2.valence
    );
}

#[test]
fn test_emotional_state_labels() {
    let mut es = EmotionalState::new();

    // Test all four quadrants
    es.valence = 0.5;
    es.arousal = 0.7;
    assert_eq!(es.label(), "excited/happy");

    es.valence = 0.5;
    es.arousal = 0.3;
    assert_eq!(es.label(), "calm/content");

    es.valence = -0.5;
    es.arousal = 0.7;
    assert_eq!(es.label(), "stressed/anxious");

    es.valence = -0.5;
    es.arousal = 0.3;
    assert_eq!(es.label(), "sad/bored");

    println!("Emotional label test: all quadrants verified");
}

#[test]
fn test_emotional_stability() {
    let mut es = EmotionalState::new();

    // Build up some history with consistent emotions
    for _ in 0..10 {
        es.update(0.6, 0.2, 0.7); // Consistent good performance
    }

    let stability = es.stability();
    assert!(stability > 0.5, "Consistent emotions should be stable");
    println!("Stability test: {:.2} (should be high)", stability);
}

#[test]
fn test_qualia_texture_creation() {
    let qualia = QualiaTexture::new(0.5, 0.7, 0.6, 0.8, 0.9);

    assert_eq!(qualia.warmth, 0.5);
    assert_eq!(qualia.depth, 0.7);
    assert_eq!(qualia.spaciousness, 0.6);
    assert_eq!(qualia.flow, 0.8);
    assert_eq!(qualia.presence, 0.9);

    println!("Qualia texture: {}", qualia);
}

#[test]
fn test_qualia_texture_description() {
    // Warm, profound, expansive
    let warm_deep = QualiaTexture::new(0.6, 0.8, 0.8, 0.5, 0.5);
    let desc = warm_deep.describe();
    assert!(desc.contains("warm"));
    assert!(desc.contains("profound"));
    assert!(desc.contains("expansive"));

    // Cool, surface, intimate
    let cool_surface = QualiaTexture::new(-0.6, 0.2, 0.2, 0.5, 0.5);
    let desc2 = cool_surface.describe();
    assert!(desc2.contains("cool"));
    assert!(desc2.contains("surface"));
    assert!(desc2.contains("intimate"));

    println!("Qualia descriptions: \"{}\" and \"{}\"", desc, desc2);
}

#[test]
fn test_qualia_clamping() {
    // Test that values are properly clamped
    let qualia = QualiaTexture::new(2.0, 5.0, -3.0, 10.0, -1.0);

    assert_eq!(qualia.warmth, 1.0); // Clamped from 2.0
    assert_eq!(qualia.depth, 1.0); // Clamped from 5.0
    assert_eq!(qualia.spaciousness, 0.0); // Clamped from -3.0
    assert_eq!(qualia.flow, 1.0); // Clamped from 10.0
    assert_eq!(qualia.presence, 0.0); // Clamped from -1.0

    println!("Qualia clamping test passed");
}

#[test]
fn test_phenomenal_content_display() {
    let config = AgentConfig {
        dim: 1024,
        n_processes: 16,
        ..Default::default()
    };

    let mut agent = IntegratedConsciousAgent::new(config);

    // Process a few inputs to build up state
    for i in 0..5 {
        let input = ContinuousHV::random(1024, i * 100);
        agent.process(&input);
    }

    // Get latest phenomenal content and display it
    if let Some(content) = agent.latest_phenomenal_content() {
        println!("\n{}", content);

        // Verify new fields exist
        assert!(content.arousal >= 0.0 && content.arousal <= 1.0);
        assert!(content.groundedness >= 0.0 && content.groundedness <= 1.0);
        assert!(content.cognitive_load >= 0.0 && content.cognitive_load <= 1.0);
    }
}

#[test]
fn test_integrated_agent_with_new_features() {
    let config = AgentConfig {
        dim: 1024,
        n_processes: 16,
        self_directed_attention: true,
        phi_guided: true,
        ..Default::default()
    };

    let mut agent = IntegratedConsciousAgent::new(config);

    // Add goals for warmth
    let goal = ContinuousHV::random(1024, 42);
    agent.add_goal("test_goal", goal.clone(), 0.9);

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("     INTEGRATED TEST: Working Memory + Emotions + Qualia");
    println!("═══════════════════════════════════════════════════════════════\n");

    for i in 0..15 {
        let input = if i % 3 == 0 {
            // Goal-relevant input
            goal.add(&ContinuousHV::random(1024, i as u64 * 100).scale(0.2))
                .normalize()
        } else {
            ContinuousHV::random(1024, i as u64 * 100)
        };

        let update = agent.process(&input);

        if i % 3 == 0 || i == 14 {
            println!(
                "Step {}: {}",
                update.step, update.phenomenal_content.description
            );
            println!("  Qualia: {}", update.phenomenal_content.qualia_texture);
            println!(
                "  Arousal: {:.0}% | Groundedness: {:.0}% | Cognitive Load: {:.0}%",
                update.phenomenal_content.arousal * 100.0,
                update.phenomenal_content.groundedness * 100.0,
                update.phenomenal_content.cognitive_load * 100.0
            );
            println!();
        }
    }

    // Check working memory state
    let wm = agent.working_memory();
    println!(
        "Working Memory: {} items, {:.0}% load",
        wm.episodic_buffer.len(),
        wm.load() * 100.0
    );

    // Check emotional state
    let es = agent.emotional_state();
    println!(
        "Emotional State: {} (valence={:+.2}, arousal={:.2})",
        es.label(),
        es.valence,
        es.arousal
    );

    // Check optimal processing state
    let optimal = agent.is_optimal_processing_state();
    println!("Optimal Processing: {}", if optimal { "Yes" } else { "No" });

    // Full introspection
    println!("\n{}", agent.introspect());

    assert!(
        wm.episodic_buffer.len() > 0,
        "working memory should contain items after processing"
    );
    assert!(
        es.arousal >= 0.0 && es.arousal <= 1.0,
        "arousal should be in [0,1], got {}",
        es.arousal
    );
}

#[test]
fn test_memory_source_types() {
    let mut wm = WorkingMemory::new(7);

    // Add items from different sources
    wm.add_to_episodic(
        ContinuousHV::random(1024, 1),
        MemorySource::Perception,
        0.9,
        0,
    );
    wm.add_to_episodic(
        ContinuousHV::random(1024, 2),
        MemorySource::LongTermMemory,
        0.7,
        1,
    );
    wm.add_to_episodic(
        ContinuousHV::random(1024, 3),
        MemorySource::InternalGeneration,
        0.5,
        2,
    );
    wm.add_to_episodic(
        ContinuousHV::random(1024, 4),
        MemorySource::GoalActivation,
        0.8,
        3,
    );

    // Verify sources are tracked
    let sources: Vec<_> = wm.episodic_buffer.iter().map(|item| &item.source).collect();
    assert_eq!(sources.len(), 4);

    println!("Memory source types test passed: {:?}", sources);
}