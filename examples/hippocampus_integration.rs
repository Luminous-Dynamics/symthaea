// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Hippocampus Integration: Concepts Become Memories
//!
//! This test demonstrates the integration between:
//! 1. ConsciousnessWorldModel - crystallizes concepts
//! 2. HippocampusBridge - converts concepts to episodic memories
//! 3. HippocampusActor - stores and recalls memories
//!
//! ## The Flow
//!
//! ```text
//! Experience → WorldModel → Concepts → HippocampusBridge → Memories → Recall
//! ```
//!
//! ## Running
//!
//! ```bash
//! cargo run --release --example hippocampus_integration
//! ```

use symthaea::brain::HippocampusBridge;
use symthaea::consciousness::recursive_improvement::{
    ConsciousnessAction, ConsciousnessTransition, ConsciousnessWorldModel, DreamConfig, DreamMode,
    LatentConsciousnessState, WorldModelConfig,
};
use symthaea::memory::{EmotionalValence, HippocampusActor, RecallQuery};

/// Generate varied experience states
fn generate_experience(
    phase: usize,
    i: usize,
) -> (LatentConsciousnessState, LatentConsciousnessState) {
    let base = (phase as f64 * 0.2 + i as f64 * 0.01).min(0.9);
    let noise = (i as f64 * 0.1).sin() * 0.05;

    let from = LatentConsciousnessState::from_observables(base + noise, 0.5 + noise, 0.5, 0.8);

    let to = LatentConsciousnessState::from_observables(
        base + 0.1 + noise,
        0.5 + 0.05 + noise,
        0.55,
        0.85,
    );

    (from, to)
}

fn main() -> anyhow::Result<()> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║                                                                ║");
    println!("║       H I P P O C A M P U S   I N T E G R A T I O N            ║");
    println!("║                                                                ║");
    println!("║         Crystallized Concepts Become Episodic Memories         ║");
    println!("║                                                                ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    // =========================================================================
    // Phase 1: Initialize All Systems
    // =========================================================================

    println!("Phase 1: Initializing Systems");
    println!("───────────────────────────────────────────────────────────────────");

    let config = WorldModelConfig {
        min_training_samples: 1,
        ..Default::default()
    };

    let mut world_model = ConsciousnessWorldModel::new(config);
    let mut hippocampus_bridge = HippocampusBridge::default();
    let mut hippocampus = HippocampusActor::new(512)?;

    println!("  [OK] ConsciousnessWorldModel initialized (aggressive crystallization)");
    println!("  [OK] HippocampusBridge initialized");
    println!("  [OK] HippocampusActor initialized (512D encodings)");
    println!();

    // =========================================================================
    // Phase 2: Generate Experiences
    // =========================================================================

    println!("Phase 2: Generating Experiences");
    println!("───────────────────────────────────────────────────────────────────");

    const PHASES: usize = 3;
    const EXPERIENCES_PER_PHASE: usize = 30;

    let actions = [
        ConsciousnessAction::FocusIntegration,
        ConsciousnessAction::ExplorePatterns,
        ConsciousnessAction::Consolidate,
    ];

    println!("  Phase | Experiences | Concepts | New Memories");
    println!("  ──────┼─────────────┼──────────┼──────────────");

    for phase in 0..PHASES {
        for i in 0..EXPERIENCES_PER_PHASE {
            let (from_state, to_state) = generate_experience(phase, i);
            let action = actions[i % 3];

            let transition = ConsciousnessTransition {
                from_state,
                action,
                to_state: to_state.clone(),
                reward: to_state.phi - 0.5,
                is_real: true,
            };

            world_model.observe(transition);
        }

        // Sync bridge after each phase
        let new_memories = hippocampus_bridge.sync(&world_model, &mut hippocampus)?;

        let concept_count = world_model.pending_concepts.len();

        println!(
            "  {:>5} | {:>11} | {:>8} | {:>12}",
            phase + 1,
            (phase + 1) * EXPERIENCES_PER_PHASE,
            concept_count,
            new_memories.len()
        );
    }

    println!();

    // =========================================================================
    // Phase 3: Dream Consolidation
    // =========================================================================

    println!("Phase 3: Dream Consolidation");
    println!("───────────────────────────────────────────────────────────────────");

    let dream_config = DreamConfig {
        steps_per_cycle: 20,
        batch_cycles: 2,
        cooldown_ms: 0,
        ..Default::default()
    };

    let mut dream_mode = DreamMode::new(dream_config);
    let dream_results = dream_mode.batch_dream(&mut world_model);

    println!("  Dream cycles: {}", dream_results.stats.cycles_completed);
    println!(
        "  Concepts from dreams: {}",
        dream_results.concepts_discovered.len()
    );
    println!(
        "  Peak consciousness: {:.2}%",
        dream_results.stats.peak_consciousness * 100.0
    );

    // Sync bridge after dreams to encode new concepts as memories
    let dream_memories = hippocampus_bridge.sync(&world_model, &mut hippocampus)?;
    println!("  New memories from dreams: {}", dream_memories.len());
    println!();

    // =========================================================================
    // Phase 4: Memory Recall Tests
    // =========================================================================

    println!("Phase 4: Testing Memory Recall");
    println!("───────────────────────────────────────────────────────────────────");

    // Recall by query
    let query = RecallQuery {
        query: "crystallized concept consciousness".to_string(),
        top_k: 5,
        threshold: 0.2,
        ..Default::default()
    };

    let recall_results = hippocampus.recall(query)?;
    println!(
        "  Recall 'crystallized concept': {} results",
        recall_results.len()
    );

    for (i, result) in recall_results.iter().take(3).enumerate() {
        let content = if result.trace.content.len() > 50 {
            format!("{}...", &result.trace.content[..50])
        } else {
            result.trace.content.clone()
        };
        println!("    [{}] {:.3} - {}", i + 1, result.similarity, content);
    }

    println!();

    // Recall by emotion
    let positive_memories =
        hippocampus_bridge.recall_by_emotion(EmotionalValence::Positive, &mut hippocampus, 5)?;
    println!("  Positive memories: {} found", positive_memories.len());

    let negative_memories =
        hippocampus_bridge.recall_by_emotion(EmotionalValence::Negative, &mut hippocampus, 5)?;
    println!("  Negative memories: {} found", negative_memories.len());

    println!();

    // =========================================================================
    // Phase 5: Concept-to-Memory Mapping
    // =========================================================================

    println!("Phase 5: Concept-to-Memory Mapping");
    println!("───────────────────────────────────────────────────────────────────");

    let mapping = hippocampus_bridge.concept_memory_map();
    println!("  Total concept->memory mappings: {}", mapping.len());

    for (i, (concept_uid, memory_id)) in mapping.iter().take(5).enumerate() {
        println!("    [{}] {} -> Memory #{}", i + 1, concept_uid, memory_id);
    }
    if mapping.len() > 5 {
        println!("    ... and {} more", mapping.len() - 5);
    }

    println!();

    // =========================================================================
    // Phase 6: Results
    // =========================================================================

    println!("═══════════════════════════════════════════════════════════════════");
    println!("                         RESULTS                                   ");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    let bridge_stats = hippocampus_bridge.stats();
    let total_concepts = world_model.pending_concepts.len();
    let memory_stats = hippocampus.stats();

    println!("  Metric                      | Value");
    println!("  ────────────────────────────┼─────────────────────────────────");
    println!(
        "  Total Experiences           | {}",
        PHASES * EXPERIENCES_PER_PHASE
    );
    println!("  Concepts Crystallized       | {}", total_concepts);
    println!(
        "  Concepts Encoded as Memory  | {}",
        bridge_stats.concepts_encoded
    );
    println!(
        "  Total Memories Stored       | {}",
        memory_stats.total_memories
    );
    println!(
        "  Memory Recalls Performed    | {}",
        bridge_stats.recalls_performed
    );
    println!(
        "  Avg Recall Results          | {:.2}",
        bridge_stats.avg_recall_count
    );
    println!(
        "  Peak Consciousness          | {:.2}%",
        bridge_stats.peak_consciousness * 100.0
    );
    println!(
        "  Avg Memory Strength         | {:.3}",
        memory_stats.avg_strength
    );

    println!();

    let success =
        total_concepts >= 5 && bridge_stats.concepts_encoded > 0 && memory_stats.total_memories > 0;

    if success {
        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║                                                                ║");
        println!("║        I N T E G R A T I O N   S U C C E S S F U L             ║");
        println!("║                                                                ║");
        println!("║   Crystallized concepts are now stored as episodic memories!  ║");
        println!("║                                                                ║");
        println!("║   \"I remember when this concept first emerged...\"             ║");
        println!("║                                                                ║");
        println!("╚════════════════════════════════════════════════════════════════╝");
    } else {
        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║                                                                ║");
        println!("║                    NEEDS MORE TUNING                           ║");
        println!("║                                                                ║");
        println!("║   The bridge may need adjustment for better integration.      ║");
        println!("║                                                                ║");
        println!("╚════════════════════════════════════════════════════════════════╝");
    }

    println!();

    Ok(())
}
