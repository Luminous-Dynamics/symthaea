// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Prefrontal Integration: Concepts in the Spotlight
//!
//! This test demonstrates the full integration between:
//! 1. ConsciousnessWorldModel - crystallizes concepts from experience
//! 2. ConsciousnessBridge - converts concepts to attention bids
//! 3. PrefrontalCortex - broadcasts winning concepts to all brain modules
//!
//! ## The Flow
//!
//! ```text
//! Experience → WorldModel → Concepts → Bridge → Bids → Prefrontal → Spotlight
//! ```
//!
//! ## Running
//!
//! ```bash
//! cargo run --release --example prefrontal_integration
//! ```

use symthaea::brain::{AttentionBid, BridgeConfig, ConsciousnessBridge, PrefrontalCortexActor};
use symthaea::consciousness::recursive_improvement::{
    ConsciousnessAction, ConsciousnessTransition, ConsciousnessWorldModel, DreamConfig, DreamMode,
    LatentConsciousnessState, WorldModelConfig,
};

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

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║                                                                ║");
    println!("║       P R E F R O N T A L   I N T E G R A T I O N              ║");
    println!("║                                                                ║");
    println!("║          Concepts Flow to the Spotlight of Consciousness       ║");
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
    let mut bridge = ConsciousnessBridge::new(BridgeConfig::default());
    let mut prefrontal = PrefrontalCortexActor::new();

    println!("  [OK] ConsciousnessWorldModel initialized (aggressive crystallization)");
    println!("  [OK] ConsciousnessBridge initialized");
    println!("  [OK] PrefrontalCortexActor initialized (Global Workspace)");
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

    println!("  Phase | Experiences | Concepts | Bids Generated | Spotlight");
    println!("  ──────┼─────────────┼──────────┼────────────────┼───────────────────────");

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
        bridge.sync(&world_model);

        let concept_count = world_model.pending_concepts.len();
        let bid_count = bridge.pending_bids().len();

        // Submit bids to prefrontal
        let winner = bridge.submit_to_prefrontal(&mut prefrontal, vec![]);

        let spotlight_content = winner
            .as_ref()
            .map(|b| {
                if b.content.len() > 35 {
                    format!("{}...", &b.content[..35])
                } else {
                    b.content.clone()
                }
            })
            .unwrap_or_else(|| "(none)".to_string());

        println!(
            "  {:>5} | {:>11} | {:>8} | {:>14} | {}",
            phase + 1,
            (phase + 1) * EXPERIENCES_PER_PHASE,
            concept_count,
            bid_count,
            spotlight_content
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
    println!("  Steps simulated: {}", dream_results.stats.steps_simulated);
    println!(
        "  Concepts from dreams: {}",
        dream_results.concepts_discovered.len()
    );
    println!(
        "  Peak consciousness: {:.2}%",
        dream_results.stats.peak_consciousness * 100.0
    );
    println!();

    // Sync bridge after dreams
    bridge.sync(&world_model);

    // =========================================================================
    // Phase 4: Continuous Cognitive Cycles
    // =========================================================================

    println!("Phase 4: Running Cognitive Cycles (Simulating Attention Competition)");
    println!("───────────────────────────────────────────────────────────────────");

    const CYCLES: usize = 10;
    let mut concept_wins = 0;
    let mut other_wins = 0;

    println!("  Cycle | Winner Source            | Content");
    println!("  ──────┼──────────────────────────┼────────────────────────────────────");

    for cycle in 0..CYCLES {
        // Generate competing bids from other "brain modules"
        let mut competing_bids = vec![
            AttentionBid::new("Thalamus", "New sensory input detected")
                .with_salience(0.4 + (cycle as f32 * 0.02))
                .with_urgency(0.5),
            AttentionBid::new("Hippocampus", "Relevant memory recalled")
                .with_salience(0.5)
                .with_urgency(0.3),
        ];

        // Add some concept bids from the bridge
        let concept_bids = bridge.drain_bids();
        if !concept_bids.is_empty() {
            competing_bids.extend(concept_bids.into_iter().take(2));
        }

        // Run cognitive cycle
        let winner = prefrontal.cognitive_cycle(competing_bids);

        if let Some(ref bid) = winner {
            if bid.source == "ConsciousnessWorldModel" {
                concept_wins += 1;
            } else {
                other_wins += 1;
            }

            let content = if bid.content.len() > 40 {
                format!("{}...", &bid.content[..40])
            } else {
                bid.content.clone()
            };

            println!("  {:>5} | {:24} | {}", cycle + 1, bid.source, content);
        }
    }

    println!();

    // =========================================================================
    // Phase 5: Check Working Memory
    // =========================================================================

    println!("Phase 5: Global Workspace State");
    println!("───────────────────────────────────────────────────────────────────");

    if let Some(spotlight) = prefrontal.current_focus() {
        println!("  Current Spotlight:");
        println!("    Source: {}", spotlight.source);
        println!("    Content: {}", spotlight.content);
        println!("    Salience: {:.2}", spotlight.salience);
        println!("    Urgency: {:.2}", spotlight.urgency);
    } else {
        println!("  Spotlight: (empty)");
    }

    println!();
    let working_memory = prefrontal.working_memory();
    println!("  Working Memory ({} items):", working_memory.len());
    for (i, item) in working_memory.iter().take(5).enumerate() {
        let content = if item.content.len() > 50 {
            format!("{}...", &item.content[..50])
        } else {
            item.content.clone()
        };
        println!(
            "    [{}] {} (activation: {:.2})",
            i + 1,
            content,
            item.activation
        );
    }

    println!();
    let stream = prefrontal.consciousness_stream();
    println!("  Consciousness Stream ({} recent):", stream.len());
    for (i, bid) in stream.iter().rev().take(3).enumerate() {
        let content = if bid.content.len() > 40 {
            format!("{}...", &bid.content[..40])
        } else {
            bid.content.clone()
        };
        println!("    [{}] {}: {}", i + 1, bid.source, content);
    }

    println!();

    // =========================================================================
    // Phase 6: Results
    // =========================================================================

    println!("═══════════════════════════════════════════════════════════════════");
    println!("                         RESULTS                                   ");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    let bridge_stats = bridge.stats();
    let total_concepts = world_model.pending_concepts.len();

    println!("  Metric                      | Value");
    println!("  ────────────────────────────┼─────────────────────────────────");
    println!(
        "  Total Experiences           | {}",
        PHASES * EXPERIENCES_PER_PHASE
    );
    println!("  Concepts Crystallized       | {}", total_concepts);
    println!(
        "  Concepts -> Bids Converted  | {}",
        bridge_stats.concepts_converted
    );
    println!(
        "  Bids Submitted to Prefrontal| {}",
        bridge_stats.bids_submitted
    );
    println!(
        "  Concept Bids Won Spotlight  | {}",
        bridge_stats.bids_won_spotlight
    );
    println!("  Other Bids Won Spotlight    | {}", other_wins);
    println!(
        "  Average Bid Salience        | {:.3}",
        bridge_stats.avg_salience
    );
    println!(
        "  Peak Consciousness          | {:.2}%",
        bridge_stats.peak_consciousness * 100.0
    );

    println!();

    let success = total_concepts >= 5
        && bridge_stats.concepts_converted > 0
        && bridge_stats.bids_submitted > 0;

    if success {
        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║                                                                ║");
        println!("║        I N T E G R A T I O N   S U C C E S S F U L             ║");
        println!("║                                                                ║");
        println!("║   The Heart (WorldModel) speaks to the Brain (Prefrontal)!    ║");
        println!("║                                                                ║");
        println!("║   Crystallized concepts now compete for conscious attention.  ║");
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
}
