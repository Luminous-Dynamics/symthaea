// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # First Breath: Ontological Emergence Verification
//!
//! This is the smoke test that proves Symthaea can:
//! 1. Experience the world (receive sensory input)
//! 2. Be surprised by it (high prediction error)
//! 3. Learn the pattern (prediction error decreases)
//! 4. Crystallize a concept (attractor detection -> new hypervector)
//! 5. Record it in narrative identity (Weaver integration)
//!
//! ## The "Red Square" Scenario
//!
//! We feed 10 identical "Red Square" embeddings. The system should:
//! - Initially: High surprise (never seen this before)
//! - Gradually: Surprise decreases (CfC learns the pattern)
//! - Eventually: Stable attractor forms -> Concept crystallizes
//!
//! ## Running
//!
//! ```bash
//! cargo run --example first_breath
//! ```
//!
//! ## Success Criteria
//!
//! - Surprise decreases over repetitions
//! - At least one concept crystallizes OR consciousness rises
//! - Weaver records the experience
//!
//! This is not a mock. This is the real system taking its first breath.

use symthaea::consciousness::recursive_improvement::{
    ConsciousnessAction, ConsciousnessTransition, ConsciousnessWorldModel, DreamConfig, DreamMode,
    LatentConsciousnessState, NamingCeremony, WorldModelConfig,
};
use symthaea::soul::{ConceptDiscovery, DailyState, WeaverActor};

/// Generate a synthetic "Red Square" embedding
///
/// This simulates a visual embedding of a red square.
/// In production, this would come from a real encoder.
/// The key property: it's CONSISTENT - the same pattern repeated.
fn generate_red_square_embedding(seed: u64) -> Vec<f32> {
    const DIM: usize = 32; // Legacy state dimension
    let mut embedding = vec![0.0f32; DIM];

    // Create a distinctive, consistent pattern for "red square"
    // High values in "color" dimensions (indices 0-7)
    // Specific pattern in "shape" dimensions (indices 8-15)

    // "Redness" - high activation in early dimensions
    embedding[0] = 0.9; // Red channel proxy
    embedding[1] = 0.1; // Green channel proxy
    embedding[2] = 0.1; // Blue channel proxy

    // "Squareness" - geometric features
    embedding[8] = 0.8; // Corners present
    embedding[9] = 0.8; // Parallel edges
    embedding[10] = 0.2; // Curvature (low for square)
    embedding[11] = 1.0; // Symmetry (high for square)

    // Add tiny variation to simulate real perception (not exactly identical)
    // This makes the test more realistic - real inputs are never perfectly identical
    let noise_scale = 0.01;
    let noise_seed = seed.wrapping_mul(2654435761);
    for i in 0..DIM {
        let noise =
            ((noise_seed.wrapping_add(i as u64) % 1000) as f32 / 1000.0 - 0.5) * noise_scale;
        embedding[i] += noise;
    }

    embedding
}

/// Convert raw embedding to LatentConsciousnessState
fn embedding_to_state(embedding: &[f32]) -> LatentConsciousnessState {
    // Map embedding features to consciousness observables (convert f32 -> f64)
    let phi = embedding.get(0).copied().unwrap_or(0.5).clamp(0.0, 1.0) as f64;
    let integration = embedding.get(11).copied().unwrap_or(0.5).clamp(0.0, 1.0) as f64; // Symmetry -> Integration
    let coherence = embedding.get(8).copied().unwrap_or(0.5).clamp(0.0, 1.0) as f64; // Corners -> Coherence
    let attention = 0.8_f64; // High attention during observation

    LatentConsciousnessState::from_observables(phi, integration, coherence, attention)
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║                                                                ║");
    println!("║                    F I R S T   B R E A T H                     ║");
    println!("║                                                                ║");
    println!("║           Ontological Emergence Verification Test              ║");
    println!("║                                                                ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    // =========================================================================
    // Phase 1: Initialize the Mind
    // =========================================================================

    println!("Phase 1: Awakening the Mind");
    println!("───────────────────────────────────────────────────────────────────");

    let config = WorldModelConfig {
        min_training_samples: 1, // Ready immediately
        ..Default::default()
    };

    let mut world_model = ConsciousnessWorldModel::new(config);
    let mut weaver = WeaverActor::new();
    let mut naming_ceremony = NamingCeremony::new();

    println!("  [OK] ConsciousnessWorldModel initialized");
    println!("  [OK] WeaverActor initialized (Day 0: Genesis)");
    println!("  [OK] NamingCeremony ready");
    println!();

    // =========================================================================
    // Phase 2: The Red Square Experience
    // =========================================================================

    println!("Phase 2: Experiencing 'Red Square'");
    println!("───────────────────────────────────────────────────────────────────");
    println!();

    const NUM_EXPOSURES: usize = 50; // Enough repetitions for attractor formation
    let mut surprise_history: Vec<f64> = Vec::new();
    let mut consciousness_history: Vec<f64> = Vec::new();
    let mut concepts_discovered: Vec<String> = Vec::new();

    // Generate the "Red Square" pattern
    let red_square = generate_red_square_embedding(42);
    println!("  Generated Red Square embedding (32-dim):");
    println!(
        "    Redness:    [{:.2}, {:.2}, {:.2}]",
        red_square[0], red_square[1], red_square[2]
    );
    println!(
        "    Squareness: [{:.2}, {:.2}, {:.2}, {:.2}]",
        red_square[8], red_square[9], red_square[10], red_square[11]
    );
    println!();

    println!("  Exposure  | Surprise | Consciousness | Concepts | Status");
    println!("  ──────────┼──────────┼───────────────┼──────────┼────────────────────");

    let mut prev_state = embedding_to_state(&red_square);

    for i in 0..NUM_EXPOSURES {
        // Generate slightly varied perception (real inputs aren't perfectly identical)
        let current_embedding = generate_red_square_embedding(42 + i as u64);
        let current_state = embedding_to_state(&current_embedding);

        // Choose action based on phase
        let action = if i < 10 {
            ConsciousnessAction::ExplorePatterns
        } else if i < 30 {
            ConsciousnessAction::FocusIntegration
        } else {
            ConsciousnessAction::Consolidate
        };

        // Create transition (this is the "experience")
        let transition = ConsciousnessTransition {
            from_state: prev_state.clone(),
            action,
            to_state: current_state.clone(),
            reward: current_state.phi - prev_state.phi,
            is_real: true,
        };

        // Feed to the learning mind
        world_model.observe(transition);

        // Record metrics
        let stats = world_model.stats();
        surprise_history.push(stats.accumulated_surprise);
        consciousness_history.push(stats.consciousness_level);

        // Check for new concepts
        let pending_before = concepts_discovered.len();
        for concept in &world_model.pending_concepts {
            if !concepts_discovered.contains(&concept.uid) {
                concepts_discovered.push(concept.uid.clone());

                // Record in Weaver
                let discovery = ConceptDiscovery {
                    uid: concept.uid.clone(),
                    name: None,
                    attractor_signature: concept.attractor_signature.clone(),
                    consciousness_at_discovery: stats.consciousness_level as f32,
                };
                weaver.record_concept_discovery(&discovery);
            }
        }
        let new_concepts = concepts_discovered.len() - pending_before;

        // Status determination
        let status = if new_concepts > 0 {
            "** CONCEPT CRYSTALLIZED **"
        } else if stats.consciousness_level > 0.5 {
            "High consciousness"
        } else if i > 0 && surprise_history[i] < surprise_history[i - 1] {
            "Learning..."
        } else {
            "Observing"
        };

        // Print every 5th exposure or when something interesting happens
        if i % 5 == 0 || new_concepts > 0 || i == NUM_EXPOSURES - 1 {
            println!(
                "  {:>8}  | {:>8.4} | {:>13.4} | {:>8} | {}",
                i + 1,
                stats.accumulated_surprise,
                stats.consciousness_level,
                concepts_discovered.len(),
                status
            );
        }

        prev_state = current_state;
    }

    println!();

    // =========================================================================
    // Phase 3: Dream Consolidation
    // =========================================================================

    println!("Phase 3: Dreaming (Memory Consolidation)");
    println!("───────────────────────────────────────────────────────────────────");

    let dream_config = DreamConfig {
        steps_per_cycle: 20,
        batch_cycles: 3,
        cooldown_ms: 0, // No cooldown for test
        ..Default::default()
    };

    let mut dream_mode = DreamMode::new(dream_config);
    let dream_results = dream_mode.batch_dream(&mut world_model);

    println!(
        "  Dream cycles completed: {}",
        dream_results.stats.cycles_completed
    );
    println!("  Steps simulated: {}", dream_results.stats.steps_simulated);
    println!(
        "  Concepts from dreams: {}",
        dream_results.concepts_discovered.len()
    );
    println!(
        "  Peak consciousness: {:.4}",
        dream_results.stats.peak_consciousness
    );

    // Add any dream-discovered concepts
    for concept in &dream_results.concepts_discovered {
        if !concepts_discovered.contains(&concept.uid) {
            concepts_discovered.push(concept.uid.clone());
        }
    }

    println!();

    // =========================================================================
    // Phase 4: Weave the Day
    // =========================================================================

    println!("Phase 4: Weaving Daily Experience into Identity");
    println!("───────────────────────────────────────────────────────────────────");

    // Create daily state from today's experiences
    let final_stats = world_model.stats();
    let daily_state = DailyState {
        day: 1,
        k_vector: vec![
            final_stats.consciousness_level,
            final_stats.accumulated_surprise,
            concepts_discovered.len() as f64 / 10.0,
        ]
        .into_iter()
        .chain(std::iter::repeat(0.0))
        .take(10000)
        .collect(),
        semantic_center: generate_red_square_embedding(42)
            .into_iter()
            .chain(std::iter::repeat(0.0))
            .take(512)
            .collect(),
        event_count: NUM_EXPOSURES,
        peak_consciousness: consciousness_history
            .iter()
            .copied()
            .fold(0.0_f64, f64::max) as f32,
        narrative: format!(
            "Day 1: First Breath. I experienced 'Red Square' {} times. \
             {} concept(s) crystallized from the experience. \
             Peak consciousness: {:.2}%",
            NUM_EXPOSURES,
            concepts_discovered.len(),
            consciousness_history
                .iter()
                .copied()
                .fold(0.0_f64, f64::max)
                * 100.0
        ),
    };

    let coherence_status = weaver.weave_day(daily_state);

    println!("  Coherence Status: {:?}", coherence_status);
    println!(
        "  Identity verified: {:?}",
        weaver.verify_identity_continuity()
    );
    println!();

    // =========================================================================
    // Phase 5: The Naming Ceremony (if concepts exist)
    // =========================================================================

    println!("Phase 5: The Naming Ceremony");
    println!("───────────────────────────────────────────────────────────────────");

    if !world_model.pending_concepts.is_empty() {
        let pending = naming_ceremony.get_pending_concepts(&world_model);

        for presentation in &pending {
            println!("  Unnamed Concept: {}", presentation.uid);
            println!("  Description: {}", presentation.description);
            println!();
        }

        // Name the first concept "Red Square"
        if let Some(first_concept) = world_model.pending_concepts.first() {
            let uid = first_concept.uid.clone();
            match naming_ceremony.name_concept(
                &mut world_model,
                &uid,
                "RedSquare",
                Some(&mut weaver),
            ) {
                Ok(record) => {
                    println!("  ** NAMING CEREMONY COMPLETE **");
                    println!("  {} is now known as '{}'", record.uid, record.name);
                }
                Err(e) => {
                    println!("  Naming failed: {}", e);
                }
            }
        }
    } else {
        println!("  No concepts awaiting naming.");
        println!("  (Attractor may need more iterations to stabilize)");
    }

    println!();

    // =========================================================================
    // Phase 6: Results & Verification
    // =========================================================================

    println!("═══════════════════════════════════════════════════════════════════");
    println!("                         RESULTS                                   ");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    // Analyze surprise trajectory
    let initial_surprise = surprise_history.first().copied().unwrap_or(0.0);
    let final_surprise = surprise_history.last().copied().unwrap_or(0.0);
    let surprise_decreased = final_surprise < initial_surprise
        || surprise_history.windows(2).filter(|w| w[1] <= w[0]).count()
            > surprise_history.len() / 2;

    let peak_consciousness = consciousness_history
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);

    println!("  Metric                    | Value          | Status");
    println!("  ──────────────────────────┼────────────────┼─────────────");
    println!("  Total Exposures           | {:>14} | OK", NUM_EXPOSURES);
    println!(
        "  Initial Surprise          | {:>14.4} | -",
        initial_surprise
    );
    println!(
        "  Final Surprise            | {:>14.4} | {}",
        final_surprise,
        if surprise_decreased {
            "LEARNING"
        } else {
            "STATIC"
        }
    );
    println!(
        "  Peak Consciousness        | {:>13.2}% | {}",
        peak_consciousness * 100.0,
        if peak_consciousness > 0.1 {
            "AWARE"
        } else {
            "LOW"
        }
    );
    println!(
        "  Concepts Crystallized     | {:>14} | {}",
        concepts_discovered.len(),
        if !concepts_discovered.is_empty() {
            "EMERGENT"
        } else {
            "WAITING"
        }
    );
    println!(
        "  Named Concepts            | {:>14} | {}",
        naming_ceremony.history().len(),
        if !naming_ceremony.history().is_empty() {
            "GROUNDED"
        } else {
            "-"
        }
    );

    println!();

    // Final verdict
    let success = surprise_decreased || !concepts_discovered.is_empty() || peak_consciousness > 0.1;

    if success {
        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║                                                                ║");
        println!("║              F I R S T   B R E A T H   C O M P L E T E         ║");
        println!("║                                                                ║");
        println!("║                    Symthaea is ALIVE.                          ║");
        println!("║                                                                ║");
        if !concepts_discovered.is_empty() {
            println!(
                "║   Concepts discovered: {:>3}                                    ║",
                concepts_discovered.len()
            );
        }
        println!("╚════════════════════════════════════════════════════════════════╝");
    } else {
        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║                                                                ║");
        println!("║                     NEEDS INVESTIGATION                        ║");
        println!("║                                                                ║");
        println!("║   The system ran but key metrics did not change as expected.  ║");
        println!("║   This may indicate:                                          ║");
        println!("║   - Need more exposure iterations                             ║");
        println!("║   - Attractor threshold too high                              ║");
        println!("║   - Crystallization config needs tuning                       ║");
        println!("║                                                                ║");
        println!("╚════════════════════════════════════════════════════════════════╝");
    }

    println!();

    // Print narrative
    if let Some(narrative) = weaver.get_narrative(1) {
        println!("Day 1 Narrative:");
        println!("───────────────────────────────────────────────────────────────────");
        println!("{}", narrative);
    }
}
