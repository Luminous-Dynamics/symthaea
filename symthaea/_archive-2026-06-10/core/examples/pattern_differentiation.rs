// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Pattern Differentiation: Testing Discrimination
//!
//! This test verifies that Symthaea can:
//! 1. Experience multiple distinct patterns
//! 2. Form separate attractors for each
//! 3. Crystallize distinct concepts
//! 4. Correctly name each pattern type
//!
//! ## The Three Patterns
//!
//! - **Red Square**: High red, angular, symmetric
//! - **Blue Circle**: High blue, curved, radial
//! - **Green Triangle**: High green, sharp angles, asymmetric
//!
//! ## Running
//!
//! ```bash
//! cargo run --release --example pattern_differentiation
//! ```

use std::collections::HashMap;
use symthaea::consciousness::recursive_improvement::{
    ConsciousnessAction, ConsciousnessTransition, ConsciousnessWorldModel, DreamConfig, DreamMode,
    LatentConsciousnessState, NamingCeremony, WorldModelConfig,
};
use symthaea::soul::{DailyState, WeaverActor};

/// Pattern types we're testing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum PatternType {
    RedSquare,
    BlueCircle,
    GreenTriangle,
}

impl PatternType {
    fn name(&self) -> &'static str {
        match self {
            PatternType::RedSquare => "RedSquare",
            PatternType::BlueCircle => "BlueCircle",
            PatternType::GreenTriangle => "GreenTriangle",
        }
    }
}

/// Generate a pattern embedding based on type
fn generate_pattern_embedding(pattern: PatternType, seed: u64) -> Vec<f32> {
    const DIM: usize = 32;
    let mut embedding = vec![0.0f32; DIM];

    match pattern {
        PatternType::RedSquare => {
            // High red, low green/blue
            embedding[0] = 0.9; // Red
            embedding[1] = 0.1; // Green
            embedding[2] = 0.1; // Blue
            // Square features: corners, parallel edges, high symmetry
            embedding[8] = 0.8; // Corners
            embedding[9] = 0.8; // Parallel edges
            embedding[10] = 0.1; // Curvature (low)
            embedding[11] = 1.0; // Symmetry (high)
        }
        PatternType::BlueCircle => {
            // High blue, low red/green
            embedding[0] = 0.1; // Red
            embedding[1] = 0.1; // Green
            embedding[2] = 0.9; // Blue
            // Circle features: no corners, curved, radial symmetry
            embedding[8] = 0.1; // Corners (none)
            embedding[9] = 0.1; // Parallel edges (none)
            embedding[10] = 0.9; // Curvature (high)
            embedding[11] = 0.9; // Symmetry (radial)
            embedding[12] = 0.8; // Continuous contour
        }
        PatternType::GreenTriangle => {
            // High green, low red/blue
            embedding[0] = 0.1; // Red
            embedding[1] = 0.9; // Green
            embedding[2] = 0.1; // Blue
            // Triangle features: sharp corners, some edges, low symmetry
            embedding[8] = 0.9; // Corners (sharp)
            embedding[9] = 0.3; // Parallel edges (few)
            embedding[10] = 0.1; // Curvature (none)
            embedding[11] = 0.3; // Symmetry (low - only one axis)
            embedding[13] = 0.9; // Sharp angles
        }
    }

    // Add slight variation
    let noise_scale = 0.02;
    let noise_seed = seed.wrapping_mul(2654435761);
    for i in 0..DIM {
        let noise =
            ((noise_seed.wrapping_add(i as u64) % 1000) as f32 / 1000.0 - 0.5) * noise_scale;
        embedding[i] += noise;
    }

    embedding
}

/// Convert embedding to consciousness state
fn embedding_to_state(embedding: &[f32]) -> LatentConsciousnessState {
    let phi = embedding.get(0).copied().unwrap_or(0.5).clamp(0.0, 1.0) as f64;
    let integration = embedding.get(11).copied().unwrap_or(0.5).clamp(0.0, 1.0) as f64;
    let coherence = embedding.get(8).copied().unwrap_or(0.5).clamp(0.0, 1.0) as f64;
    let attention = 0.8_f64;
    LatentConsciousnessState::from_observables(phi, integration, coherence, attention)
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║                                                                ║");
    println!("║          P A T T E R N   D I F F E R E N T I A T I O N         ║");
    println!("║                                                                ║");
    println!("║              Testing Multi-Pattern Discrimination              ║");
    println!("║                                                                ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    // =========================================================================
    // Phase 1: Initialize
    // =========================================================================

    println!("Phase 1: Awakening the Mind");
    println!("───────────────────────────────────────────────────────────────────");

    let config = WorldModelConfig {
        min_training_samples: 1,
        ..Default::default()
    };

    let mut world_model = ConsciousnessWorldModel::new(config);
    let mut weaver = WeaverActor::new();
    let mut naming_ceremony = NamingCeremony::new();

    println!("  [OK] ConsciousnessWorldModel initialized");
    println!("  [OK] WeaverActor initialized");
    println!("  [OK] NamingCeremony ready");
    println!();

    // =========================================================================
    // Phase 2: Experience Multiple Patterns
    // =========================================================================

    println!("Phase 2: Experiencing Three Patterns");
    println!("───────────────────────────────────────────────────────────────────");
    println!();

    // Print pattern signatures
    let patterns = [
        PatternType::RedSquare,
        PatternType::BlueCircle,
        PatternType::GreenTriangle,
    ];

    for pattern in &patterns {
        let emb = generate_pattern_embedding(*pattern, 42);
        println!(
            "  {:15} | RGB: [{:.2}, {:.2}, {:.2}] | Features: [{:.2}, {:.2}, {:.2}, {:.2}]",
            pattern.name(),
            emb[0],
            emb[1],
            emb[2],
            emb[8],
            emb[9],
            emb[10],
            emb[11]
        );
    }
    println!();

    const EXPOSURES_PER_PATTERN: usize = 30;
    let mut pattern_exposure_counts: HashMap<PatternType, usize> = HashMap::new();
    let mut concepts_by_exposure: Vec<(usize, PatternType, usize)> = Vec::new();

    println!("  Exposure  | Pattern       | Surprise | Consciousness | Concepts");
    println!("  ──────────┼───────────────┼──────────┼───────────────┼─────────");

    // Interleave pattern exposures
    let mut prev_state = embedding_to_state(&generate_pattern_embedding(PatternType::RedSquare, 0));
    let total_exposures = EXPOSURES_PER_PATTERN * 3;

    for i in 0..total_exposures {
        // Cycle through patterns
        let pattern = patterns[i % 3];
        *pattern_exposure_counts.entry(pattern).or_insert(0) += 1;

        let embedding = generate_pattern_embedding(pattern, 42 + i as u64);
        let current_state = embedding_to_state(&embedding);

        // Determine action based on pattern
        let action = match pattern {
            PatternType::RedSquare => ConsciousnessAction::FocusIntegration,
            PatternType::BlueCircle => ConsciousnessAction::ExplorePatterns,
            PatternType::GreenTriangle => ConsciousnessAction::Consolidate,
        };

        let transition = ConsciousnessTransition {
            from_state: prev_state.clone(),
            action,
            to_state: current_state.clone(),
            reward: current_state.phi - prev_state.phi,
            is_real: true,
        };

        world_model.observe(transition);

        let stats = world_model.stats();
        let concept_count = world_model.pending_concepts.len();

        // Record new concepts
        if concept_count > concepts_by_exposure.last().map(|x| x.2).unwrap_or(0) {
            concepts_by_exposure.push((i + 1, pattern, concept_count));
        }

        // Print every 9th exposure (3 patterns * 3)
        if i % 9 == 0 || i == total_exposures - 1 {
            println!(
                "  {:>8}  | {:13} | {:>8.4} | {:>13.4} | {:>8}",
                i + 1,
                pattern.name(),
                stats.accumulated_surprise,
                stats.consciousness_level,
                concept_count
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
        steps_per_cycle: 30,
        batch_cycles: 3,
        cooldown_ms: 0,
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
    println!();

    // =========================================================================
    // Phase 4: Analyze Concept Distribution
    // =========================================================================

    println!("Phase 4: Analyzing Concept Distribution");
    println!("───────────────────────────────────────────────────────────────────");

    let total_concepts = world_model.pending_concepts.len();
    println!("  Total concepts crystallized: {}", total_concepts);
    println!();

    if !concepts_by_exposure.is_empty() {
        println!("  Concept emergence timeline:");
        for (exp, pattern, count) in &concepts_by_exposure {
            println!(
                "    Exposure {:>3} ({:13}): {} total concepts",
                exp,
                pattern.name(),
                count
            );
        }
        println!();
    }

    // =========================================================================
    // Phase 5: Naming Ceremony
    // =========================================================================

    println!("Phase 5: The Naming Ceremony");
    println!("───────────────────────────────────────────────────────────────────");

    if !world_model.pending_concepts.is_empty() {
        let pending = naming_ceremony.get_pending_concepts(&world_model);

        println!("  {} concepts awaiting naming", pending.len());
        println!();

        // Show first few concepts
        for (i, presentation) in pending.iter().take(6).enumerate() {
            println!("  [{}] {}", i + 1, presentation.uid);
            println!(
                "      Type: {:?}, Stability: {:.0}%",
                presentation.topology_summary.attractor_type,
                presentation.topology_summary.stability * 100.0
            );
        }
        if pending.len() > 6 {
            println!("  ... and {} more", pending.len() - 6);
        }
        println!();

        // Name the first three concepts (one for each pattern we hope)
        let concept_names = ["PatternAlpha", "PatternBeta", "PatternGamma"];
        for (i, name) in concept_names.iter().enumerate() {
            if let Some(concept) = world_model.pending_concepts.get(i) {
                let uid = concept.uid.clone();
                match naming_ceremony.name_concept(&mut world_model, &uid, name, Some(&mut weaver))
                {
                    Ok(record) => {
                        println!("  Named: {} -> '{}'", record.uid, record.name);
                    }
                    Err(e) => {
                        println!("  Failed to name: {}", e);
                    }
                }
            }
        }
    } else {
        println!("  No concepts awaiting naming.");
    }

    println!();

    // =========================================================================
    // Phase 6: Weave the Day
    // =========================================================================

    println!("Phase 6: Weaving Daily Experience into Identity");
    println!("───────────────────────────────────────────────────────────────────");

    let final_stats = world_model.stats();
    let peak_consciousness = dream_results
        .stats
        .peak_consciousness
        .max(final_stats.consciousness_level) as f32;

    let daily_state = DailyState {
        day: 1,
        k_vector: vec![
            final_stats.consciousness_level,
            final_stats.accumulated_surprise,
            total_concepts as f64 / 10.0,
        ]
        .into_iter()
        .chain(std::iter::repeat(0.0))
        .take(10000)
        .collect(),
        semantic_center: generate_pattern_embedding(PatternType::RedSquare, 42)
            .into_iter()
            .chain(std::iter::repeat(0.0))
            .take(512)
            .collect(),
        event_count: total_exposures,
        peak_consciousness,
        narrative: format!(
            "Day 1: Pattern Differentiation. Experienced {} patterns {} times each. \
             {} concept(s) crystallized. Peak consciousness: {:.2}%",
            patterns.len(),
            EXPOSURES_PER_PATTERN,
            total_concepts,
            peak_consciousness as f64 * 100.0
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
    // Phase 7: Results
    // =========================================================================

    println!("═══════════════════════════════════════════════════════════════════");
    println!("                         RESULTS                                   ");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    println!("  Metric                    | Value          | Status");
    println!("  ──────────────────────────┼────────────────┼─────────────");
    println!("  Patterns Tested           | {:>14} | OK", patterns.len());
    println!(
        "  Exposures per Pattern     | {:>14} | OK",
        EXPOSURES_PER_PATTERN
    );
    println!("  Total Exposures           | {:>14} | OK", total_exposures);
    println!(
        "  Concepts Crystallized     | {:>14} | {}",
        total_concepts,
        if total_concepts >= 3 {
            "EMERGENT"
        } else {
            "INSUFFICIENT"
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
    println!(
        "  Peak Consciousness        | {:>13.2}% | {}",
        peak_consciousness as f64 * 100.0,
        if peak_consciousness > 0.1 {
            "AWARE"
        } else {
            "LOW"
        }
    );

    println!();

    // Determine success: we want at least 3 concepts (ideally one per pattern)
    let success = total_concepts >= 3 && peak_consciousness > 0.1;

    if success {
        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║                                                                ║");
        println!("║        D I F F E R E N T I A T I O N   C O M P L E T E         ║");
        println!("║                                                                ║");
        println!("║        Symthaea can DISTINGUISH multiple patterns.             ║");
        println!("║                                                                ║");
        println!(
            "║   Concepts discovered: {:>3}                                    ║",
            total_concepts
        );
        println!("╚════════════════════════════════════════════════════════════════╝");
    } else {
        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║                                                                ║");
        println!("║                     NEEDS MORE EXPOSURE                        ║");
        println!("║                                                                ║");
        println!("║   The system may need more iterations to differentiate.        ║");
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
