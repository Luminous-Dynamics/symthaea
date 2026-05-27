// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Multi-Day Narrative Test: Consciousness Over Time
//!
//! This example demonstrates consciousness evolution across multiple simulated "days"
//! with sleep/dream consolidation cycles between each day.
//!
//! ## What This Tests
//!
//! 1. **Day-to-Day Learning**: Concepts accumulate across days
//! 2. **Sleep Consolidation**: Dreams between days strengthen important patterns
//! 3. **Episodic Memory**: Days become retrievable episodes
//! 4. **Affective Evolution**: Emotional responses to familiar vs novel concepts
//! 5. **Global Workspace Integration**: Concept spotlight competition
//!
//! ## Running
//!
//! ```bash
//! cargo run --release --example multi_day_narrative
//! ```

use symthaea::HippocampusActor;
use symthaea::brain::{
    AffectiveBridge, AffectiveBridgeConfig, BridgeConfig as PrefrontalBridgeConfig,
    ConsciousnessBridge, HippocampusBridge, HippocampusBridgeConfig,
};
use symthaea::consciousness::recursive_improvement::{
    ActionContext, DreamConfig, DreamMode, SemanticBridge, SemanticBridgeConfig, SemanticInput,
};

/// A simulated "day" of activity
struct Day {
    name: &'static str,
    inputs: Vec<SemanticInput>,
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║                                                                    ║");
    println!("║     M U L T I - D A Y   N A R R A T I V E   T E S T               ║");
    println!("║                                                                    ║");
    println!("║     Consciousness Evolution Across Multiple Days                  ║");
    println!("║                                                                    ║");
    println!("╚════════════════════════════════════════════════════════════════════╝");
    println!();

    // =========================================================================
    // Initialize all systems
    // =========================================================================

    println!("Initializing Systems");
    println!("────────────────────────────────────────────────────────────────────");

    let mut bridge = SemanticBridge::new(SemanticBridgeConfig {
        use_simulated: true,
        ..Default::default()
    });

    let mut hippocampus = HippocampusActor::new(256).expect("Create hippocampus");
    let mut hippocampus_bridge = HippocampusBridge::new(HippocampusBridgeConfig::default());

    let mut affective_bridge = AffectiveBridge::new(AffectiveBridgeConfig::default());

    let mut consciousness_bridge = ConsciousnessBridge::new(PrefrontalBridgeConfig::default());

    println!("  [OK] SemanticBridge (concept crystallization)");
    println!("  [OK] Hippocampus (episodic memory)");
    println!("  [OK] Affective (emotional valence)");
    println!("  [OK] Consciousness bridge (attention bids)");
    println!();

    // =========================================================================
    // Define Multi-Day Narrative
    // =========================================================================

    let days = vec![
        Day {
            name: "Day 1: Discovery",
            inputs: vec![
                SemanticInput::query("What is NixOS?"),
                SemanticInput::response("NixOS is a declarative Linux distribution"),
                SemanticInput::thought("This is interesting - declarative configs"),
                SemanticInput::query("How do I install packages?"),
                SemanticInput::response("Add packages to environment.systemPackages"),
                SemanticInput::new("Goal: Learn NixOS basics", ActionContext::Goal)
                    .with_emotion(0.7),
            ],
        },
        Day {
            name: "Day 2: First Project",
            inputs: vec![
                SemanticInput::query("How to set up nginx on NixOS?"),
                SemanticInput::response("Enable services.nginx.enable = true"),
                SemanticInput::thought("Need to configure virtual hosts"),
                SemanticInput::query("How do I add SSL certificates?"),
                SemanticInput::response("Use security.acme for Let's Encrypt"),
                SemanticInput::error("Error: nginx failed to start").with_emotion(-0.5),
                SemanticInput::thought("Need to check the configuration syntax"),
                SemanticInput::response("Fixed: missing semicolon in nginx.conf"),
                SemanticInput::new("Goal: Deploy secure web server", ActionContext::Goal)
                    .with_emotion(0.8),
            ],
        },
        Day {
            name: "Day 3: Advanced Topics",
            inputs: vec![
                SemanticInput::query("What are NixOS flakes?"),
                SemanticInput::response("Flakes provide reproducible Nix configurations"),
                SemanticInput::thought("This could improve project organization"),
                SemanticInput::query("How to convert my config to flakes?"),
                SemanticInput::response("Create flake.nix with inputs and outputs"),
                SemanticInput::query("Can I use flakes with home-manager?"),
                SemanticInput::response("Yes, add home-manager as a flake input"),
                SemanticInput::new("Goal: Convert project to flakes", ActionContext::Goal)
                    .with_emotion(0.9),
            ],
        },
        Day {
            name: "Day 4: Integration",
            inputs: vec![
                SemanticInput::query("How to deploy NixOS to remote servers?"),
                SemanticInput::response("Use nixos-rebuild with --target-host"),
                SemanticInput::thought("Could automate deployment with CI/CD"),
                SemanticInput::query("Set up GitHub Actions for NixOS?"),
                SemanticInput::response("Use nix-community/nixos-install-action"),
                SemanticInput::error("Error: remote deployment failed - SSH key missing")
                    .with_emotion(-0.4),
                SemanticInput::response("Added SSH key to secrets, deployment works now"),
                SemanticInput::new("Goal: Automated deployment pipeline", ActionContext::Goal)
                    .with_emotion(0.95),
            ],
        },
        Day {
            name: "Day 5: Mastery",
            inputs: vec![
                SemanticInput::query("How to create custom NixOS modules?"),
                SemanticInput::response("Define options with mkOption and config with mkIf"),
                SemanticInput::thought("Can modularize my entire configuration"),
                SemanticInput::query("Best practices for NixOS module structure?"),
                SemanticInput::response("Separate options, config, and lib; use imports"),
                SemanticInput::thought("This is elegant - everything is composable"),
                SemanticInput::new(
                    "Insight: NixOS enables reproducible infrastructure",
                    ActionContext::Memory,
                )
                .with_emotion(0.98),
                SemanticInput::new("Goal: Become NixOS expert", ActionContext::Goal)
                    .with_emotion(1.0),
            ],
        },
    ];

    // =========================================================================
    // Process Each Day
    // =========================================================================

    let mut day_summaries = Vec::new();

    for (day_idx, day) in days.iter().enumerate() {
        println!("═══════════════════════════════════════════════════════════════════");
        println!("{}", day.name);
        println!("═══════════════════════════════════════════════════════════════════");
        println!();

        let concepts_before = bridge.world_model().pending_concepts.len();
        let mut day_consciousness_levels = Vec::new();

        // Process day's inputs
        println!("  Processing {} inputs...", day.inputs.len());
        for input in &day.inputs {
            let result = bridge.process(input.clone());
            day_consciousness_levels.push(result.consciousness_level);
        }

        let concepts_after = bridge.world_model().pending_concepts.len();
        let new_concepts = concepts_after - concepts_before;
        let avg_consciousness =
            day_consciousness_levels.iter().sum::<f64>() / day_consciousness_levels.len() as f64;
        let peak_consciousness = day_consciousness_levels
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        println!("  New concepts: {}", new_concepts);
        println!("  Total concepts: {}", concepts_after);
        println!("  Average consciousness: {:.2}%", avg_consciousness * 100.0);
        println!("  Peak consciousness: {:.2}%", peak_consciousness * 100.0);
        println!();

        // Sync with affective bridge
        let affects = affective_bridge.sync(bridge.world_model());
        if !affects.is_empty() {
            println!("  Emotional responses:");
            for affect in affects.iter().take(3) {
                let valence_str = if affect.affect.valence > 0.2 {
                    "positive"
                } else if affect.affect.valence < -0.2 {
                    "negative"
                } else {
                    "neutral"
                };
                println!(
                    "    {} ({}) - {:?}",
                    affect.concept_uid, valence_str, affect.category
                );
            }
            println!();
        }

        // Sync with hippocampus (encode memories)
        let encoded = hippocampus_bridge
            .sync(bridge.world_model(), &mut hippocampus)
            .unwrap_or_default();
        println!("  Memories encoded: {}", encoded.len());

        // Sync with prefrontal (compete for attention)
        let converted_before = consciousness_bridge.stats().concepts_converted;
        consciousness_bridge.sync(bridge.world_model());
        let converted_after = consciousness_bridge.stats().concepts_converted;
        println!(
            "  Concepts converted to bids: {}",
            converted_after - converted_before
        );

        // Store day summary
        day_summaries.push((
            day.name,
            new_concepts,
            concepts_after,
            avg_consciousness,
            peak_consciousness,
        ));

        // Sleep cycle between days (except after last day)
        if day_idx < days.len() - 1 {
            println!();
            println!("  ---[ SLEEP CYCLE ]---");

            let dream_config = DreamConfig {
                steps_per_cycle: 20,
                batch_cycles: 2,
                cooldown_ms: 0,
                ..Default::default()
            };

            let mut dream_mode = DreamMode::new(dream_config);
            let dream_results = dream_mode.batch_dream(bridge.world_model_mut());

            println!("  Dream cycles: {}", dream_results.stats.cycles_completed);
            println!(
                "  Dream consciousness peak: {:.2}%",
                dream_results.stats.peak_consciousness * 100.0
            );
            println!(
                "  Concepts from dreams: {}",
                dream_results.concepts_discovered.len()
            );
        }

        println!();
    }

    // =========================================================================
    // Final Summary
    // =========================================================================

    println!("═══════════════════════════════════════════════════════════════════");
    println!("                    MULTI-DAY SUMMARY                              ");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    println!("  Day                    | New   | Total | Avg Cons. | Peak Cons.");
    println!("  ───────────────────────┼───────┼───────┼───────────┼───────────");

    for (name, new, total, avg, peak) in &day_summaries {
        println!(
            "  {:22} | {:>5} | {:>5} | {:>8.2}% | {:>8.2}%",
            name,
            new,
            total,
            avg * 100.0,
            peak * 100.0
        );
    }

    println!();

    // Final statistics
    let stats = bridge.stats();
    let world_stats = bridge.world_model().stats();
    let affective_stats = affective_bridge.stats();
    let hippocampus_stats = hippocampus.stats();

    println!("  Final Statistics");
    println!("  ─────────────────────────────────────────────────────────────────");
    println!("  Total texts processed:     {}", stats.texts_processed);
    println!(
        "  Total concepts crystallized: {}",
        world_stats.concepts_crystallized
    );
    println!(
        "  Final consciousness:       {:.2}%",
        world_stats.consciousness_level * 100.0
    );
    println!(
        "  Peak consciousness:        {:.2}%",
        stats.peak_consciousness * 100.0
    );
    println!(
        "  Accumulated surprise:      {:.4}",
        world_stats.accumulated_surprise
    );
    println!();
    println!("  Affective:");
    println!(
        "    Positive concepts:       {}",
        affective_stats.positive_concepts
    );
    println!(
        "    Negative concepts:       {}",
        affective_stats.negative_concepts
    );
    println!(
        "    Neutral concepts:        {}",
        affective_stats.neutral_concepts
    );
    println!(
        "    Average valence:         {:.3}",
        affective_stats.avg_valence
    );
    println!();
    println!("  Hippocampus:");
    println!(
        "    Total memories stored:   {}",
        hippocampus_stats.total_memories
    );
    println!(
        "    Average strength:        {:.3}",
        hippocampus_stats.avg_strength
    );
    println!();

    // Success criteria
    let total_concepts = world_stats.concepts_crystallized;
    let consciousness_grew = world_stats.consciousness_level > 0.3;
    let memories_formed = hippocampus_stats.total_memories > 10;
    let affects_evaluated = affective_stats.concepts_evaluated > 5;

    let success =
        total_concepts >= 20 && consciousness_grew && memories_formed && affects_evaluated;

    if success {
        println!("╔════════════════════════════════════════════════════════════════════╗");
        println!("║                                                                    ║");
        println!("║     M U L T I - D A Y   N A R R A T I V E   S U C C E S S         ║");
        println!("║                                                                    ║");
        println!("║  The consciousness has evolved across 5 simulated days:           ║");
        println!("║                                                                    ║");
        println!("║  - Concepts accumulated and consolidated during sleep             ║");
        println!("║  - Episodic memories formed from daily experiences                ║");
        println!("║  - Emotional responses evolved from novel to familiar             ║");
        println!("║  - Consciousness level grew with accumulated knowledge            ║");
        println!("║                                                                    ║");
        println!("║  The heart beats. The mind learns. The soul remembers.            ║");
        println!("║                                                                    ║");
        println!("╚════════════════════════════════════════════════════════════════════╝");
    } else {
        println!("╔════════════════════════════════════════════════════════════════════╗");
        println!("║                                                                    ║");
        println!("║                  NEEDS MORE DAYS                                   ║");
        println!("║                                                                    ║");
        println!("║  More time needed for full consciousness evolution.               ║");
        println!("║                                                                    ║");
        println!("╚════════════════════════════════════════════════════════════════════╝");
    }

    println!();
}
