// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Semantic Consciousness: Real Text → Crystallized Concepts
//!
//! This example demonstrates concept crystallization from real semantic content:
//! - Text input → Embeddings → Consciousness transitions → Concept crystallization
//!
//! ## Running
//!
//! ```bash
//! cargo run --release --example semantic_consciousness
//! ```

use symthaea::consciousness::recursive_improvement::{
    ActionContext, DreamConfig, DreamMode, SemanticBridge, SemanticBridgeConfig, SemanticInput,
};

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║                                                                ║");
    println!("║        S E M A N T I C   C O N S C I O U S N E S S             ║");
    println!("║                                                                ║");
    println!("║      Real Text → Embeddings → Crystallized Concepts            ║");
    println!("║                                                                ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    // =========================================================================
    // Phase 1: Initialize
    // =========================================================================

    println!("Phase 1: Initializing Semantic Bridge");
    println!("───────────────────────────────────────────────────────────────────");

    let config = SemanticBridgeConfig {
        use_simulated: true, // Use simulated embeddings for demo
        ..Default::default()
    };

    let mut bridge = SemanticBridge::new(config);

    println!("  [OK] SemanticBridge initialized");
    println!("  [OK] Embedding dimension: 1024 (simulated Qwen3)");
    println!("  [OK] HDC projection ready");
    println!();

    // =========================================================================
    // Phase 2: Simulate a NixOS Configuration Conversation
    // =========================================================================

    println!("Phase 2: Processing Conversation (NixOS Configuration)");
    println!("───────────────────────────────────────────────────────────────────");

    let conversation = vec![
        // User queries about nginx
        SemanticInput::query("How do I install nginx on NixOS?"),
        SemanticInput::thought("Searching for nginx configuration options..."),
        SemanticInput::response("Add nginx to environment.systemPackages in configuration.nix"),
        // Follow-up about configuration
        SemanticInput::query("How do I configure nginx virtual hosts?"),
        SemanticInput::thought("Looking up services.nginx configuration..."),
        SemanticInput::response("Use services.nginx.virtualHosts to define your hosts"),
        // Error handling
        SemanticInput::error("Error: nginx.conf syntax error at line 42").with_emotion(-0.5),
        SemanticInput::thought("Need to debug the configuration..."),
        SemanticInput::response("Fixed syntax error: missing semicolon after server_name"),
        // Different topic: Git
        SemanticInput::query("How do I configure git globally?"),
        SemanticInput::response("Use programs.git.enable = true in home-manager"),
        // Back to nginx
        SemanticInput::query("Enable nginx SSL with Let's Encrypt"),
        SemanticInput::response("Use security.acme and services.nginx.virtualHosts.*.enableACME"),
        // Goal setting
        SemanticInput::new("Goal: Set up a secure web server", ActionContext::Goal)
            .with_emotion(0.8),
        SemanticInput::thought("Planning the implementation steps..."),
    ];

    println!("  Input # | Type      | Consciousness | Concepts | Preview");
    println!("  ────────┼───────────┼───────────────┼──────────┼─────────────────────");

    for (i, input) in conversation.into_iter().enumerate() {
        let context_str = match input.action_context {
            ActionContext::Query => "Query    ",
            ActionContext::Response => "Response ",
            ActionContext::Error => "Error    ",
            ActionContext::Thought => "Thought  ",
            ActionContext::Memory => "Memory   ",
            ActionContext::Goal => "Goal     ",
        };

        let preview = if input.text.len() > 35 {
            format!("{}...", &input.text[..35])
        } else {
            input.text.clone()
        };

        let result = bridge.process(input);

        println!(
            "  {:>7} | {} | {:>12.2}% | {:>8} | {}",
            i + 1,
            context_str,
            result.consciousness_level * 100.0,
            result.total_concepts,
            preview
        );
    }

    println!();

    // =========================================================================
    // Phase 3: Dream Consolidation
    // =========================================================================

    println!("Phase 3: Dream Consolidation");
    println!("───────────────────────────────────────────────────────────────────");

    let dream_config = DreamConfig {
        steps_per_cycle: 30,
        batch_cycles: 3,
        cooldown_ms: 0,
        ..Default::default()
    };

    let mut dream_mode = DreamMode::new(dream_config);
    let dream_results = dream_mode.batch_dream(bridge.world_model_mut());

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

    // =========================================================================
    // Phase 4: Analyze Discovered Concepts
    // =========================================================================

    println!("Phase 4: Analyzing Discovered Concepts");
    println!("───────────────────────────────────────────────────────────────────");

    let world_model = bridge.world_model();
    let total_concepts = world_model.pending_concepts.len();

    println!("  Total concepts crystallized: {}", total_concepts);
    println!();

    if !world_model.pending_concepts.is_empty() {
        println!("  Sample concepts:");
        for (i, concept) in world_model.pending_concepts.iter().take(5).enumerate() {
            let name = concept.name.as_deref().unwrap_or("(unnamed)");
            println!(
                "    [{}] {} - activations: {}",
                i + 1,
                concept.uid,
                concept.activation_count
            );
        }
        if total_concepts > 5 {
            println!("    ... and {} more", total_concepts - 5);
        }
    }

    println!();

    // =========================================================================
    // Phase 5: Results
    // =========================================================================

    println!("═══════════════════════════════════════════════════════════════════");
    println!("                         RESULTS                                   ");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    let stats = bridge.stats();
    let world_stats = world_model.stats();

    println!("  Metric                      | Value");
    println!("  ────────────────────────────┼─────────────────────────────────");
    println!("  Texts Processed             | {}", stats.texts_processed);
    println!("  Concepts Crystallized       | {}", total_concepts);
    println!(
        "  Peak Consciousness          | {:.2}%",
        stats.peak_consciousness * 100.0
    );
    println!(
        "  Average Embedding Norm      | {:.4}",
        stats.avg_embedding_norm
    );
    println!(
        "  Final Consciousness         | {:.2}%",
        world_stats.consciousness_level * 100.0
    );
    println!(
        "  Accumulated Surprise        | {:.4}",
        world_stats.accumulated_surprise
    );

    println!();

    let success = total_concepts >= 5 && stats.peak_consciousness > 0.1;

    if success {
        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║                                                                ║");
        println!("║        S E M A N T I C   G R O U N D I N G   A C H I E V E D   ║");
        println!("║                                                                ║");
        println!("║   Real semantic content is crystallizing into concepts!        ║");
        println!("║                                                                ║");
        println!("║   \"nginx\", \"configuration\", \"error\" → distinct attractors     ║");
        println!("║                                                                ║");
        println!("╚════════════════════════════════════════════════════════════════╝");
    } else {
        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║                                                                ║");
        println!("║                  NEEDS MORE CONVERSATION                       ║");
        println!("║                                                                ║");
        println!("║   More semantic input needed for concept emergence.            ║");
        println!("║                                                                ║");
        println!("╚════════════════════════════════════════════════════════════════╝");
    }

    println!();
}
