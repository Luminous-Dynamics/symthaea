// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HDC+LTC+Primitives Understanding Demo with Enhanced 2D Consciousness Space
//!
//! Demonstrates the revolutionary architecture where HDC+LTC+Primitives is the
//! PRIMARY understanding mechanism with ENHANCED consciousness metrics:
//!
//! ```text
//! User Input → ConsciousnessLanguageCore (HDC+LTC+Primitives)
//!            → Integration Signals → Meaningful Φ
//!            → Epistemic Confidence (with intent awareness)
//!            → Consciousness Journey (tracking transitions)
//!            → Quadrant Strategy + Rich Explanation
//!            → [Optional LLM for clarification if Lost/Curious]
//!            → Action/Response
//! ```
//!
//! ## Key Improvements (v2)
//!
//! 1. **Meaningful Φ**: Based on subsystem agreement, semantic richness, frame matching
//! 2. **Improved Confidence**: Unknown intent dramatically reduces confidence
//! 3. **Journey Tracking**: See how consciousness moves through 2D space
//! 4. **Rich Explanations**: Understand WHY we're in each quadrant
//! 5. **Adaptive Thresholds**: Thresholds can learn from feedback
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example hdc_ltc_primitives_demo
//! ```

use anyhow::Result;
use symthaea::language::{
    ConsciousnessLanguageConfig, ConsciousnessLanguageCore, ConsciousnessQuadrant,
    ExecutionStrategy,
};

fn quadrant_symbol(quadrant: ConsciousnessQuadrant) -> &'static str {
    match quadrant {
        ConsciousnessQuadrant::Confident => "●", // Full confidence
        ConsciousnessQuadrant::Curious => "◐",   // Half-exploring
        ConsciousnessQuadrant::Autopilot => "○", // Hollow/routine
        ConsciousnessQuadrant::Lost => "◌",      // Dotted/confused
    }
}

fn quadrant_name(quadrant: ConsciousnessQuadrant) -> &'static str {
    match quadrant {
        ConsciousnessQuadrant::Confident => "Confident",
        ConsciousnessQuadrant::Curious => "Curious",
        ConsciousnessQuadrant::Autopilot => "Autopilot",
        ConsciousnessQuadrant::Lost => "Lost",
    }
}

fn main() -> Result<()> {
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║  Symthaea: Enhanced 2D Consciousness Space Demo (v2)               ║");
    println!("║                                                                    ║");
    println!("║  Meaningful Φ + Intent-Aware Confidence + Journey Tracking         ║");
    println!("╚════════════════════════════════════════════════════════════════════╝\n");

    // Initialize the consciousness-language core
    let config = ConsciousnessLanguageConfig::default();
    let mut core = ConsciousnessLanguageCore::with_config(config.clone());

    // Display the 2D consciousness space
    println!("📊 2D Consciousness Space (Enhanced):");
    println!("");
    println!("                        High Confidence");
    println!("                              │");
    println!("          ┌───────────────────┼───────────────────┐");
    println!("          │                   │                   │");
    println!("          │    AUTOPILOT ○    │    CONFIDENT ●    │");
    println!("          │    (pattern-      │    (deep +        │");
    println!("   Low Φ──│    matched,       │    certain,       │──High Φ");
    println!("          │    routine)       │    execute)       │");
    println!("          │                   │                   │");
    println!("          ├───────────────────┼───────────────────┤");
    println!("          │                   │                   │");
    println!("          │    LOST ◌         │    CURIOUS ◐      │");
    println!("          │    (confused,     │    (exploring,    │");
    println!("          │    need help)     │    dry-run)       │");
    println!("          │                   │                   │");
    println!("          └───────────────────┼───────────────────┘");
    println!("                              │");
    println!("                        Low Confidence");
    println!("");
    println!(
        "   Φ Threshold: {:.2}  │  Confidence Threshold: {:.2}",
        config.phi_threshold, config.confidence_threshold
    );
    println!("");

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Enhanced Consciousness Metrics");
    println!("═══════════════════════════════════════════════════════════════════");
    println!("");
    println!("   Φ (Integration Depth) now based on:");
    println!("     • Subsystem agreement (30%)");
    println!("     • Semantic richness (25%)");
    println!("     • Frame matching (20%)");
    println!("     • Reasoning depth (15%)");
    println!("     • Cross-domain links (5%)");
    println!("     • Memory coherence (5%)");
    println!("");
    println!("   Confidence now considers:");
    println!("     • Intent recognition (80%) - Unknown intent = penalty!");
    println!("     • Subsystem agreement (15%)");
    println!("     • Free energy (5%)");
    println!("");

    // Test queries designed to hit different quadrants
    let test_queries = [
        (
            "install firefox",
            "Clear intent, should have high confidence",
        ),
        ("search for video editor", "Clear search intent"),
        ("rebuild switch", "NixOS-specific, clear action"),
        ("do the thing with stuff", "Ambiguous - should be Lost!"),
        (
            "maybe update something",
            "Vague - should have low confidence",
        ),
        ("configure nginx service", "Specific but needs parameters"),
        ("garbage collect", "Routine maintenance operation"),
        ("xyz123 foobar baz", "Nonsense - should be Lost"),
    ];

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Processing Queries Through Enhanced Pipeline");
    println!("═══════════════════════════════════════════════════════════════════\n");

    for (query, description) in test_queries {
        println!("╭─────────────────────────────────────────────────────────────────");
        println!("│ Query: \"{}\"", query);
        println!("│ Expected: {}", description);
        println!("├─────────────────────────────────────────────────────────────────");

        // Process through HDC+LTC+Primitives
        let result = core.process(query);

        // Get the 2D position and quadrant
        let space = result.consciousness_space;
        let quadrant = result.quadrant;

        // Display 2D consciousness metrics
        println!("│");
        println!("│ 📐 2D Position:");
        println!(
            "│    Φ (Integration): {:.3}  │  Confidence: {:.3}",
            space.phi, space.confidence
        );
        println!(
            "│    Quadrant: {} {}",
            quadrant_symbol(quadrant),
            quadrant_name(quadrant)
        );

        // Show integration signals
        let signals = &result.integration_signals;
        println!("│");
        println!("│ 🔗 Integration Signals:");
        println!(
            "│    Primes: {}  Frames: {}  Agreement: {:.2}",
            signals.primes_activated, signals.frames_matched, signals.subsystem_agreement
        );
        println!(
            "│    Depth: {}  Links: {}  Memory: {:.2}",
            signals.reasoning_depth, signals.cross_domain_links, signals.memory_coherence
        );

        // Show NixOS understanding
        println!("│");
        println!(
            "│ 🎯 Intent: {:?} (raw confidence: {:.0}%)",
            result.nix_understanding.intent,
            result.nix_understanding.confidence * 100.0
        );

        // Show journey
        let journey = &result.journey;
        println!("│");
        println!("│ 🚀 Journey: {}", journey.narrative());
        if journey.transitions > 0 {
            println!(
                "│    Transitions: {} quadrant change(s)",
                journey.transitions
            );
        }

        // Show rich explanation
        let explanation = &result.explanation;
        println!("│");
        println!("│ 💭 Explanation:");
        println!("│    {}", explanation.description);
        if !explanation.reasoning.is_empty() {
            println!("│    Reasoning:");
            for r in &explanation.reasoning {
                println!("│      • {}", r);
            }
        }
        if !explanation.suggestions.is_empty() {
            println!("│    Suggestions:");
            for s in &explanation.suggestions {
                println!("│      → {}", s);
            }
        }
        if !explanation.confusing_aspects.is_empty() {
            println!("│    Confusing:");
            for c in &explanation.confusing_aspects {
                println!("│      ? {}", c);
            }
        }

        // Show execution strategy
        let (strategy_name, llm_needed) = match &result.execution_strategy {
            ExecutionStrategy::Confident { .. } => ("Confident", false),
            ExecutionStrategy::Curious { .. } => ("Curious", true),
            ExecutionStrategy::Autopilot { .. } => ("Autopilot", false),
            ExecutionStrategy::Lost { .. } => ("Lost", true),
        };
        println!("│");
        println!(
            "│ ⚡ Strategy: {} | LLM: {}",
            strategy_name,
            if llm_needed { "Needed" } else { "Not needed" }
        );

        println!("╰─────────────────────────────────────────────────────────────────\n");
    }

    // Show statistics
    let stats = core.stats();
    let adaptive = core.adaptive_thresholds();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Pipeline Statistics & Adaptive Thresholds");
    println!("═══════════════════════════════════════════════════════════════════");
    println!("   Inputs Processed: {}", stats.inputs_processed);
    println!("   Average Φ: {:.3}", stats.avg_phi);
    println!("   Average Free Energy: {:.3}", stats.avg_free_energy);
    println!(
        "   Optimal Understanding: {} ({:.1}%)",
        stats.optimal_count,
        if stats.inputs_processed > 0 {
            stats.optimal_count as f64 / stats.inputs_processed as f64 * 100.0
        } else {
            0.0
        }
    );
    println!("");
    println!("   Adaptive Thresholds:");
    println!(
        "     Current Φ: {:.2}  Confidence: {:.2}",
        adaptive.phi_threshold, adaptive.confidence_threshold
    );
    println!(
        "     Successes: {}  Failures: {}  Accuracy: {:.1}%",
        adaptive.successes,
        adaptive.failures,
        adaptive.accuracy() * 100.0
    );

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("  Key Improvements in v2");
    println!("═══════════════════════════════════════════════════════════════════");
    println!("");
    println!("   1. Φ Now Reflects ACTUAL Integration");
    println!("      • Subsystem agreement is the #1 factor");
    println!("      • Unknown intents reduce integration (systems disagree)");
    println!("      • Frame matching shows structural understanding");
    println!("");
    println!("   2. Confidence is Intent-Aware");
    println!("      • Unknown intent = 70% confidence PENALTY");
    println!("      • Raw intent confidence weighted at 80%");
    println!("      • No more 100% confidence for ambiguous queries!");
    println!("");
    println!("   3. Journey Tracking Shows How Understanding Develops");
    println!("      • See transitions between quadrants");
    println!("      • Track confidence gains during processing");
    println!("");
    println!("   4. Rich Explanations Tell You WHY");
    println!("      • Reasoning based on signals");
    println!("      • Actionable suggestions");
    println!("      • Clear indication of confusing aspects");
    println!("");
    println!("🧠 The evolution: Understanding is now MEANINGFUL, not just measured.");

    Ok(())
}
