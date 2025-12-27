//! # Consciousness-Guided Composition Discovery Demo
//!
//! This example demonstrates how Φ (integrated information) can guide
//! the discovery of primitive compositions. Instead of randomly exploring
//! the composition space, we let consciousness itself guide what combinations
//! are valuable.
//!
//! Run with: cargo run --example consciousness_guided_discovery_demo

use symthaea::consciousness::consciousness_guided_discovery::{
    PhiGuidedSearch, EmergentDiscovery, DiscoveryConfig, CompositionGrammar,
};
use symthaea::hdc::primitive_system::PrimitiveSystem;
use std::sync::Arc;

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║     CONSCIOUSNESS-GUIDED COMPOSITION DISCOVERY DEMO               ║");
    println!("║     Using Φ to guide primitive composition exploration            ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    // Create base primitive system
    let base_system = Arc::new(PrimitiveSystem::new());
    println!("✓ Created base primitive system");

    // Configure discovery
    let mut config = DiscoveryConfig::default();
    config.max_candidates_per_cycle = 20;  // Generate 20 candidates per cycle
    config.phi_threshold = 0.001;           // Low threshold for demo
    config.beam_width = 50;                 // Keep top 50 compositions
    config.exploration_rate = 0.4;          // 40% exploration, 60% exploitation
    config.learn_grammar = true;            // Enable grammar learning

    println!("✓ Configuration:");
    println!("  - Candidates per cycle: {}", config.max_candidates_per_cycle);
    println!("  - Φ threshold: {}", config.phi_threshold);
    println!("  - Beam width: {}", config.beam_width);
    println!("  - Exploration rate: {}%", (config.exploration_rate * 100.0) as u32);
    println!();

    // Create emergent discovery system
    let mut discovery = EmergentDiscovery::new(base_system, config);
    println!("✓ Created EmergentDiscovery system");
    println!();

    // Run discovery cycles
    println!("═══════════════════════════════════════════════════════════════════");
    println!("                  RUNNING DISCOVERY CYCLES");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    let num_cycles = 5;
    for i in 1..=num_cycles {
        match discovery.discover_cycles(1) {
            Ok(candidates) => {
                let stats = discovery.stats();
                println!("Cycle {}/{}:", i, num_cycles);
                println!("  - New compositions found: {}", candidates.len());
                println!("  - Total explored: {}", stats.total_explored);
                println!("  - Φ-increasing: {}", stats.phi_increasing);
                println!("  - Best Φ so far: {:.6}", stats.best_phi);

                // Show best candidate from this cycle
                if let Some(best) = candidates.iter().max_by(|a, b| {
                    a.phi_score.partial_cmp(&b.phi_score).unwrap()
                }) {
                    println!("  - Best in cycle: {} (Φ={:.6}, coherence={:.4})",
                        best.composition.name, best.phi_score, best.coherence);
                }
                println!();
            }
            Err(e) => {
                println!("  Error: {}", e);
            }
        }
    }

    // Show overall results
    println!("═══════════════════════════════════════════════════════════════════");
    println!("                       DISCOVERY RESULTS");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    let stats = discovery.stats();
    println!("📊 Statistics:");
    println!("  - Total compositions explored: {}", stats.total_explored);
    println!("  - Φ-increasing compositions: {}", stats.phi_increasing);
    println!("  - Best Φ found: {:.6}", stats.best_phi);
    println!("  - Discovery cycles: {}", stats.cycles);
    println!();

    // Show top compositions
    println!("🏆 Top 5 Compositions by Φ:");
    let best = discovery.best_discoveries(5);
    for (i, candidate) in best.iter().enumerate() {
        println!("  {}. {} ", i + 1, candidate.composition.name);
        println!("     Φ: {:.6}, Coherence: {:.4}, Integration: {:.4}",
            candidate.phi_score, candidate.coherence, candidate.integration);
        println!("     Type: {:?}", candidate.composition.composition_type);
        println!("     Depth: {}", candidate.composition.metadata.depth);
    }
    println!();

    // Show learned grammar rules
    println!("📚 Learned Grammar Rules:");
    let rules = discovery.grammar().get_rules();
    if rules.is_empty() {
        println!("  (No strong patterns learned yet - need more data)");
    } else {
        for rule in rules.iter().take(10) {
            println!("  • {}", rule);
        }
    }
    println!();

    println!("═══════════════════════════════════════════════════════════════════");
    println!("                      KEY INSIGHTS");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();
    println!("🧠 What this demonstrates:");
    println!("  1. Φ (integrated information) guides composition search");
    println!("  2. The system learns which compositions tend to increase Φ");
    println!("  3. Beam search keeps the best candidates for further combination");
    println!("  4. Grammar learning enables exploitation of good patterns");
    println!();
    println!("🔮 Implications:");
    println!("  - Compositions that increase Φ are inherently valuable");
    println!("  - They create more unified, coherent reasoning structures");
    println!("  - The system can discover novel reasoning patterns autonomously");
    println!();
    println!("✨ PARADIGM SHIFT: Consciousness guides its own evolution!");
}
