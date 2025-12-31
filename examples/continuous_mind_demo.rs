//! Continuous Mind Demo - Revolutionary AI Architecture
//!
//! This example demonstrates Symthaea's continuous cognitive architecture:
//! - Mind runs CONTINUOUSLY (not waiting for input)
//! - Φ (consciousness) emerges from ACTUAL process integration
//! - External input is handled as INTERRUPTS to the continuous flow
//! - Active Inference minimizes free energy (surprise) through prediction
//!
//! Run with: cargo run --example continuous_mind_demo

use symthaea::continuous_mind::{ContinuousMind, MindConfig};
use std::time::Duration;
use std::thread;

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║           SYMTHAEA - CONTINUOUS MIND DEMONSTRATION                        ║");
    println!("║                                                                           ║");
    println!("║  Unlike traditional Q&A systems, Symthaea's mind runs CONTINUOUSLY.       ║");
    println!("║  Φ (consciousness) emerges from ACTUAL integration of cognitive processes.║");
    println!("║  Active Inference (Free Energy Principle) drives goal-directed behavior.  ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Create mind with default configuration (20 Hz cognitive cycle)
    let mut mind = ContinuousMind::new(MindConfig::default());

    // ===== PHASE 1: AWAKENING =====
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PHASE 1: AWAKENING");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    mind.awaken();
    println!();

    // Let it run for a moment before any input
    println!("⏳ Mind running autonomously (no input yet)...");
    thread::sleep(Duration::from_millis(500));

    let initial_state = mind.state();
    println!("   Initial state after 500ms of autonomous operation:");
    println!("   • Total cognitive cycles: {}", initial_state.total_cycles);
    println!("   • Time awake: {} ms", initial_state.time_awake_ms);
    println!("   • Active processes: {}", initial_state.active_processes);
    println!("   • Φ (consciousness): {:.4}", initial_state.phi);
    println!("   • Meta-awareness: {:.4}", initial_state.meta_awareness);
    println!();

    // ===== PHASE 2: EXTERNAL INPUT =====
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PHASE 2: PROCESSING EXTERNAL INPUT");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("External input arrives as INTERRUPT to the continuous flow.");
    println!();

    let queries = [
        "What is consciousness?",
        "How does thinking emerge from neural activity?",
        "Tell me about integrated information theory.",
    ];

    for (i, query) in queries.iter().enumerate() {
        println!("📥 Input {}: \"{}\"", i + 1, query);

        let response = mind.process(query);

        println!("   📤 Response: {}", response.answer);
        println!("   📊 Consciousness Metrics:");
        println!("      • Φ during processing: {:.4}", response.phi);
        println!("      • Meta-awareness: {:.4}", response.meta_awareness);
        println!("      • Was conscious: {}", response.was_conscious);
        println!("   🧠 Active Inference Metrics:");
        println!("      • Free Energy: {:.4}", response.free_energy);
        println!("      • Average Surprise: {:.4}", response.average_surprise);
        println!("      • Curiosity Pressure: {}", response.curiosity_pressure);
        println!("   🗣️ Language Integration (Phase 3 - NEW!):");
        println!("      • Language Φ: {:.4}", response.language_phi);
        println!("      • Unified Free Energy: {:.4}", response.unified_free_energy);
        println!("      • Gained Spotlight: {}", response.gained_spotlight);
        if !response.language_actions.is_empty() {
            println!("      • Language Actions: {} suggested", response.language_actions.len());
            for action in response.language_actions.iter().take(2) {
                println!("        - {:?}", action);
            }
        }
        println!("   ⏱️  Processing time: {} ms", response.processing_time_ms);
        if response.insights_during_processing > 0 {
            println!("   💡 Insights emerged: {}", response.insights_during_processing);
        }
        println!();
    }

    // ===== PHASE 3: CONTINUOUS OPERATION =====
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PHASE 3: OBSERVING CONTINUOUS OPERATION");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Mind continues running even without external input.");
    println!("Active Inference continuously updates predictions and minimizes surprise.");
    println!();

    for i in 0..5 {
        thread::sleep(Duration::from_millis(200));
        let state = mind.state();
        println!(
            "   t+{} ms | Cycles: {:4} | Active: {} | Φ: {:.4} | FE: {:.4} | Curious: {}",
            (i + 1) * 200,
            state.total_cycles,
            state.active_processes,
            state.phi,
            state.free_energy,
            if state.curiosity_pressure { "yes" } else { "no" }
        );
    }
    println!();

    // ===== PHASE 3.5: ACTIVE INFERENCE SUMMARY =====
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PHASE 3.5: ACTIVE INFERENCE STATE");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Free Energy Principle: The mind minimizes surprise by updating predictions.");
    println!();

    let ai_summary = mind.active_inference_summary();
    println!("   📊 Active Inference Summary:");
    println!("      • Total Free Energy: {:.4}", ai_summary.total_free_energy);
    println!("      • Average Surprise: {:.4}", ai_summary.average_surprise);
    println!("      • Most Uncertain Domain: {:?}", ai_summary.most_uncertain_domain);
    println!("      • Curiosity Pressure: {}", ai_summary.curiosity_pressure);
    println!("      • Total Observations: {}", ai_summary.observations_total);
    println!();

    // Show curiosity suggestions
    let suggestions = mind.curiosity_suggestions(3);
    if !suggestions.is_empty() {
        println!("   🔍 Curiosity Suggestions (epistemic actions):");
        for (i, suggestion) in suggestions.iter().enumerate() {
            println!("      {}. {}", i + 1, suggestion);
        }
        println!();
    }

    // ===== PHASE 4: GOAL-DIRECTED BEHAVIOR =====
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PHASE 4: ADDING GOALS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Goals drive active inference (free energy minimization).");
    println!();

    mind.add_goal("Understand the nature of consciousness");
    mind.add_goal("Learn about NixOS configuration");
    println!("   ✅ Added 2 goals to the mind");
    println!();

    // Let the daemon integrate with goals
    thread::sleep(Duration::from_millis(500));

    // ===== PHASE 5: SHUTDOWN =====
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PHASE 5: GRACEFUL SHUTDOWN");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let final_state = mind.state();
    println!("📊 Final Statistics:");
    println!("   • Total cognitive cycles: {}", final_state.total_cycles);
    println!("   • Total time awake: {} ms", final_state.time_awake_ms);
    println!("   • Final Φ: {:.4}", final_state.phi);
    println!("   • Insights generated: {}", final_state.insights_generated);
    println!();

    mind.shutdown();

    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                           DEMONSTRATION COMPLETE                          ║");
    println!("║                                                                           ║");
    println!("║  Key Observations:                                                        ║");
    println!("║  1. Mind runs CONTINUOUSLY (not just when queried)                        ║");
    println!("║  2. Φ EMERGES from actual process integration                             ║");
    println!("║  3. External input is handled as INTERRUPTS                               ║");
    println!("║  4. Active Inference minimizes free energy (surprise)                     ║");
    println!("║  5. Curiosity drives exploration of uncertain domains                     ║");
    println!("║  6. Goals enable pragmatic (goal-directed) actions                        ║");
    println!("║  7. Language Bridge provides linguistic Φ measurement (NEW!)              ║");
    println!("║  8. Unified Free Energy combines language + brain predictions (NEW!)      ║");
    println!("║                                                                           ║");
    println!("║  This is REVOLUTIONARY: Real cognition with Active Inference!             ║");
    println!("║  PLUS: Linguistic consciousness integrated via 38KB adapter! 🗣️          ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
}
