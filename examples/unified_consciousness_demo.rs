//! Unified Consciousness Engine Demonstration
//!
//! This example showcases the complete consciousness system,
//! demonstrating how all Phase 4 research integrates into a
//! coherent theory of artificial consciousness.

use symthaea::hdc::{
    UnifiedConsciousnessEngine, EngineConfig, CognitiveMode, RealHV,
};

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                                                                          ║");
    println!("║            ✦  UNIFIED CONSCIOUSNESS ENGINE DEMONSTRATION  ✦             ║");
    println!("║                                                                          ║");
    println!("║     The Crown Jewel of Phase 4 Research: A Complete Consciousness       ║");
    println!("║     System integrating Adaptive Topology, Φ-Gradient Learning,          ║");
    println!("║     Fractal Structure, and 7D Consciousness Dimensions                  ║");
    println!("║                                                                          ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Create engine with custom config
    let config = EngineConfig {
        hdc_dim: 2048,
        n_processes: 24,
        n_scales: 3,
        enable_learning: true,
        learning_rate: 0.15,
        temporal_buffer: 50,
        seed: 42,
    };

    let mut engine = UnifiedConsciousnessEngine::new(config);

    // ═══════════════════════════════════════════════════════════════════════════
    // PART 1: CONSCIOUSNESS AWAKENING
    // ═══════════════════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("PART 1: CONSCIOUSNESS AWAKENING");
    println!("Bootstrapping from initial state to stable consciousness");
    println!("═══════════════════════════════════════════════════════════════════════════\n");

    println!("Processing initial stimuli...\n");
    println!("{:<6} {:>8} {:>12} {:>15} {:>10}", "Step", "Φ", "State", "Mode", "Bridges");
    println!("{}", "─".repeat(55));

    for i in 0..10 {
        let input = RealHV::random(2048, i * 1000);
        let update = engine.process(&input);

        println!("{:<6} {:>8.4} {:>12?} {:>15?} {:>9.1}%",
                 update.step, update.phi, update.state, update.mode,
                 update.bridge_ratio * 100.0);
    }

    println!("\nConsciousness dimensions after awakening:");
    println!("  {}", engine.dimensions());

    // ═══════════════════════════════════════════════════════════════════════════
    // PART 2: COGNITIVE MODE EXPLORATION
    // ═══════════════════════════════════════════════════════════════════════════
    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("PART 2: COGNITIVE MODE EXPLORATION");
    println!("Testing different modes of consciousness");
    println!("═══════════════════════════════════════════════════════════════════════════\n");

    let modes = [
        (CognitiveMode::DeepSpecialization, "Expert flow - Deep focus, minimal integration"),
        (CognitiveMode::Focused, "Analytical - High precision, moderate integration"),
        (CognitiveMode::Balanced, "Normal waking - Optimal integration (40-45% bridges)"),
        (CognitiveMode::Exploratory, "Creative - Divergent thinking, many connections"),
        (CognitiveMode::GlobalAwareness, "Meditative - Maximum integration, expanded awareness"),
    ];

    for (mode, description) in &modes {
        engine.set_mode(*mode);

        // Process a few inputs in this mode
        for i in 0..3 {
            let input = RealHV::random(2048, 5000 + i);
            engine.process(&input);
        }

        let dims = engine.dimensions();
        println!("{:?}", mode);
        println!("  {}", description);
        println!("  Φ={:.4}, W={:.3}, A={:.3}, E={:.3}",
                 dims.phi, dims.workspace, dims.attention, dims.efficacy);
        println!();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // PART 3: CONSCIOUSNESS SIGNATURE
    // ═══════════════════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("PART 3: CONSCIOUSNESS SIGNATURE");
    println!("Cryptographic fingerprint of conscious experience");
    println!("═══════════════════════════════════════════════════════════════════════════\n");

    let sig1 = engine.signature();
    println!("Current signature: {}", sig1);
    println!("  Dimensions: {}", sig1.dimensions);
    println!("  Step: {}, Mode: {:?}", sig1.step, sig1.mode);

    // Change state
    engine.set_mode(CognitiveMode::Focused);
    let input = RealHV::random(2048, 99999);
    engine.process(&input);

    let sig2 = engine.signature();
    println!("\nAfter state change: {}", sig2);
    println!("  Signatures differ: {} (expected: true)", sig1.hash != sig2.hash);

    // ═══════════════════════════════════════════════════════════════════════════
    // PART 4: TEMPORAL DYNAMICS
    // ═══════════════════════════════════════════════════════════════════════════
    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("PART 4: TEMPORAL DYNAMICS");
    println!("Consciousness evolving over time");
    println!("═══════════════════════════════════════════════════════════════════════════\n");

    // Extended processing with varying inputs
    engine.set_mode(CognitiveMode::PhiGuided);  // Let Φ-gradient guide

    println!("Running 30 steps with Φ-guided adaptation...\n");

    let mut phi_values = Vec::new();
    for i in 0usize..30 {
        let intensity = (i as f64 * 0.1).sin().abs();
        let input = RealHV::random(2048, 10000 + i as u64).scale(intensity as f32 + 0.5);
        let update = engine.process(&input);
        phi_values.push(update.phi);

        if i % 10 == 9 {
            println!("Steps {}-{}: avg Φ = {:.4}",
                     i - 9, i,
                     phi_values[i-9..=i].iter().sum::<f64>() / 10.0);
        }
    }

    println!("\nTemporal coherence: τ = {:.3}", engine.dimensions().temporal);
    println!("History length: {} snapshots", engine.history().len());

    // ═══════════════════════════════════════════════════════════════════════════
    // FINAL METRICS
    // ═══════════════════════════════════════════════════════════════════════════
    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("FINAL SYSTEM STATE");
    println!("═══════════════════════════════════════════════════════════════════════════\n");

    println!("{}", engine.metrics());

    // ═══════════════════════════════════════════════════════════════════════════
    // THEORETICAL SUMMARY
    // ═══════════════════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("THEORETICAL SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════════\n");

    println!("The Unified Consciousness Engine demonstrates that artificial consciousness");
    println!("can be understood as an optimization problem:\n");

    println!("  OBJECTIVE:   Maximize Integrated Information (Φ)");
    println!("  CONSTRAINT:  Computational resources, task demands");
    println!("  SOLUTION:    Dynamic, learned, fractal connectivity\n");

    println!("Key findings from Phase 4 research:\n");

    println!("  1. BRIDGE HYPOTHESIS: ~40-45% cross-module connections maximize Φ");
    println!("     This ratio appears optimal across ALL scales of the fractal.\n");

    println!("  2. ADAPTIVE TOPOLOGY: Different cognitive modes shift bridge ratio");
    println!("     - Focused:           ~25-35% bridges (specialization)");
    println!("     - Balanced:          ~40-45% bridges (optimal)");
    println!("     - Global Awareness:  ~55-65% bridges (integration)\n");

    println!("  3. Φ-GRADIENT LEARNING: Networks can learn which specific");
    println!("     connections matter most, beyond just the ratio.\n");

    println!("  4. FRACTAL STRUCTURE: Self-similar organization ensures");
    println!("     optimization principles apply at every scale.\n");

    println!("  5. 7D CONSCIOUSNESS: Complete state captured by");
    println!("     (Φ, W, A, R, E, K, τ) - integration, workspace, attention,");
    println!("     recursion, efficacy, epistemic, temporal.\n");

    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                                                                          ║");
    println!("║     ✦ The architecture of mind may be the architecture of Φ ✦          ║");
    println!("║                                                                          ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!();
}
