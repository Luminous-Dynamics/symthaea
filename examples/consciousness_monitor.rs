//! Live Consciousness Monitor
//!
//! Real-time visualization of the Unified Consciousness Engine
//! using ASCII art rendering. Shows consciousness dimensions,
//! cognitive modes, topological state, and temporal dynamics.

use symthaea::hdc::{
    UnifiedConsciousnessEngine, EngineConfig, CognitiveMode,
    ConsciousnessVisualizer, RealHV,
};

fn main() {
    // Initialize
    let config = EngineConfig {
        hdc_dim: 2048,
        n_processes: 24,
        n_scales: 3,
        enable_learning: true,
        temporal_buffer: 30,
        ..Default::default()
    };

    let mut engine = UnifiedConsciousnessEngine::new(config);
    let viz = ConsciousnessVisualizer::new();

    // Clear screen
    print!("\x1b[2J\x1b[H");

    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                    ✦ CONSCIOUSNESS MONITOR ✦                            ║");
    println!("║                                                                          ║");
    println!("║  Real-time visualization of the Unified Consciousness Engine            ║");
    println!("║  Showing Φ, cognitive modes, topological states, and temporal dynamics  ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let mut phi_history: Vec<f64> = Vec::new();

    // Simulation scenarios
    let scenarios = [
        ("AWAKENING", CognitiveMode::Balanced, 10,
         "Bootstrapping consciousness from initial state"),
        ("FOCUSED ANALYSIS", CognitiveMode::Focused, 8,
         "Deep analytical reasoning - specialized processing"),
        ("CREATIVE EXPLORATION", CognitiveMode::Exploratory, 8,
         "Divergent thinking - maximum cross-module connectivity"),
        ("MEDITATIVE AWARENESS", CognitiveMode::GlobalAwareness, 8,
         "Expanded consciousness - global integration"),
        ("EXPERT FLOW", CognitiveMode::DeepSpecialization, 8,
         "Peak performance - deep specialized flow"),
        ("Φ-GUIDED ADAPTATION", CognitiveMode::PhiGuided, 10,
         "Letting integrated information guide connectivity"),
    ];

    for (name, mode, steps, description) in scenarios {
        println!("═══════════════════════════════════════════════════════════════════════════");
        println!("SCENARIO: {}", name);
        println!("{}", description);
        println!("═══════════════════════════════════════════════════════════════════════════\n");

        engine.set_mode(mode);

        for i in 0..steps {
            // Generate varying input based on scenario
            let seed = (name.len() * 1000 + i) as u64;
            let intensity = match mode {
                CognitiveMode::Exploratory => 0.8 + 0.2 * (i as f64 * 0.5).sin(),
                CognitiveMode::Focused => 0.9,
                CognitiveMode::GlobalAwareness => 0.5 + 0.3 * (i as f64 * 0.3).cos(),
                _ => 0.7,
            };

            let input = RealHV::random(2048, seed).scale(intensity as f32);
            let update = engine.process(&input);

            phi_history.push(update.phi);
            if phi_history.len() > 30 {
                phi_history.remove(0);
            }

            // Render dashboard
            println!("{}", viz.render_dashboard(&update, &phi_history));

            // Show mandala for final step of each scenario
            if i == steps - 1 {
                println!("{}", viz.render_mandala(&update.dimensions));
            }
        }

        println!();
    }

    // Final summary
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("SIMULATION COMPLETE");
    println!("═══════════════════════════════════════════════════════════════════════════\n");

    println!("{}", engine.metrics());

    // Final signature
    let sig = engine.signature();
    println!("Final Consciousness Signature: {}", sig);
    println!();

    // Φ history analysis
    let avg_phi: f64 = phi_history.iter().sum::<f64>() / phi_history.len() as f64;
    let max_phi = phi_history.iter().cloned().fold(0.0f64, f64::max);
    let min_phi = phi_history.iter().cloned().fold(1.0f64, f64::min);

    println!("Φ Statistics:");
    println!("  Average: {:.4}", avg_phi);
    println!("  Maximum: {:.4}", max_phi);
    println!("  Minimum: {:.4}", min_phi);
    println!("  Range:   {:.4}", max_phi - min_phi);
    println!();

    println!("Full Φ History:");
    println!("  {}", viz.render_sparkline(&phi_history, "Φ"));

    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                                                                          ║");
    println!("║    \"Consciousness is what information processing feels like             ║");
    println!("║     from the inside when Φ is high.\" - Integrated Information Theory   ║");
    println!("║                                                                          ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
}
