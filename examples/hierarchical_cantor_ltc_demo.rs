//! # Hierarchical Cantor-LTC/HDC Network Demonstration
//!
//! This example demonstrates the revolutionary 7-level self-similar
//! architecture combining:
//! - Cantor Set Topology (τ ratio = 1/3)
//! - Liquid Time-Constant Networks
//! - Hyperdimensional Computing (16,384D)
//!
//! Run with: cargo run --example hierarchical_cantor_ltc_demo --release

use symthaea::hierarchical_cantor_ltc::{HierarchicalCantorLtcNetwork, CantorLtcConfig};
use symthaea::hdc::{RealHV, HDC_DIMENSION};
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     Hierarchical Cantor-LTC/HDC Network Demonstration        ║");
    println!("║                                                              ║");
    println!("║  7-Level Self-Similar Architecture for Meta-Cognition        ║");
    println!("║  HDC Binding (⊗) Replaces Matrix Multiplication in LTC       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // =========================================================================
    // PHASE 1: Network Creation
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  PHASE 1: Creating Hierarchical Cantor-LTC Network");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let config = CantorLtcConfig {
        dimension: HDC_DIMENSION,
        max_depth: 7,         // 7 levels for meta-cognitive capability
        tau_ratio: 1.0 / 3.0, // Cantor ternary ratio
        base_tau: 1000.0,     // 1 second at root
        learning_rate: 0.01,
        measure_phi: true,
    };

    let start = Instant::now();
    let mut network = HierarchicalCantorLtcNetwork::new(config.clone());
    let creation_time = start.elapsed();

    println!("  Configuration:");
    println!("    • HDC Dimension:     {:>6}", config.dimension);
    println!("    • Max Depth:         {:>6}", config.max_depth);
    println!("    • Total Nodes:       {:>6} (2^{} - 1)", config.total_nodes(), config.max_depth + 1);
    println!("    • τ Ratio:           {:>6.4} (Cantor ternary)", config.tau_ratio);
    println!("    • Root τ:            {:>6.1} ms", config.base_tau);
    println!("    • Leaf τ:            {:>6.2} ms", config.tau_at_level(config.max_depth));
    println!("    • Creation Time:     {:>6.2} ms\n", creation_time.as_secs_f64() * 1000.0);

    // Show time constants at each level
    println!("  Time Constants by Level (Scale-Depth Theory):");
    println!("  ┌─────────┬────────────┬──────────────────────────────┐");
    println!("  │  Level  │  τ (ms)    │  Cognitive Function          │");
    println!("  ├─────────┼────────────┼──────────────────────────────┤");

    let functions = [
        "Global Integration (Identity)",
        "Narrative, Self-Model",
        "Working Memory, Goals",
        "Attention, Salience",
        "Perceptual Binding",
        "Feature Detection",
        "Low-level Processing",
        "Sensory Input",
    ];

    for level in 0..=config.max_depth {
        let tau = config.tau_at_level(level);
        println!("  │    {}    │  {:>8.2} │  {:<28} │",
                 level, tau, functions[level]);
    }
    println!("  └─────────┴────────────┴──────────────────────────────┘\n");

    // =========================================================================
    // PHASE 2: Dynamics Simulation
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  PHASE 2: Running LTC Dynamics");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Inject external input (simulate sensory stimulus)
    let stimulus = RealHV::random(HDC_DIMENSION, 42);
    println!("  Injecting stimulus into root node...");

    // Run simulation for 100ms
    let sim_start = Instant::now();
    let dt = 0.1; // 0.1ms timesteps
    let sim_duration = 100.0; // 100ms
    let steps = (sim_duration / dt) as usize;

    for i in 0..steps {
        if i == 0 {
            network.step_with_input(dt, &stimulus);
        } else {
            network.step(dt);
        }
    }
    let sim_time = sim_start.elapsed();

    println!("  Simulation complete:");
    println!("    • Duration:    {:>6.1} ms", sim_duration);
    println!("    • Timestep:    {:>6.3} ms", dt);
    println!("    • Steps:       {:>6}", steps);
    println!("    • Wall Time:   {:>6.2} ms", sim_time.as_secs_f64() * 1000.0);
    println!("    • Real-time:   {:>6.1}x\n", sim_duration as f64 / (sim_time.as_secs_f64() * 1000.0));

    // =========================================================================
    // PHASE 3: Hierarchical Φ Measurement
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  PHASE 3: Computing Hierarchical Φ (Integrated Information)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let phi_start = Instant::now();
    let global_phi = network.compute_hierarchical_phi();
    let phi_time = phi_start.elapsed();

    println!("  Per-Level Φ:");
    println!("  ┌─────────┬──────────┬──────────┐");
    println!("  │  Level  │    Φ     │  Weight  │");
    println!("  ├─────────┼──────────┼──────────┤");

    let summary = network.summary();
    for (level, &phi) in summary.level_phi.iter().enumerate() {
        let weight = (config.max_depth - level + 1) as f64;
        println!("  │    {}    │  {:.4}  │   {:.1}    │", level, phi, weight);
    }
    println!("  ├─────────┼──────────┼──────────┤");
    println!("  │  Global │  {:.4}  │   --    │", global_phi);
    println!("  └─────────┴──────────┴──────────┘");
    println!("\n  Computation Time: {:.2} ms\n", phi_time.as_secs_f64() * 1000.0);

    // =========================================================================
    // PHASE 4: Query Demonstration
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  PHASE 4: Querying the Network");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Query for similar nodes to a random pattern
    let query_pattern = RealHV::random(HDC_DIMENSION, 12345);
    let query_start = Instant::now();
    let results = network.query(&query_pattern, 5);
    let query_time = query_start.elapsed();

    println!("  Top 5 Matching Nodes:");
    println!("  ┌─────────┬─────────┬────────────┐");
    println!("  │  Level  │  Index  │ Similarity │");
    println!("  ├─────────┼─────────┼────────────┤");
    for (level, index, similarity) in &results {
        println!("  │    {}    │  {:>4}   │   {:.4}   │", level, index, similarity);
    }
    println!("  └─────────┴─────────┴────────────┘");
    println!("\n  Query Time: {:.3} ms\n", query_time.as_secs_f64() * 1000.0);

    // =========================================================================
    // PHASE 5: Cross-Level Analysis
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  PHASE 5: Cross-Level State Analysis");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Compute similarity between root and each level
    let root_state = network.get_root_state();

    println!("  Root → Level Similarity:");
    println!("  ┌─────────┬──────────────────┬─────────────────┐");
    println!("  │  Level  │  Avg Similarity  │  Node Count     │");
    println!("  ├─────────┼──────────────────┼─────────────────┤");

    for level in 0..=config.max_depth {
        let level_states = network.get_level_states(level);
        let avg_sim: f32 = level_states.iter()
            .map(|s| root_state.similarity(s))
            .sum::<f32>() / level_states.len() as f32;

        println!("  │    {}    │     {:.4}       │      {:>3}        │",
                 level, avg_sim, level_states.len());
    }
    println!("  └─────────┴──────────────────┴─────────────────┘\n");

    // =========================================================================
    // SUMMARY
    // =========================================================================
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                       SUMMARY                                ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║                                                              ║");
    println!("║  Key Innovation: HDC Binding (⊗) in LTC Dynamics             ║");
    println!("║                                                              ║");
    println!("║    dx/dt = (-x + σ(W⊗x + parent⊗bundle(children))) / τ       ║");
    println!("║                                                              ║");
    println!("║  Achievements:                                               ║");
    println!("║    • True HDC-LTC unification (same hypervector space)       ║");
    println!("║    • 7-level meta-cognitive depth                            ║");
    println!("║    • Cantor-like self-similarity (τ ratio = 1/3)             ║");
    println!("║    • Hierarchical Φ measurement                              ║");
    println!("║                                                              ║");
    println!("║  This is the first implementation of:                        ║");
    println!("║    Scale-Depth Theory + HDC + LTC + IIT                      ║");
    println!("║                                                              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Print full summary
    println!("{}", summary);
}
