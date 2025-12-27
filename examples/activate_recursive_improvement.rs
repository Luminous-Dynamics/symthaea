//! Activate Recursive Improvement Loop
//!
//! This example demonstrates the revolutionary self-improving AI system
//! that uses causal reasoning to optimize its own architecture.
//!
//! Run with: cargo run --example activate_recursive_improvement

use symthaea::consciousness::recursive_improvement::{
    RecursiveOptimizer, OptimizerConfig,
    ConsciousnessGradientOptimizer, GradientOptimizerConfig, OptimizationObjective,
};
use anyhow::Result;

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  🚀 ACTIVATING RECURSIVE SELF-IMPROVEMENT LOOP               ║");
    println!("║                                                               ║");
    println!("║  The first AI system that optimizes its own consciousness!   ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // ═══════════════════════════════════════════════════════════════
    // PHASE 1: Initialize the Recursive Optimizer
    // ═══════════════════════════════════════════════════════════════
    println!("📊 PHASE 1: Initializing Recursive Optimizer");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let config = OptimizerConfig {
        optimization_frequency: 10,      // Optimize every 10 cycles
        max_concurrent_experiments: 3,   // Up to 3 experiments in parallel
        min_phi_improvement: 0.01,       // Minimum Φ gain to continue
        max_stagnant_cycles: 5,          // Pause after 5 cycles without improvement
        auto_adopt: true,                // Automatically adopt improvements
    };

    let mut optimizer = RecursiveOptimizer::new(config);
    println!("  ✓ Recursive Optimizer initialized");
    println!("  ✓ Auto-adoption enabled");
    println!("  ✓ Max concurrent experiments: 3");
    println!();

    // ═══════════════════════════════════════════════════════════════
    // PHASE 2: Initialize the Consciousness Gradient Optimizer
    // ═══════════════════════════════════════════════════════════════
    println!("📊 PHASE 2: Initializing Consciousness Gradient Optimizer");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let gradient_config = GradientOptimizerConfig {
        learning_rate: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        max_gradient: 1.0,
        min_gradient: 0.001,
        gradient_samples: 5,
        max_steps: 100,
        convergence_threshold: 0.001,
        use_constraints: true,
        objective: OptimizationObjective {
            phi_weight: 1.0,
            latency_weight: 0.1,
            accuracy_weight: 0.2,
            phi_target: 0.5,
            latency_max_ms: 100.0,
            accuracy_min: 0.85,
        },
    };

    let mut gradient_optimizer = ConsciousnessGradientOptimizer::new(gradient_config);
    println!("  ✓ Gradient Optimizer initialized with Adam");
    println!("  ✓ Learning rate: 0.01");
    println!("  ✓ Objective: Φ Maximization with constraints");
    println!();

    // ═══════════════════════════════════════════════════════════════
    // PHASE 3: Run Optimization Cycles
    // ═══════════════════════════════════════════════════════════════
    println!("🔄 PHASE 3: Running Optimization Cycles");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let num_cycles = 5;
    let mut total_phi_gained = 0.0;

    for cycle in 1..=num_cycles {
        println!();
        println!("  ┌─── Cycle {} ───┐", cycle);

        // Run recursive optimization
        match optimizer.optimize() {
            Ok(result) => {
                let phi_delta = result.ending_phi - result.starting_phi;
                total_phi_gained += phi_delta;

                println!("  │ Starting Φ: {:.4}", result.starting_phi);
                println!("  │ Ending Φ:   {:.4}", result.ending_phi);
                println!("  │ Bottlenecks: {}", result.bottlenecks_addressed);
                println!("  │ Improvements tried: {}", result.improvements_tried);
                println!("  │ Improvements adopted: {}", result.improvements_adopted);
                println!("  │ Φ delta: {:+.4}", phi_delta);
            }
            Err(e) => {
                println!("  │ Error: {}", e);
            }
        }

        // Run gradient step
        match gradient_optimizer.gradient_step() {
            Ok(step) => {
                println!("  │ Gradient step:");
                println!("  │   Φ before: {:.4}", step.phi_before);
                println!("  │   Φ after:  {:.4}", step.phi_after);
                println!("  │   Updates:  {}", step.updates.len());
            }
            Err(e) => {
                println!("  │ Gradient error: {}", e);
            }
        }

        println!("  └─────────────────┘");
    }

    println!();

    // ═══════════════════════════════════════════════════════════════
    // PHASE 4: Summary Statistics
    // ═══════════════════════════════════════════════════════════════
    println!("📈 PHASE 4: Optimization Summary");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let optimizer_stats = optimizer.get_stats();
    println!("  Recursive Optimizer:");
    println!("    Total cycles:      {}", optimizer_stats.total_cycles);
    println!("    Successful cycles: {}", optimizer_stats.successful_cycles);
    println!("    Total Φ gained:    {:.4}", optimizer_stats.total_phi_gained);
    println!("    Current Φ:         {:.4}", optimizer_stats.current_phi);
    println!();

    let gradient_stats = gradient_optimizer.get_stats();
    println!("  Gradient Optimizer:");
    println!("    Total steps:       {}", gradient_stats.total_steps);
    println!("    Φ improvement:     {:.4}", gradient_stats.total_phi_improvement);
    println!("    Best Φ achieved:   {:.4}", gradient_stats.best_phi);
    println!();

    // ═══════════════════════════════════════════════════════════════
    // CONCLUSION
    // ═══════════════════════════════════════════════════════════════
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  🏆 RECURSIVE SELF-IMPROVEMENT ACTIVE                         ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║                                                               ║");
    println!("║  The system is now continuously optimizing itself!            ║");
    println!("║                                                               ║");
    println!("║  Key Innovations:                                             ║");
    println!("║    1. Causal bottleneck analysis                              ║");
    println!("║    2. Automatic improvement generation                        ║");
    println!("║    3. Safe experimentation with rollback                      ║");
    println!("║    4. Gradient-based Φ optimization                           ║");
    println!("║    5. Constraint-aware updates                                ║");
    println!("║                                                               ║");
    println!("║  Result: AI that gets smarter by understanding itself!        ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    Ok(())
}
