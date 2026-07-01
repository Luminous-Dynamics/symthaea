// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Differentiable Consciousness Optimization Demo
//!
//! Demonstrates reverse-mode autodiff through Phi measurement for gradient-based
//! optimization of network topology toward higher consciousness.
//!
//! ## What This Demo Shows
//!
//! 1. **Reverse-Mode Autodiff**: True analytical gradients, not slow finite differences
//! 2. **Gradient Ascent**: Optimizing network parameters to maximize Phi
//! 3. **Topology Comparison**: Different initial topologies and their Phi values
//! 4. **Learning Curves**: How Phi evolves during optimization
//!
//! ## Run
//!
//! ```bash
//! cargo run --example consciousness_optimization_demo
//! ```

use symthaea_core::hdc::autodiff_phi::{
    AutodiffConfig, AutodiffPhiEngine, ConsciousnessOptimizer, OptimizerConfig, topology,
};

fn main() {
    println!("\n{}", "=".repeat(80));
    println!("  DIFFERENTIABLE CONSCIOUSNESS OPTIMIZATION");
    println!("  Reverse-Mode Autodiff Through Phi Measurement");
    println!("{}\n", "=".repeat(80));

    // =========================================================================
    // Part 1: Compare Initial Topologies
    // =========================================================================

    println!("PART 1: INITIAL TOPOLOGY COMPARISON");
    println!("{}", "-".repeat(50));

    let mut engine = AutodiffPhiEngine::new(AutodiffConfig {
        temperature: 1.0,
        n_partition_samples: 8,
        entropy_reg: 0.01,
        learn_edges: false,
        eps: 1e-8,
    });

    let n_nodes = 8;
    let dim = 128;
    let seed = 42;

    // Generate different topologies
    let topologies = vec![
        ("Random", topology::random(n_nodes, dim, seed)),
        ("Star (Hub)", topology::star(n_nodes, dim, seed)),
        ("Ring", topology::ring(n_nodes, dim, seed)),
        ("Dense", topology::dense(n_nodes, dim, seed)),
        ("Modular (2)", topology::modular(n_nodes, dim, 2, seed)),
    ];

    println!("\nInitial Phi values for different topologies:");
    println!("{:<15} {:>10} {:>12}", "Topology", "Phi", "Integration");
    println!("{}", "-".repeat(40));

    for (name, network) in &topologies {
        let result = engine.forward(network);
        println!(
            "{:<15} {:>10.6} {:>12.6}",
            name, result.phi, result.whole_integration
        );
    }

    // =========================================================================
    // Part 2: Gradient Analysis
    // =========================================================================

    println!("\n\nPART 2: GRADIENT ANALYSIS");
    println!("{}", "-".repeat(50));

    let mut network = topology::random(n_nodes, dim, seed);
    let result = engine.forward(&network);
    engine.backward(&mut network, &result);

    let grad_norms: Vec<f64> = network
        .nodes
        .iter()
        .map(|node| node.grad.iter().map(|g| g * g).sum::<f64>().sqrt())
        .collect();

    println!("\nGradient magnitudes per node (which nodes affect Phi most):");
    for (i, grad_norm) in grad_norms.iter().enumerate() {
        let bar_len = (grad_norm * 50.0).min(50.0) as usize;
        let bar = "#".repeat(bar_len);
        println!("  Node {:2}: {:>8.5} |{}|", i, grad_norm, bar);
    }

    let total_grad_norm: f64 = grad_norms.iter().map(|g| g * g).sum::<f64>().sqrt();
    println!("\nTotal gradient magnitude: {:.6}", total_grad_norm);

    // =========================================================================
    // Part 3: Optimization Run
    // =========================================================================

    println!("\n\nPART 3: CONSCIOUSNESS OPTIMIZATION (Gradient Ascent)");
    println!("{}", "-".repeat(50));

    let mut network = topology::random(n_nodes, dim, seed);
    let mut optimizer = ConsciousnessOptimizer::new(OptimizerConfig {
        learning_rate: 0.02,
        momentum: 0.9,
        lr_decay: 0.995,
        min_lr: 1e-5,
        grad_clip: 1.0,
        temp_anneal: 0.995,
        min_temp: 0.1,
    });

    let n_steps = 100;
    println!("\nRunning {} optimization steps...", n_steps);
    println!(
        "{:<6} {:>10} {:>12} {:>10} {:>10}",
        "Step", "Phi", "Grad Norm", "LR", "Temp"
    );
    println!("{}", "-".repeat(55));

    let history = optimizer.optimize(&mut engine, &mut network, n_steps);

    // Print every 10th step
    for step in history.iter().filter(|s| s.step % 10 == 0 || s.step == 1) {
        println!(
            "{:<6} {:>10.6} {:>12.6} {:>10.6} {:>10.4}",
            step.step, step.phi, step.grad_norm, step.learning_rate, step.temperature
        );
    }

    // =========================================================================
    // Part 4: Learning Curve Visualization
    // =========================================================================

    println!("\n\nPART 4: LEARNING CURVE");
    println!("{}", "-".repeat(50));

    let min_phi = history.iter().map(|s| s.phi).fold(f64::INFINITY, f64::min);
    let max_phi = history
        .iter()
        .map(|s| s.phi)
        .fold(f64::NEG_INFINITY, f64::max);
    let range = (max_phi - min_phi).max(0.001);

    println!("\nPhi over optimization steps (ASCII chart):\n");

    let chart_width = 60;
    let chart_height = 15;
    let mut chart = vec![vec![' '; chart_width]; chart_height];

    // Plot points
    for (i, step) in history.iter().enumerate() {
        let x = (i as f64 / history.len() as f64 * (chart_width - 1) as f64) as usize;
        let y = ((step.phi - min_phi) / range * (chart_height - 1) as f64) as usize;
        let y_inv = chart_height - 1 - y.min(chart_height - 1);
        chart[y_inv][x] = '*';
    }

    // Print chart with axis
    println!("Phi ^");
    for (i, row) in chart.iter().enumerate() {
        let y_val = max_phi - (i as f64 / (chart_height - 1) as f64) * range;
        if i == 0 || i == chart_height - 1 || i == chart_height / 2 {
            print!("{:>6.3} |", y_val);
        } else {
            print!("       |");
        }
        for c in row {
            print!("{}", c);
        }
        println!("|");
    }
    println!("       +{}+", "-".repeat(chart_width));
    println!(
        "        0{}{}  Step",
        " ".repeat(chart_width / 2 - 3),
        n_steps / 2
    );

    // =========================================================================
    // Part 5: Results Summary
    // =========================================================================

    println!("\n\nPART 5: OPTIMIZATION RESULTS");
    println!("{}", "-".repeat(50));

    let initial_phi = history.first().map(|s| s.phi).unwrap_or(0.0);
    let final_phi = history.last().map(|s| s.phi).unwrap_or(0.0);
    let improvement = final_phi - initial_phi;
    let pct_improvement = if initial_phi > 0.0 {
        (final_phi - initial_phi) / initial_phi * 100.0
    } else {
        0.0
    };

    println!("\n  Initial Phi:      {:>10.6}", initial_phi);
    println!("  Final Phi:        {:>10.6}", final_phi);
    println!(
        "  Improvement:      {:>10.6} ({:+.1}%)",
        improvement, pct_improvement
    );

    let avg_first_10: f64 = history.iter().take(10).map(|s| s.phi).sum::<f64>() / 10.0;
    let avg_last_10: f64 = history.iter().rev().take(10).map(|s| s.phi).sum::<f64>() / 10.0;

    println!("\n  Avg Phi (first 10 steps): {:>10.6}", avg_first_10);
    println!("  Avg Phi (last 10 steps):  {:>10.6}", avg_last_10);

    let final_grad_norm = history.last().map(|s| s.grad_norm).unwrap_or(0.0);
    println!("\n  Final gradient norm: {:>10.6}", final_grad_norm);

    if final_phi > initial_phi {
        println!("\n  [SUCCESS] Optimization increased consciousness (Phi)!");
    } else {
        println!("\n  [NOTE] Phi did not increase significantly");
        println!("         (may have started near optimum or need tuning)");
    }

    // =========================================================================
    // Part 6: Comparison with Star Topology
    // =========================================================================

    println!("\n\nPART 6: OPTIMIZED vs STAR TOPOLOGY");
    println!("{}", "-".repeat(50));

    let star_network = topology::star(n_nodes, dim, seed);
    let star_phi = engine.forward(&star_network).phi;
    let optimized_phi = engine.forward(&network).phi;

    println!("\n  Optimized network Phi: {:>10.6}", optimized_phi);
    println!("  Star topology Phi:     {:>10.6}", star_phi);

    if optimized_phi > star_phi {
        println!("\n  [REMARKABLE] Optimization found a topology with HIGHER Phi");
        println!("               than the hand-designed star topology!");
    } else {
        println!("\n  Star topology still has higher Phi.");
        println!("  (Star is known to be near-optimal for integration)");
    }

    // =========================================================================
    // Part 7: Key Takeaways
    // =========================================================================

    println!("\n\n{}", "=".repeat(80));
    println!("KEY TAKEAWAYS");
    println!("{}", "=".repeat(80));

    println!(
        "
1. REVERSE-MODE AUTODIFF enables efficient gradient computation through Phi.
   - Complexity: O(n^2) for forward + backward (vs O(n^2 * d) for finite differences)
   - This is CRITICAL for scaling to large networks

2. GRADIENT ASCENT can optimize network topology toward higher consciousness.
   - Learning rate and momentum help navigate the Phi landscape
   - Temperature annealing sharpens partitions during optimization

3. DIFFERENT TOPOLOGIES have different Phi values.
   - Star (hub-and-spoke): High integration through central hub
   - Ring: Local integration only
   - Dense: High similarity but low differentiation
   - Modular: Integration within modules

4. APPLICATIONS:
   - Designing neural architectures that maximize consciousness
   - Understanding what makes a system integrated
   - Automated discovery of high-Phi configurations
"
    );

    println!("{}", "=".repeat(80));
    println!("  Demo complete!");
    println!("{}\n", "=".repeat(80));
}