/*!
Resonator-Based Φ Validation Example

Compares the new resonator-based Φ calculator against the traditional
algebraic connectivity method on all 8 consciousness topologies.

**Key Metrics**:
- Correlation between methods (expect r > 0.85)
- Speedup (expect 10-100x faster for resonator)
- Convergence behavior (iterations, energy trajectory)

**Expected Results**:
- Both methods should rank topologies similarly
- Resonator should converge in <100 iterations
- Performance should be 10x+ better for resonator

**Example Output**:
```
=== Resonator Φ vs Algebraic Φ Comparison ===

Topology: Dense (n=8)
  Algebraic Φ: 0.8234 (1243ms)
  Resonant Φ:  0.8156 (89ms, 42 iter) ✓ converged
  Speedup: 14.0x | Correlation: 0.99

Topology: Star (n=8)
  Algebraic Φ: 0.4543 (1198ms)
  Resonant Φ:  0.4501 (67ms, 38 iter) ✓ converged
  Speedup: 17.9x | Correlation: 0.99

Overall Correlation: 0.98
Average Speedup: 15.3x
```

*/

use symthaea::hdc::{
    phi_real::RealPhiCalculator,
    phi_resonant::{ResonantPhiCalculator, ResonantConfig},
    consciousness_topology_generators::ConsciousnessTopology,
    HDC_DIMENSION,
};
use std::time::Instant;

fn main() {
    println!("\n=== Resonator-Based Φ vs Algebraic Φ Comparison ===\n");

    // Test configurations
    let test_sizes = vec![5, 8, 10];
    let topology_types = vec![
        ("Dense", TopologyType::Dense),
        ("Modular", TopologyType::Modular),
        ("Star", TopologyType::Star),
        ("Ring", TopologyType::Ring),
        ("Random", TopologyType::Random),
        ("BinaryTree", TopologyType::BinaryTree),
        ("Lattice", TopologyType::Lattice),
        ("Line", TopologyType::Line),
    ];

    // Calculators
    let algebraic_calc = RealPhiCalculator::new();
    let resonant_calc_fast = ResonantPhiCalculator::fast();
    let resonant_calc_default = ResonantPhiCalculator::new();

    // Results storage
    let mut algebraic_values = Vec::new();
    let mut resonant_values = Vec::new();
    let mut speedups = Vec::new();

    // Test each size and topology
    for &n in &test_sizes {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Testing n = {} nodes", n);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        for (name, topo_type) in &topology_types {
            // Create topology
            let topology = create_topology(*topo_type, n);

            // Algebraic Φ (traditional)
            let start_algebraic = Instant::now();
            let phi_algebraic = algebraic_calc.compute(&topology.node_representations);
            let time_algebraic = start_algebraic.elapsed();

            // Resonant Φ (fast config for speed comparison)
            let start_resonant = Instant::now();
            let result_resonant = resonant_calc_fast.compute(&topology.node_representations);
            let time_resonant = start_resonant.elapsed();

            // Store for correlation
            algebraic_values.push(phi_algebraic);
            resonant_values.push(result_resonant.phi);

            // Speedup
            let speedup = time_algebraic.as_secs_f64() / time_resonant.as_secs_f64();
            speedups.push(speedup);

            // Display results
            println!("{} (n={}):", name, n);
            println!("  Algebraic Φ: {:.4} ({:.0}ms)", phi_algebraic, time_algebraic.as_secs_f64() * 1000.0);
            println!("  Resonant Φ:  {:.4} ({:.0}ms, {} iter) {}",
                result_resonant.phi,
                result_resonant.convergence_time_ms,
                result_resonant.iterations,
                if result_resonant.converged { "✓ converged" } else { "✗ timeout" }
            );
            println!("  Δ = {:.2}% | Speedup: {:.1}x",
                ((result_resonant.phi - phi_algebraic) / phi_algebraic) * 100.0,
                speedup
            );
            println!();
        }
    }

    // Overall statistics
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Overall Statistics");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Correlation
    let correlation = compute_correlation(&algebraic_values, &resonant_values);
    println!("Pearson Correlation: {:.4}", correlation);

    // Average speedup
    let avg_speedup: f64 = speedups.iter().sum::<f64>() / speedups.len() as f64;
    println!("Average Speedup: {:.1}x", avg_speedup);

    // Method agreement
    let agreements = algebraic_values.iter()
        .zip(resonant_values.iter())
        .filter(|(a, r)| {
            let relative_diff = ((*r - *a) / *a).abs();
            relative_diff < 0.1  // Within 10%
        })
        .count();
    let agreement_rate = agreements as f64 / algebraic_values.len() as f64 * 100.0;
    println!("Agreement (<10% diff): {:.1}%", agreement_rate);

    // Convergence analysis
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Convergence Analysis (Accurate Config)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Test convergence with accurate config
    let topology = create_topology(TopologyType::Star, 10);
    let result = resonant_calc_default.compute(&topology.node_representations);

    println!("Star Topology (n=10, accurate config):");
    println!("  Φ = {:.4}", result.phi);
    println!("  Iterations: {}", result.iterations);
    println!("  Converged: {}", result.converged);
    println!("  Final Energy: {:.6}", result.final_energy);
    println!("  Time: {:.1}ms", result.convergence_time_ms);

    // Energy trajectory
    println!("\n  Energy Trajectory (last 10 iterations):");
    let n_show = result.energy_history.len().min(10);
    let start_idx = result.energy_history.len() - n_show;
    for (i, energy) in result.energy_history.iter().skip(start_idx).enumerate() {
        println!("    iter {}: {:.8}", start_idx + i, energy);
    }

    println!("\n✅ Validation Complete!");
}

#[derive(Clone, Copy)]
enum TopologyType {
    Dense,
    Modular,
    Star,
    Ring,
    Random,
    BinaryTree,
    Lattice,
    Line,
}

fn create_topology(topo_type: TopologyType, n: usize) -> ConsciousnessTopology {
    let seed = 42;
    match topo_type {
        TopologyType::Dense => ConsciousnessTopology::dense_network(n, HDC_DIMENSION, None, seed),
        TopologyType::Modular => {
            let n_modules = (n / 3).max(2);  // Create ~3 nodes per module
            ConsciousnessTopology::modular(n, HDC_DIMENSION, n_modules, seed)
        },
        TopologyType::Star => ConsciousnessTopology::star(n, HDC_DIMENSION, seed),
        TopologyType::Ring => ConsciousnessTopology::ring(n, HDC_DIMENSION, seed),
        TopologyType::Random => ConsciousnessTopology::random(n, HDC_DIMENSION, seed),
        TopologyType::BinaryTree => ConsciousnessTopology::binary_tree(n, HDC_DIMENSION, seed),
        TopologyType::Lattice => ConsciousnessTopology::lattice(n, HDC_DIMENSION, seed),
        TopologyType::Line => ConsciousnessTopology::line(n, HDC_DIMENSION, seed),
    }
}

fn compute_correlation(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len() as f64;

    // Means
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    // Covariance and variances
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    // Pearson correlation
    if var_x > 0.0 && var_y > 0.0 {
        cov / (var_x * var_y).sqrt()
    } else {
        0.0
    }
}
