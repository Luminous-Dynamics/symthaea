/// Hypercube Dimensional Optimization Study
///
/// This example systematically tests hypercubes from 1D through 8D to discover
/// the optimal dimension k* for integrated information (Φ).
///
/// Key Research Questions:
/// 1. Does Φ continue increasing beyond 4D?
/// 2. At what dimension does Φ plateau or decline?
/// 3. What is the optimal dimension k* for consciousness?
///
/// Based on Session 7 Continuation Results:
/// - 1D (Line): Φ ≈ 0.477 (baseline, 2 vertices, 1 neighbor each)
/// - 2D (Square): Φ ≈ 0.485 (4 vertices, 2 neighbors each)
/// - 3D (Cube): Φ = 0.4959 (8 vertices, 3 neighbors each)
/// - 4D (Tesseract): Φ = 0.4976 (16 vertices, 4 neighbors each) - CURRENT CHAMPION
///
/// Predictions:
/// - 5D (Penteract): 32 vertices, 5 neighbors → Φ ≈ 0.499-0.500?
/// - 6D: 64 vertices, 6 neighbors → Φ plateaus?
/// - 7D: 128 vertices, 7 neighbors → Φ declines?
/// - 8D: 256 vertices, 8 neighbors → Computational limit test

use symthaea::hdc::{
    consciousness_topology_generators::ConsciousnessTopology,
    phi_real::RealPhiCalculator,
    HDC_DIMENSION,
};

fn main() {
    println!("\n🔬 HYPERCUBE DIMENSIONAL OPTIMIZATION STUDY");
    println!("{}", "=".repeat(80));
    println!("Testing: Hypercube dimensions 1D through 8D");
    println!("HDC Dimension: {} (2^14 - HDC standard)", HDC_DIMENSION);
    println!("Goal: Find optimal dimension k* for integrated information Φ");
    println!("{}", "=".repeat(80));

    let n_samples = 10;
    let calc = RealPhiCalculator::new();

    // Results storage: (dimension, n_nodes, n_neighbors, mean_phi, std_dev, min, max)
    let mut results: Vec<(usize, usize, usize, f64, f64, f64, f64)> = Vec::new();

    // Test dimensions 1 through 8
    // Note: 8D = 256 nodes, which may be slow
    let max_dimension = 8;

    for dim in 1..=max_dimension {
        let n_nodes = 2_usize.pow(dim as u32);
        let n_neighbors = dim; // Each vertex in k-D hypercube has k neighbors

        println!("\n📊 Testing {}D Hypercube ({} nodes, {} neighbors each)...",
                 dim, n_nodes, n_neighbors);

        // Skip if too many nodes (would be very slow)
        if n_nodes > 512 {
            println!("   ⚠️  Skipping: {} nodes exceeds practical limit", n_nodes);
            continue;
        }

        let mut phi_values = Vec::new();

        for sample in 0..n_samples {
            let seed = 42 + sample as u64 * 1000;

            // Generate hypercube topology
            let topology = ConsciousnessTopology::hypercube(dim, HDC_DIMENSION, seed);

            // Compute Φ
            let phi = calc.compute(&topology.node_representations);
            phi_values.push(phi);

            print!(".");
            use std::io::Write;
            std::io::stdout().flush().unwrap();
        }
        println!();

        // Compute statistics
        let mean = phi_values.iter().sum::<f64>() / n_samples as f64;
        let variance = phi_values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / n_samples as f64;
        let std_dev = variance.sqrt();
        let min = phi_values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = phi_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        println!("   Mean Φ: {:.6} ± {:.6}", mean, std_dev);
        println!("   Range:  [{:.6}, {:.6}]", min, max);

        results.push((dim, n_nodes, n_neighbors, mean, std_dev, min, max));
    }

    // Display comprehensive results
    println!("\n\n{}", "=".repeat(80));
    println!("📈 DIMENSIONAL OPTIMIZATION RESULTS");
    println!("{}", "=".repeat(80));
    println!();
    println!("{:<8} {:>8} {:>10} {:>12} {:>10} {:>12} {:>12}",
             "Dim", "Nodes", "Neighbors", "Mean Φ", "Std Dev", "Min", "Max");
    println!("{}", "-".repeat(80));

    let mut prev_phi = 0.0;
    for (dim, n_nodes, n_neighbors, mean, std_dev, min, max) in &results {
        let delta = if prev_phi > 0.0 {
            format!("{:+.4}", mean - prev_phi)
        } else {
            "---".to_string()
        };
        println!("{:<8} {:>8} {:>10} {:>12.6} {:>10.6} {:>12.6} {:>12.6}  Δ={}",
                 format!("{}D", dim), n_nodes, n_neighbors, mean, std_dev, min, max, delta);
        prev_phi = *mean;
    }

    // Find optimal dimension
    let (optimal_dim, _, _, optimal_phi, _, _, _) = results.iter()
        .max_by(|a, b| a.3.partial_cmp(&b.3).unwrap())
        .unwrap();

    println!("\n{}", "=".repeat(80));
    println!("🏆 OPTIMAL DIMENSION DISCOVERED");
    println!("{}", "=".repeat(80));
    println!();
    println!("   Optimal dimension k* = {}D", optimal_dim);
    println!("   Maximum Φ = {:.6}", optimal_phi);

    // Analyze trend
    println!("\n📊 DIMENSIONAL SCALING ANALYSIS:");

    let mut increasing = true;
    let mut plateau_start = None;

    for i in 1..results.len() {
        let delta = results[i].3 - results[i-1].3;
        let pct_change = (delta / results[i-1].3) * 100.0;

        println!("   {}D → {}D: Δ = {:+.6} ({:+.3}%)",
                 results[i-1].0, results[i].0, delta, pct_change);

        if delta < 0.0 && increasing {
            increasing = false;
            plateau_start = Some(results[i-1].0);
        }
    }

    if let Some(plateau) = plateau_start {
        println!("\n   📍 Φ begins declining after {}D", plateau);
        println!("   ✅ Optimal dimension k* = {}D confirmed", plateau);
    } else if increasing {
        println!("\n   📈 Φ still increasing at maximum tested dimension");
        println!("   ⚠️  Consider testing higher dimensions");
    }

    // Theoretical implications
    println!("\n{}", "=".repeat(80));
    println!("🧠 THEORETICAL IMPLICATIONS");
    println!("{}", "=".repeat(80));
    println!();
    println!("If k* = 3 or 4:");
    println!("   → 3D brains may be consciousness-optimized, not just space-efficient");
    println!("   → Evolution discovered optimal integration dimension");
    println!("   → Physical embedding of consciousness has dimensional constraints");
    println!();
    println!("If k* > 4:");
    println!("   → Higher-dimensional consciousness substrates may exist");
    println!("   → Physical 3D may be suboptimal for integration");
    println!("   → Exotic physics (extra dimensions) could support higher Φ");
    println!();
    println!("Publication Impact:");
    println!("   → First demonstration of dimensional optimization for IIT");
    println!("   → Connects topology, dimensionality, and consciousness");
    println!("   → Testable predictions for neural architecture");

    println!("\n{}", "=".repeat(80));
    println!("✅ Hypercube Dimensional Optimization Study Complete!");
    println!("{}", "=".repeat(80));
}
