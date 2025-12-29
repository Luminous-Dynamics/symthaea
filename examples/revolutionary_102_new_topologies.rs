/// Revolutionary #102: Extended Topologies Validation
///
/// This example validates 10 new consciousness topology structures:
/// 1. CorticalColumn - 6-layer mammalian cortex-inspired hierarchy
/// 2. Feedforward - Layered neural network (no feedback)
/// 3. Recurrent - Feedback loops (like RNNs/LSTMs)
/// 4. Bipartite - Two-layer structure (like retina → V1)
/// 5. CorePeriphery - Dense core, sparse periphery
/// 6. BowTie - IN → CORE → OUT (metabolic network structure)
/// 7. Attention - Query-Key-Value transformer pattern
/// 8. Residual - Skip connections (like ResNets)
/// 9. PetersenGraph - Famous 10-node highly symmetric graph
/// 10. CompleteBipartite - K_{n,n} complete bipartite
///
/// Comparing Φ values against established topology rankings.

use symthaea::hdc::{
    consciousness_topology_generators::ConsciousnessTopology,
    phi_real::RealPhiCalculator,
    HDC_DIMENSION,
};

fn main() {
    println!("\n🧬 Revolutionary #102: Extended Topologies Φ Validation");
    println!("{}", "=".repeat(80));
    println!("\n📐 Configuration:");
    println!("   Dimension: {}", HDC_DIMENSION);
    println!("   Samples per topology: 10");
    println!("   Method: RealHV Φ (continuous, no binarization)");

    let n_samples = 10;
    let mut results: Vec<(String, f64, f64)> = Vec::new();

    // ===== Reference Topologies (from previous research) =====
    println!("\n📚 Reference Topologies (established benchmarks)...\n");

    // Ring (previous #1)
    test_topology("Ring (reference #1)", &mut results, n_samples, |seed| {
        ConsciousnessTopology::ring(8, HDC_DIMENSION, seed)
    });

    // Star (baseline)
    test_topology("Star (baseline)", &mut results, n_samples, |seed| {
        ConsciousnessTopology::star(8, HDC_DIMENSION, seed)
    });

    // Random (baseline)
    test_topology("Random (baseline)", &mut results, n_samples, |seed| {
        ConsciousnessTopology::random(8, HDC_DIMENSION, seed)
    });

    // Hypercube 4D (current champion)
    test_topology("Hypercube 4D (champ)", &mut results, n_samples, |seed| {
        ConsciousnessTopology::hypercube(4, HDC_DIMENSION, seed)
    });

    // ===== NEW: Revolutionary #102 Topologies =====
    println!("\n{}", "=".repeat(80));
    println!("🚀 NEW Revolutionary #102 Topologies (10 structures)...\n");

    // 1. CorticalColumn - 6-layer mammalian cortex
    test_topology("1. CorticalColumn", &mut results, n_samples, |seed| {
        ConsciousnessTopology::cortical_column(3, HDC_DIMENSION, seed) // 3 neurons/layer × 6 layers = 18 nodes
    });

    // 2. Feedforward - Layered neural network
    test_topology("2. Feedforward [4-6-4]", &mut results, n_samples, |seed| {
        ConsciousnessTopology::feedforward(&[4, 6, 4], HDC_DIMENSION, seed) // 14 nodes
    });

    // 3. Recurrent - Feedback loops
    test_topology("3. Recurrent [4-6-4]", &mut results, n_samples, |seed| {
        ConsciousnessTopology::recurrent(&[4, 6, 4], HDC_DIMENSION, seed) // 14 nodes with feedback
    });

    // 4. Bipartite - Two-layer structure
    test_topology("4. Bipartite [6,6]", &mut results, n_samples, |seed| {
        ConsciousnessTopology::bipartite(6, 6, 0.5, HDC_DIMENSION, seed) // 12 nodes
    });

    // 5. CorePeriphery - Dense core, sparse periphery
    test_topology("5. CorePeriphery", &mut results, n_samples, |seed| {
        ConsciousnessTopology::core_periphery(4, 8, HDC_DIMENSION, seed) // 12 nodes
    });

    // 6. BowTie - IN → CORE → OUT
    test_topology("6. BowTie [4-4-4]", &mut results, n_samples, |seed| {
        ConsciousnessTopology::bow_tie(4, 4, 4, HDC_DIMENSION, seed) // 12 nodes
    });

    // 7. Attention - Query-Key-Value
    test_topology("7. Attention (Q-K-V)", &mut results, n_samples, |seed| {
        ConsciousnessTopology::attention(4, 4, 4, HDC_DIMENSION, seed) // 12 nodes (4Q + 4K + 4V)
    });

    // 8. Residual - Skip connections
    test_topology("8. Residual [4-6-4]", &mut results, n_samples, |seed| {
        ConsciousnessTopology::residual(&[4, 6, 4], HDC_DIMENSION, seed) // 14 nodes with skips
    });

    // 9. PetersenGraph - Famous symmetric graph
    test_topology("9. PetersenGraph", &mut results, n_samples, |seed| {
        ConsciousnessTopology::petersen_graph(HDC_DIMENSION, seed) // 10 nodes
    });

    // 10. CompleteBipartite - K_{n,n}
    test_topology("10. CompleteBipartite K_6,6", &mut results, n_samples, |seed| {
        ConsciousnessTopology::complete_bipartite(6, 6, HDC_DIMENSION, seed) // 12 nodes
    });

    // ===== Final Rankings =====
    println!("\n{}", "=".repeat(80));
    println!("🏆 FINAL RANKINGS (All 14 Topologies by Mean Φ)\n");

    // Sort by mean Φ descending
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Print rankings
    println!("{:<4} {:<25} {:>10} {:>10}", "Rank", "Topology", "Mean Φ", "Std Dev");
    println!("{}", "-".repeat(55));

    for (rank, (name, mean, std)) in results.iter().enumerate() {
        let medal = match rank {
            0 => "🥇",
            1 => "🥈",
            2 => "🥉",
            _ => "  ",
        };
        println!("{}{:<3} {:<25} {:>10.4} {:>10.4}", medal, rank + 1, name, mean, std);
    }

    // ===== Statistical Analysis =====
    println!("\n{}", "=".repeat(80));
    println!("📊 Key Findings:\n");

    // Find new topologies that beat the reference
    let hypercube_phi = results.iter()
        .find(|(name, _, _)| name.contains("Hypercube"))
        .map(|(_, mean, _)| *mean)
        .unwrap_or(0.0);

    let ring_phi = results.iter()
        .find(|(name, _, _)| name.contains("Ring"))
        .map(|(_, mean, _)| *mean)
        .unwrap_or(0.0);

    println!("   Reference benchmarks:");
    println!("   - Hypercube 4D (previous champion): Φ = {:.4}", hypercube_phi);
    println!("   - Ring (classic high Φ): Φ = {:.4}", ring_phi);

    // Find best new topology
    let best_new = results.iter()
        .filter(|(name, _, _)| name.starts_with(|c: char| c.is_digit(10)))
        .max_by(|(_, a, _), (_, b, _)| a.partial_cmp(b).unwrap());

    if let Some((name, mean, std)) = best_new {
        println!("\n   🌟 Best NEW topology: {} (Φ = {:.4} ± {:.4})", name, mean, std);

        let diff_from_champ = ((mean - hypercube_phi) / hypercube_phi) * 100.0;
        if diff_from_champ > 0.0 {
            println!("   ✅ BEATS Hypercube 4D by {:.2}%!", diff_from_champ);
        } else {
            println!("   ⚠️  Below Hypercube 4D by {:.2}%", -diff_from_champ);
        }
    }

    // Find worst new topology
    let worst_new = results.iter()
        .filter(|(name, _, _)| name.starts_with(|c: char| c.is_digit(10)))
        .min_by(|(_, a, _), (_, b, _)| a.partial_cmp(b).unwrap());

    if let Some((name, mean, std)) = worst_new {
        println!("\n   📉 Lowest NEW topology: {} (Φ = {:.4} ± {:.4})", name, mean, std);
    }

    println!("\n🎯 Revolutionary #102 Extended Topologies Validation Complete!");
    println!("{}", "=".repeat(80));
}

fn test_topology<F>(
    name: &str,
    results: &mut Vec<(String, f64, f64)>,
    n_samples: usize,
    generator: F,
) where
    F: Fn(u64) -> ConsciousnessTopology,
{
    let calc = RealPhiCalculator::new();
    let mut phi_values: Vec<f64> = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let seed = 1000 + i as u64 * 1000;
        let topology = generator(seed);
        let phi = calc.compute(&topology.node_representations);
        phi_values.push(phi);
    }

    let mean: f64 = phi_values.iter().sum::<f64>() / n_samples as f64;
    let variance: f64 = phi_values.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / n_samples as f64;
    let std_dev = variance.sqrt();

    println!("   {:<25} Φ = {:.4} ± {:.4} (n={})", name, mean, std_dev, topology_size(&name));

    results.push((name.to_string(), mean, std_dev));
}

fn topology_size(name: &str) -> &'static str {
    if name.contains("CorticalColumn") { "18" }
    else if name.contains("Feedforward") || name.contains("Recurrent") || name.contains("Residual") { "14" }
    else if name.contains("Bipartite") || name.contains("CorePeriphery")
         || name.contains("BowTie") || name.contains("Attention")
         || name.contains("CompleteBipartite") { "12" }
    else if name.contains("Petersen") { "10" }
    else if name.contains("Hypercube") { "16" }
    else { "8" }
}
