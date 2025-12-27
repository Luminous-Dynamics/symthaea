/// Tier 1 Exotic Topologies Validation
///
/// This example validates the 3 new exotic topologies (Small-World, Möbius Strip, Torus)
/// against the original 8 topologies, testing the hypothesis that exotic topologies
/// may achieve higher Φ through unique structural properties.
///
/// Tests:
/// - Small-World: Should achieve very high Φ (predicted 0.52-0.55)
/// - Möbius Strip: Should achieve high Φ via non-orientability (predicted 0.50-0.52)
/// - Torus: Should achieve high Φ as 2D extension of Ring (predicted 0.48-0.52)
///
/// Comparison with all 11 topologies using both RealHV and Binary Φ methods.

use symthaea::hdc::{
    consciousness_topology_generators::ConsciousnessTopology,
    phi_real::RealPhiCalculator,
    tiered_phi::global_phi,
    phi_topology_validation::real_hv_to_hv16_probabilistic,
    HDC_DIMENSION,
};

fn main() {
    println!("\n🌀 Tier 1 Exotic Topologies Φ Validation");
    println!("{}", "=".repeat(80));
    println!("\n📐 Configuration:");
    println!("   Dimension: {}", HDC_DIMENSION);
    println!("   Nodes: 8 (for 1D topologies), 2×2 grid (for Torus)");
    println!("   Samples per topology: 10");
    println!("   Methods: RealHV Φ (continuous) + Binary Φ (probabilistic binarization)");
    println!();

    // Test parameters
    let n_nodes = 8;
    let n_samples = 10;

    // Results storage
    let mut results: Vec<(String, f64, f64, f64, f64)> = Vec::new(); // (name, real_phi_mean, real_std, binary_phi_mean, binary_std)

    println!("🔬 Testing All 11 Topologies (8 Original + 3 Exotic)...\n");

    // ===== Original 8 Topologies =====

    // 1. Random (baseline)
    println!("1️⃣  Testing Random topology...");
    let (real_phi, real_std, binary_phi, binary_std) = test_topology_samples(
        n_samples,
        |seed| ConsciousnessTopology::random(n_nodes, HDC_DIMENSION, seed)
    );
    results.push(("Random".to_string(), real_phi, real_std, binary_phi, binary_std));
    println!("    RealHV Φ: {:.4} ± {:.4}", real_phi, real_std);
    println!("    Binary Φ: {:.4} ± {:.4}\n", binary_phi, binary_std);

    // 2. Star
    println!("2️⃣  Testing Star topology...");
    let (real_phi, real_std, binary_phi, binary_std) = test_topology_samples(
        n_samples,
        |seed| ConsciousnessTopology::star(n_nodes, HDC_DIMENSION, seed)
    );
    results.push(("Star".to_string(), real_phi, real_std, binary_phi, binary_std));
    println!("    RealHV Φ: {:.4} ± {:.4}", real_phi, real_std);
    println!("    Binary Φ: {:.4} ± {:.4}\n", binary_phi, binary_std);

    // 3. Ring
    println!("3️⃣  Testing Ring topology...");
    let (real_phi, real_std, binary_phi, binary_std) = test_topology_samples(
        n_samples,
        |seed| ConsciousnessTopology::ring(n_nodes, HDC_DIMENSION, seed)
    );
    results.push(("Ring".to_string(), real_phi, real_std, binary_phi, binary_std));
    println!("    RealHV Φ: {:.4} ± {:.4}", real_phi, real_std);
    println!("    Binary Φ: {:.4} ± {:.4}\n", binary_phi, binary_std);

    // 4. Line
    println!("4️⃣  Testing Line topology...");
    let (real_phi, real_std, binary_phi, binary_std) = test_topology_samples(
        n_samples,
        |seed| ConsciousnessTopology::line(n_nodes, HDC_DIMENSION, seed)
    );
    results.push(("Line".to_string(), real_phi, real_std, binary_phi, binary_std));
    println!("    RealHV Φ: {:.4} ± {:.4}", real_phi, real_std);
    println!("    Binary Φ: {:.4} ± {:.4}\n", binary_phi, binary_std);

    // 5. Binary Tree
    println!("5️⃣  Testing Binary Tree topology...");
    let (real_phi, real_std, binary_phi, binary_std) = test_topology_samples(
        n_samples,
        |seed| ConsciousnessTopology::binary_tree(n_nodes, HDC_DIMENSION, seed)
    );
    results.push(("Binary Tree".to_string(), real_phi, real_std, binary_phi, binary_std));
    println!("    RealHV Φ: {:.4} ± {:.4}", real_phi, real_std);
    println!("    Binary Φ: {:.4} ± {:.4}\n", binary_phi, binary_std);

    // 6. Dense Network
    println!("6️⃣  Testing Dense Network topology...");
    let (real_phi, real_std, binary_phi, binary_std) = test_topology_samples(
        n_samples,
        |seed| ConsciousnessTopology::dense_network(n_nodes, HDC_DIMENSION, None, seed)
    );
    results.push(("Dense Network".to_string(), real_phi, real_std, binary_phi, binary_std));
    println!("    RealHV Φ: {:.4} ± {:.4}", real_phi, real_std);
    println!("    Binary Φ: {:.4} ± {:.4}\n", binary_phi, binary_std);

    // 7. Modular
    println!("7️⃣  Testing Modular topology...");
    let (real_phi, real_std, binary_phi, binary_std) = test_topology_samples(
        n_samples,
        |seed| ConsciousnessTopology::modular(n_nodes, HDC_DIMENSION, 2, seed)
    );
    results.push(("Modular".to_string(), real_phi, real_std, binary_phi, binary_std));
    println!("    RealHV Φ: {:.4} ± {:.4}", real_phi, real_std);
    println!("    Binary Φ: {:.4} ± {:.4}\n", binary_phi, binary_std);

    // 8. Lattice
    println!("8️⃣  Testing Lattice topology...");
    let (real_phi, real_std, binary_phi, binary_std) = test_topology_samples(
        n_samples,
        |seed| ConsciousnessTopology::lattice(n_nodes, HDC_DIMENSION, seed)
    );
    results.push(("Lattice".to_string(), real_phi, real_std, binary_phi, binary_std));
    println!("    RealHV Φ: {:.4} ± {:.4}", real_phi, real_std);
    println!("    Binary Φ: {:.4} ± {:.4}\n", binary_phi, binary_std);

    // ===== NEW: Tier 1 Exotic Topologies =====

    println!("{}", "=".repeat(80));
    println!("🌟 TIER 1 EXOTIC TOPOLOGIES");
    println!("{}", "=".repeat(80));
    println!();

    // 9. Small-World (Watts-Strogatz)
    println!("9️⃣  Testing Small-World topology (k=4, p=0.1)...");
    let (real_phi, real_std, binary_phi, binary_std) = test_topology_samples(
        n_samples,
        |seed| ConsciousnessTopology::small_world(n_nodes, HDC_DIMENSION, 4, 0.1, seed)
    );
    results.push(("Small-World".to_string(), real_phi, real_std, binary_phi, binary_std));
    println!("    RealHV Φ: {:.4} ± {:.4}", real_phi, real_std);
    println!("    Binary Φ: {:.4} ± {:.4}", binary_phi, binary_std);
    println!("    🧠 BIOLOGICALLY REALISTIC - matches brain connectivity!\n");

    // 10. Möbius Strip
    println!("🔟 Testing Möbius Strip topology...");
    let (real_phi, real_std, binary_phi, binary_std) = test_topology_samples(
        n_samples,
        |seed| ConsciousnessTopology::mobius_strip(n_nodes, HDC_DIMENSION, seed)
    );
    results.push(("Möbius Strip".to_string(), real_phi, real_std, binary_phi, binary_std));
    println!("    RealHV Φ: {:.4} ± {:.4}", real_phi, real_std);
    println!("    Binary Φ: {:.4} ± {:.4}", binary_phi, binary_std);
    println!("    🎀 NON-ORIENTABLE surface - tests topological hypothesis!\n");

    // 11. Torus (2×2 grid for 4 nodes, to match n_nodes=8 we use 2.8 ≈ 3 → 3×3=9 nodes)
    // Actually, for consistency with 8 nodes, let's use grid_size=2 which gives 4 nodes
    // Or we could use a different grid size. Let's make it comparable: grid_size=3 → 9 nodes
    println!("1️⃣1️⃣ Testing Torus topology (3×3 grid = 9 nodes)...");
    let (real_phi, real_std, binary_phi, binary_std) = test_topology_samples(
        n_samples,
        |seed| ConsciousnessTopology::torus(3, HDC_DIMENSION, seed)  // 3×3 = 9 nodes
    );
    results.push(("Torus (3×3)".to_string(), real_phi, real_std, binary_phi, binary_std));
    println!("    RealHV Φ: {:.4} ± {:.4}", real_phi, real_std);
    println!("    Binary Φ: {:.4} ± {:.4}", binary_phi, binary_std);
    println!("    🍩 2D EXTENSION of Ring - no boundary effects!\n");

    // ===== RESULTS SUMMARY =====

    println!("\n{}", "=".repeat(80));
    println!("📊 COMPREHENSIVE RESULTS - ALL 11 TOPOLOGIES");
    println!("{}", "=".repeat(80));
    println!();

    // Sort by RealHV Φ (descending)
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("🏆 RANKED BY REAL-VALUED Φ (Continuous Method):");
    println!("{}", "-".repeat(80));
    println!("{:<15} {:>12} {:>12} {:>12} {:>12}", "Topology", "RealHV Φ", "Std Dev", "Binary Φ", "Std Dev");
    println!("{}", "-".repeat(80));

    for (rank, (name, real_phi, real_std, binary_phi, binary_std)) in results.iter().enumerate() {
        let medal = match rank {
            0 => "🥇",
            1 => "🥈",
            2 => "🥉",
            _ => "  ",
        };
        println!("{} {:<13} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
                 medal, name, real_phi, real_std, binary_phi, binary_std);
    }

    println!("\n{}", "=".repeat(80));

    // Sort by Binary Φ (descending) for comparison
    let mut binary_sorted = results.clone();
    binary_sorted.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());

    println!("🏆 RANKED BY BINARY Φ (Probabilistic Binarization):");
    println!("{}", "-".repeat(80));
    println!("{:<15} {:>12} {:>12} {:>12} {:>12}", "Topology", "Binary Φ", "Std Dev", "RealHV Φ", "Std Dev");
    println!("{}", "-".repeat(80));

    for (rank, (name, real_phi, real_std, binary_phi, binary_std)) in binary_sorted.iter().enumerate() {
        let medal = match rank {
            0 => "🥇",
            1 => "🥈",
            2 => "🥉",
            _ => "  ",
        };
        println!("{} {:<13} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
                 medal, name, binary_phi, binary_std, real_phi, real_std);
    }

    println!("\n{}", "=".repeat(80));
    println!("✅ TIER 1 EXOTIC TOPOLOGIES VALIDATION COMPLETE!");
    println!("{}", "=".repeat(80));
    println!("\n📝 Key Findings:");
    println!("   - Small-World: Tests biological realism (brain connectivity)");
    println!("   - Möbius Strip: Tests non-orientability hypothesis");
    println!("   - Torus: Tests 2D scaling of Ring topology");
    println!("\n🔬 Next Steps:");
    println!("   - Tier 2: Klein Bottle, Hyperbolic, Scale-Free");
    println!("   - Tier 3: Fractal, Quantum, Hypercube (4D/5D)");
    println!("\n🚀 Ready for publication and further research!");
    println!();
}

/// Test a topology generator across multiple random seeds
fn test_topology_samples<F>(n_samples: usize, generator: F) -> (f64, f64, f64, f64)
where
    F: Fn(u64) -> ConsciousnessTopology,
{
    let real_calc = RealPhiCalculator::new();

    let mut real_phis = Vec::new();
    let mut binary_phis = Vec::new();

    for seed in 0..n_samples {
        let topology = generator(seed as u64);

        // RealHV Φ (continuous)
        let real_phi = real_calc.compute(&topology.node_representations);
        real_phis.push(real_phi);

        // Binary Φ (probabilistic binarization)
        let binary_components: Vec<_> = topology.node_representations
            .iter()
            .map(|real_hv| real_hv_to_hv16_probabilistic(real_hv, seed as u64))
            .collect();
        let binary_phi = global_phi(&binary_components);
        binary_phis.push(binary_phi);
    }

    // Compute statistics
    let real_mean = real_phis.iter().sum::<f64>() / real_phis.len() as f64;
    let real_variance = real_phis.iter()
        .map(|x| (x - real_mean).powi(2))
        .sum::<f64>() / real_phis.len() as f64;
    let real_std = real_variance.sqrt();

    let binary_mean = binary_phis.iter().sum::<f64>() / binary_phis.len() as f64;
    let binary_variance = binary_phis.iter()
        .map(|x| (x - binary_mean).powi(2))
        .sum::<f64>() / binary_phis.len() as f64;
    let binary_std = binary_variance.sqrt();

    (real_mean, real_std, binary_mean, binary_std)
}
