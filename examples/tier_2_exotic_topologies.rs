/// Tier 2 Exotic Topologies Validation
///
/// This example validates the 3 Tier 2 exotic topologies (Klein Bottle, Hyperbolic, Scale-Free)
/// testing advanced hypotheses about consciousness topology relationships.
///
/// Based on Tier 1 results:
/// - Möbius Strip (1D non-orientable) FAILED catastrophically (Φ = 0.3729)
/// - Prediction: Klein Bottle (2D non-orientable) will also fail
///
/// Tests:
/// - Klein Bottle: 2D non-orientable surface (predicted: low Φ like Möbius)
/// - Hyperbolic: Negative curvature (predicted: medium Φ ≈ 0.46-0.50)
/// - Scale-Free: Power-law hubs (predicted: medium-high Φ ≈ 0.44-0.48)
///
/// Comparison with all 14 topologies (8 original + 3 Tier 1 + 3 Tier 2).

use symthaea::hdc::{
    consciousness_topology_generators::ConsciousnessTopology,
    phi_real::RealPhiCalculator,
    tiered_phi::global_phi,
    phi_topology_validation::real_hv_to_hv16_probabilistic,
    HDC_DIMENSION,
};

fn main() {
    println!("\n🌀 Tier 2 Exotic Topologies Φ Validation");
    println!("{}", "=".repeat(80));
    println!("\n📐 Configuration:");
    println!("   Dimension: {}", HDC_DIMENSION);
    println!("   Nodes: 8 (for most), 3×3 grid (for 2D surfaces)");
    println!("   Samples per topology: 10");
    println!("   Methods: RealHV Φ (continuous) + Binary Φ (probabilistic binarization)");
    println!("\n📊 Testing All 14 Topologies (8 Original + 3 Tier 1 + 3 Tier 2)...\n");

    let n_nodes = 8;
    let n_samples = 10;

    let mut results: Vec<(String, f64, f64, f64, f64)> = Vec::new();

    // ===== Original 8 Topologies (Quick Reference) =====
    println!("📚 Original 8 Topologies (Quick Test)...\n");

    test_and_record("Random", &mut results, n_samples, |seed| {
        ConsciousnessTopology::random(n_nodes, HDC_DIMENSION, seed)
    });

    test_and_record("Star", &mut results, n_samples, |seed| {
        ConsciousnessTopology::star(n_nodes, HDC_DIMENSION, seed)
    });

    test_and_record("Ring", &mut results, n_samples, |seed| {
        ConsciousnessTopology::ring(n_nodes, HDC_DIMENSION, seed)
    });

    test_and_record("Line", &mut results, n_samples, |seed| {
        ConsciousnessTopology::line(n_nodes, HDC_DIMENSION, seed)
    });

    test_and_record("Binary Tree", &mut results, n_samples, |seed| {
        ConsciousnessTopology::binary_tree(n_nodes, HDC_DIMENSION, seed)
    });

    test_and_record("Dense Network", &mut results, n_samples, |seed| {
        ConsciousnessTopology::dense_network(n_nodes, HDC_DIMENSION, None, seed)
    });

    test_and_record("Modular", &mut results, n_samples, |seed| {
        ConsciousnessTopology::modular(n_nodes, HDC_DIMENSION, 2, seed)
    });

    test_and_record("Lattice", &mut results, n_samples, |seed| {
        ConsciousnessTopology::lattice(n_nodes, HDC_DIMENSION, seed)
    });

    // ===== Tier 1 Exotic Topologies (Quick Reference) =====
    println!("\n{}", "=".repeat(80));
    println!("🌟 Tier 1 Exotic Topologies (Quick Test)...\n");

    test_and_record("Small-World", &mut results, n_samples, |seed| {
        ConsciousnessTopology::small_world(n_nodes, HDC_DIMENSION, 4, 0.1, seed)
    });

    test_and_record("Möbius Strip", &mut results, n_samples, |seed| {
        ConsciousnessTopology::mobius_strip(n_nodes, HDC_DIMENSION, seed)
    });

    test_and_record("Torus (3×3)", &mut results, n_samples, |seed| {
        ConsciousnessTopology::torus(3, 3, HDC_DIMENSION, seed)
    });

    // ===== NEW: Tier 2 Exotic Topologies =====
    println!("\n{}", "=".repeat(80));
    println!("🔬 TIER 2 EXOTIC TOPOLOGIES (NEW!)");
    println!("{}", "=".repeat(80));
    println!();

    // 12. Klein Bottle (2D non-orientable)
    println!("1️⃣2️⃣ Testing Klein Bottle topology (3×3 grid = 9 nodes)...");
    let (real_phi, real_std, binary_phi, binary_std) = test_topology_samples(
        n_samples,
        |seed| ConsciousnessTopology::klein_bottle(3, 3, HDC_DIMENSION, seed)
    );
    results.push(("Klein Bottle (3×3)".to_string(), real_phi, real_std, binary_phi, binary_std));
    println!("    RealHV Φ: {:.4} ± {:.4}", real_phi, real_std);
    println!("    Binary Φ: {:.4} ± {:.4}", binary_phi, binary_std);
    println!("    🍾 NON-ORIENTABLE 2D surface - tests if 2D twist also fails!\n");

    // 13. Hyperbolic (negative curvature)
    println!("1️⃣3️⃣ Testing Hyperbolic topology (branching=2, 8 nodes)...");
    let (real_phi, real_std, binary_phi, binary_std) = test_topology_samples(
        n_samples,
        |seed| ConsciousnessTopology::hyperbolic(n_nodes, HDC_DIMENSION, 2, seed)
    );
    results.push(("Hyperbolic".to_string(), real_phi, real_std, binary_phi, binary_std));
    println!("    RealHV Φ: {:.4} ± {:.4}", real_phi, real_std);
    println!("    Binary Φ: {:.4} ± {:.4}", binary_phi, binary_std);
    println!("    🌀 NEGATIVE CURVATURE - exponential expansion!\n");

    // 14. Scale-Free (Barabási-Albert)
    println!("1️⃣4️⃣ Testing Scale-Free topology (m=2, 8 nodes)...");
    let (real_phi, real_std, binary_phi, binary_std) = test_topology_samples(
        n_samples,
        |seed| ConsciousnessTopology::scale_free(n_nodes, HDC_DIMENSION, 2, seed)
    );
    results.push(("Scale-Free".to_string(), real_phi, real_std, binary_phi, binary_std));
    println!("    RealHV Φ: {:.4} ± {:.4}", real_phi, real_std);
    println!("    Binary Φ: {:.4} ± {:.4}", binary_phi, binary_std);
    println!("    📊 POWER-LAW hubs - matches brain/Internet!\n");

    // ===== COMPREHENSIVE RESULTS SUMMARY =====

    println!("\n{}", "=".repeat(80));
    println!("📊 COMPREHENSIVE RESULTS - ALL 14 TOPOLOGIES");
    println!("{}", "=".repeat(80));
    println!();

    // Sort by RealHV Φ (descending)
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("🏆 RANKED BY REAL-VALUED Φ (Continuous Method):");
    println!("{}", "-".repeat(80));
    println!("{:<20} {:>12} {:>12} {:>12} {:>12}", "Topology", "RealHV Φ", "Std Dev", "Binary Φ", "Std Dev");
    println!("{}", "-".repeat(80));

    for (rank, (name, real_phi, real_std, binary_phi, binary_std)) in results.iter().enumerate() {
        let medal = match rank {
            0 => "🥇",
            1 => "🥈",
            2 => "🥉",
            _ => "  ",
        };

        // Highlight Tier 2 topologies
        let marker = if name.contains("Klein") || name.contains("Hyperbolic") || name.contains("Scale-Free") {
            "🆕"
        } else {
            "  "
        };

        println!("{}{} {:<18} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
                 medal, marker, name, real_phi, real_std, binary_phi, binary_std);
    }

    println!("\n{}", "=".repeat(80));

    // Sort by Binary Φ (descending)
    let mut binary_sorted = results.clone();
    binary_sorted.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());

    println!("🏆 RANKED BY BINARY Φ (Probabilistic Binarization):");
    println!("{}", "-".repeat(80));
    println!("{:<20} {:>12} {:>12} {:>12} {:>12}", "Topology", "Binary Φ", "Std Dev", "RealHV Φ", "Std Dev");
    println!("{}", "-".repeat(80));

    for (rank, (name, real_phi, real_std, binary_phi, binary_std)) in binary_sorted.iter().enumerate() {
        let medal = match rank {
            0 => "🥇",
            1 => "🥈",
            2 => "🥉",
            _ => "  ",
        };

        let marker = if name.contains("Klein") || name.contains("Hyperbolic") || name.contains("Scale-Free") {
            "🆕"
        } else {
            "  "
        };

        println!("{}{} {:<18} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
                 medal, marker, name, binary_phi, binary_std, real_phi, real_std);
    }

    println!("\n{}", "=".repeat(80));
    println!("✅ TIER 2 EXOTIC TOPOLOGIES VALIDATION COMPLETE!");
    println!("{}", "=".repeat(80));

    // Key findings summary
    println!("\n📝 Key Hypotheses Tested:");
    println!("   ❓ Does Klein Bottle (2D non-orientable) fail like Möbius (1D)?");
    println!("   ❓ Does hyperbolic curvature affect Φ?");
    println!("   ❓ Do power-law hubs achieve high Φ?");

    println!("\n🔬 Next Steps:");
    println!("   - Tier 3: Fractal, Quantum, Hypercube (4D/5D)");
    println!("   - Publication: Complete topology-Φ characterization");
    println!("   - Analysis: Which topologies maximize consciousness?");

    println!("\n🚀 Ready for final Tier 3 and publication!");
    println!();
}

/// Helper function to test and record results
fn test_and_record<F>(
    name: &str,
    results: &mut Vec<(String, f64, f64, f64, f64)>,
    n_samples: usize,
    generator: F,
) where
    F: Fn(u64) -> ConsciousnessTopology,
{
    let (real_phi, real_std, binary_phi, binary_std) = test_topology_samples(n_samples, generator);
    results.push((name.to_string(), real_phi, real_std, binary_phi, binary_std));
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
