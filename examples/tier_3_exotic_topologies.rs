/// Tier 3 Exotic Topologies Validation
///
/// This example validates the 3 Tier 3 exotic topologies (Fractal, Hypercube, Quantum)
/// testing advanced hypotheses about consciousness topology relationships.
///
/// Based on Tier 1 + Tier 2 results:
/// - Ring/Torus achieve highest Φ (dimensional invariance confirmed)
/// - Klein Bottle (2D non-orientable) succeeds (0.4941, 3rd place)
/// - Möbius Strip (1D non-orientable) fails (0.3729, 14th place)
///
/// Tests:
/// - Fractal: Self-similar hierarchical structure (predicted: medium Φ ≈ 0.46-0.48)
/// - Hypercube (3D/4D/5D): Tests dimensional scaling beyond 2D (predicted: ≈ 0.495 if invariant)
/// - Quantum: Superposition of Ring+Star+Random (predicted: high Φ ≈ 0.48-0.50)
///
/// Comparison with all 17 topologies (8 original + 3 Tier 1 + 3 Tier 2 + 3 Tier 3).

use symthaea::hdc::{
    consciousness_topology_generators::ConsciousnessTopology,
    phi_real::RealPhiCalculator,
    tiered_phi::global_phi,
    phi_topology_validation::real_hv_to_hv16_probabilistic,
    HDC_DIMENSION,
};

fn main() {
    println!("\n🌀 Tier 3 Exotic Topologies Φ Validation");
    println!("{}", "=".repeat(80));
    println!("\n📐 Configuration:");
    println!("   Dimension: {}", HDC_DIMENSION);
    println!("   Samples per topology: 10");
    println!("   Methods: RealHV Φ (continuous) + Binary Φ (probabilistic binarization)");
    println!("\n📊 Testing All 17 Topologies (8 Original + 3 Tier 1 + 3 Tier 2 + 3 Tier 3)...\\n");

    let n_nodes = 8;
    let n_samples = 10;

    let mut results: Vec<(String, f64, f64, f64, f64)> = Vec::new();

    // ===== Original 8 Topologies (Quick Reference) =====
    println!("📚 Original 8 Topologies (Quick Test)...\\n");

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
    println!("🌟 Tier 1 Exotic Topologies (Quick Test)...\\n");

    test_and_record("Small-World", &mut results, n_samples, |seed| {
        ConsciousnessTopology::small_world(n_nodes, HDC_DIMENSION, 4, 0.1, seed)
    });

    test_and_record("Möbius Strip", &mut results, n_samples, |seed| {
        ConsciousnessTopology::mobius_strip(n_nodes, HDC_DIMENSION, seed)
    });

    test_and_record("Torus (3×3)", &mut results, n_samples, |seed| {
        ConsciousnessTopology::torus(3, HDC_DIMENSION, seed)
    });

    // ===== Tier 2 Exotic Topologies (Quick Reference) =====
    println!("\n{}", "=".repeat(80));
    println!("🔬 Tier 2 Exotic Topologies (Quick Test)...\\n");

    test_and_record("Klein Bottle (3×3)", &mut results, n_samples, |seed| {
        ConsciousnessTopology::klein_bottle(3, HDC_DIMENSION, seed)
    });

    test_and_record("Hyperbolic", &mut results, n_samples, |seed| {
        ConsciousnessTopology::hyperbolic(n_nodes, HDC_DIMENSION, 2, seed)
    });

    test_and_record("Scale-Free", &mut results, n_samples, |seed| {
        ConsciousnessTopology::scale_free(n_nodes, HDC_DIMENSION, 2, seed)
    });

    // ===== NEW: Tier 3 Exotic Topologies =====
    println!("\n{}", "=".repeat(80));
    println!("🚀 TIER 3 EXOTIC TOPOLOGIES (NEW!)");
    println!("{}", "=".repeat(80));
    println!();

    // 15. Fractal Network (Self-similar hierarchical structure)
    println!("1️⃣5️⃣ Testing Fractal Network topology (8 nodes, Sierpiński-inspired)...");
    let (real_phi, real_std, binary_phi, binary_std) = test_topology_samples(
        n_samples,
        |seed| ConsciousnessTopology::fractal(n_nodes, HDC_DIMENSION, seed)
    );
    results.push(("Fractal".to_string(), real_phi, real_std, binary_phi, binary_std));
    println!("    RealHV Φ: {:.4} ± {:.4}", real_phi, real_std);
    println!("    Binary Φ: {:.4} ± {:.4}", binary_phi, binary_std);
    println!("    🔺 SELF-SIMILAR structure - hierarchical cross-scale connections!\\n");

    // 16. Hypercube 3D (Cube)
    println!("1️⃣6️⃣ Testing Hypercube 3D topology (8 nodes = cube)...");
    let (real_phi, real_std, binary_phi, binary_std) = test_topology_samples(
        n_samples,
        |seed| ConsciousnessTopology::hypercube(3, HDC_DIMENSION, seed)
    );
    results.push(("Hypercube 3D".to_string(), real_phi, real_std, binary_phi, binary_std));
    println!("    RealHV Φ: {:.4} ± {:.4}", real_phi, real_std);
    println!("    Binary Φ: {:.4} ± {:.4}", binary_phi, binary_std);
    println!("    📦 3D CUBE - each vertex has 3 neighbors!\\n");

    // 17. Hypercube 4D (Tesseract)
    println!("1️⃣7️⃣ Testing Hypercube 4D topology (16 nodes = tesseract)...");
    let (real_phi, real_std, binary_phi, binary_std) = test_topology_samples(
        n_samples,
        |seed| ConsciousnessTopology::hypercube(4, HDC_DIMENSION, seed)
    );
    results.push(("Hypercube 4D".to_string(), real_phi, real_std, binary_phi, binary_std));
    println!("    RealHV Φ: {:.4} ± {:.4}", real_phi, real_std);
    println!("    Binary Φ: {:.4} ± {:.4}", binary_phi, binary_std);
    println!("    🎲 4D TESSERACT - each vertex has 4 neighbors!\\n");

    // 18. Quantum Network (Equal superposition)
    println!("1️⃣8️⃣ Testing Quantum Network topology (equal Ring+Star+Random)...");
    let (real_phi, real_std, binary_phi, binary_std) = test_topology_samples(
        n_samples,
        |seed| ConsciousnessTopology::quantum(n_nodes, HDC_DIMENSION, (1.0, 1.0, 1.0), seed)
    );
    results.push(("Quantum (1:1:1)".to_string(), real_phi, real_std, binary_phi, binary_std));
    println!("    RealHV Φ: {:.4} ± {:.4}", real_phi, real_std);
    println!("    Binary Φ: {:.4} ± {:.4}", binary_phi, binary_std);
    println!("    ⚛️  QUANTUM SUPERPOSITION - blend of Ring, Star, Random!\\n");

    // 19. Quantum Network (Ring-biased: 3:1:1)
    println!("1️⃣9️⃣ Testing Quantum Network topology (Ring-biased 3:1:1)...");
    let (real_phi, real_std, binary_phi, binary_std) = test_topology_samples(
        n_samples,
        |seed| ConsciousnessTopology::quantum(n_nodes, HDC_DIMENSION, (3.0, 1.0, 1.0), seed)
    );
    results.push(("Quantum (3:1:1 Ring)".to_string(), real_phi, real_std, binary_phi, binary_std));
    println!("    RealHV Φ: {:.4} ± {:.4}", real_phi, real_std);
    println!("    Binary Φ: {:.4} ± {:.4}", binary_phi, binary_std);
    println!("    🌊 RING-BIASED quantum state - should approach Ring Φ!\\n");

    // ===== COMPREHENSIVE RESULTS SUMMARY =====

    println!("\n{}", "=".repeat(80));
    println!("📊 COMPREHENSIVE RESULTS - ALL 19 TOPOLOGIES");
    println!("{}", "=".repeat(80));
    println!();

    // Sort by RealHV Φ (descending)
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("🏆 RANKED BY REAL-VALUED Φ (Continuous Method):");
    println!("{}", "-".repeat(80));
    println!("{:<25} {:>12} {:>12} {:>12} {:>12}", "Topology", "RealHV Φ", "Std Dev", "Binary Φ", "Std Dev");
    println!("{}", "-".repeat(80));

    for (rank, (name, real_phi, real_std, binary_phi, binary_std)) in results.iter().enumerate() {
        let medal = match rank {
            0 => "🥇",
            1 => "🥈",
            2 => "🥉",
            _ => "  ",
        };

        // Highlight Tier 3 topologies
        let marker = if name.contains("Fractal") || name.contains("Hypercube") || name.contains("Quantum") {
            "🆕"
        } else {
            "  "
        };

        println!("{}{} {:<23} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
                 medal, marker, name, real_phi, real_std, binary_phi, binary_std);
    }

    println!("\n{}", "=".repeat(80));

    // Sort by Binary Φ (descending)
    let mut binary_sorted = results.clone();
    binary_sorted.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());

    println!("🏆 RANKED BY BINARY Φ (Probabilistic Binarization):");
    println!("{}", "-".repeat(80));
    println!("{:<25} {:>12} {:>12} {:>12} {:>12}", "Topology", "Binary Φ", "Std Dev", "RealHV Φ", "Std Dev");
    println!("{}", "-".repeat(80));

    for (rank, (name, real_phi, real_std, binary_phi, binary_std)) in binary_sorted.iter().enumerate() {
        let medal = match rank {
            0 => "🥇",
            1 => "🥈",
            2 => "🥉",
            _ => "  ",
        };

        let marker = if name.contains("Fractal") || name.contains("Hypercube") || name.contains("Quantum") {
            "🆕"
        } else {
            "  "
        };

        println!("{}{} {:<23} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
                 medal, marker, name, binary_phi, binary_std, real_phi, real_std);
    }

    println!("\n{}", "=".repeat(80));
    println!("✅ TIER 3 EXOTIC TOPOLOGIES VALIDATION COMPLETE!");
    println!("{}", "=".repeat(80));

    // Key findings summary
    println!("\n📝 Key Hypotheses Tested:");
    println!("   ❓ Does Fractal (self-similar) achieve high Φ?");
    println!("   ❓ Do Hypercubes (3D/4D) preserve dimensional invariance?");
    println!("   ❓ Can Quantum superposition exceed single-topology Φ?");

    println!("\n🔬 Scientific Questions:");
    println!("   - Does self-similarity enhance or reduce integration?");
    println!("   - How does Φ scale from 2D (Torus) to 3D/4D (Hypercube)?");
    println!("   - Can topology superposition create emergent benefits?");

    println!("\n🎯 Comparison Points:");
    println!("   - Ring/Torus Φ = 0.4954 (highest so far, 1D/2D invariance)");
    println!("   - Klein Bottle Φ = 0.4941 (3rd place, 2D non-orientable success)");
    println!("   - Möbius Strip Φ = 0.3729 (14th place, 1D non-orientable failure)");

    println!("\n🚀 Complete exotic topology research (Tier 1 + 2 + 3) finished!");
    println!("   Ready for publication and analysis!");
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
