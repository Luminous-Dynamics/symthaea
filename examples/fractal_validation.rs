// Fractal Topology Consciousness Validation
//
// Tests the hypothesis that fractal dimension affects integrated information (Φ).
//
// Topologies tested:
// - Sierpinski Gasket (depth 3) - Fractal dimension d≈1.585
// - Fractal Tree (depth 4, branching 2) - Self-similar binary branching
// - Binary Tree (depth 4) - Baseline non-fractal comparison
// - Ring (8 nodes) - 1D manifold baseline
// - Torus (3×3 grid) - 2D manifold baseline
//
// Research Questions:
// 1. Does fractal dimension (d≈1.585) produce Φ between 1D and 2D manifolds?
// 2. Does self-similarity provide integration benefits vs regular structures?
// 3. Does Fractal Tree outperform Binary Tree due to uniform branching?
//
// Predictions:
// - Sierpinski Gasket: Φ ≈ 0.47-0.48 (between Line and Ring)
// - Fractal Tree: Φ ≈ 0.47-0.48 (slightly better than Binary Tree ~0.4668)
// - Both should show lower variance than non-fractal counterparts

use symthaea::hdc::{
    consciousness_topology_generators::{ConsciousnessTopology, TopologyType},
    phi_real::RealPhiCalculator,
    HDC_DIMENSION,
};

fn main() {
    println!("🔬 Fractal Topology Consciousness Validation");
    println!("{}", "=".repeat(70));
    println!();
    println!("Testing hypothesis: Fractal dimension affects integrated information (Φ)");
    println!();

    let n_samples = 10;
    let base_seed = 42;
    let calc = RealPhiCalculator::new();

    // Define topologies to test
    let topologies = vec![
        (
            "Ring (8 nodes)",
            TopologyType::Ring,
            vec![8],
            "1D manifold baseline - Φ ≈ 0.4954",
        ),
        (
            "Torus (3×3)",
            TopologyType::Torus,
            vec![3, 3],
            "2D manifold baseline - Φ ≈ 0.4954",
        ),
        (
            "Binary Tree (depth 4)",
            TopologyType::BinaryTree,
            vec![15],
            "Non-fractal tree - Φ ≈ 0.4668",
        ),
        (
            "Sierpinski Gasket (depth 3)",
            TopologyType::SierpinskiGasket,
            vec![3],
            "Fractal d≈1.585 - Predicted Φ ≈ 0.47-0.48",
        ),
        (
            "Fractal Tree (depth 4, branch 2)",
            TopologyType::FractalTree,
            vec![4, 2],
            "Self-similar tree - Predicted Φ ≈ 0.47-0.48",
        ),
    ];

    let mut results = Vec::new();

    for (name, topology_type, params, description) in &topologies {
        println!("Testing {} ({} samples)...", name, n_samples);
        println!("  {}", description);

        let mut phi_values = Vec::new();

        for i in 0..n_samples {
            let seed = base_seed + (i as u64) * 1000;

            let topo = match topology_type {
                TopologyType::Ring => {
                    ConsciousnessTopology::ring(params[0], HDC_DIMENSION, seed)
                }
                TopologyType::Torus => {
                    ConsciousnessTopology::torus(params[0], params[1], HDC_DIMENSION, seed)
                }
                TopologyType::BinaryTree => {
                    ConsciousnessTopology::binary_tree(params[0], HDC_DIMENSION, seed)
                }
                TopologyType::SierpinskiGasket => {
                    ConsciousnessTopology::sierpinski_gasket(params[0], HDC_DIMENSION, seed)
                }
                TopologyType::FractalTree => {
                    ConsciousnessTopology::fractal_tree(params[0], params[1], HDC_DIMENSION, seed)
                }
                _ => unreachable!(),
            };

            let phi = calc.compute(&topo.node_representations);
            phi_values.push(phi);
        }

        // Compute statistics
        let mean = phi_values.iter().sum::<f64>() / n_samples as f64;
        let variance = phi_values
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / n_samples as f64;
        let std_dev = variance.sqrt();

        let min = phi_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = phi_values
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        println!("  Mean Φ: {:.4} ± {:.4}", mean, std_dev);
        println!("  Range: [{:.4}, {:.4}]", min, max);
        println!();

        results.push((name, mean, std_dev, phi_values, topology_type));
    }

    // Summary comparison
    println!("{}", "=".repeat(70));
    println!("FRACTAL Φ RANKING");
    println!("{}", "=".repeat(70));
    println!();

    // Sort by mean Φ (descending)
    let mut sorted_results: Vec<_> = results.iter().collect();
    sorted_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!(
        "{:<35} {:>12} {:>12}",
        "Topology", "Mean Φ", "Std Dev"
    );
    println!("{}", "-".repeat(60));

    for (rank, (name, mean, std_dev, _, _)) in sorted_results.iter().enumerate() {
        let rank_symbol = match rank {
            0 => "🥇",
            1 => "🥈",
            2 => "🥉",
            _ => "  ",
        };
        println!(
            "{} {:<33} {:>12.4} {:>12.4}",
            rank_symbol, name, mean, std_dev
        );
    }

    println!();
    println!("{}", "=".repeat(70));
    println!("FRACTAL vs BASELINE COMPARISONS");
    println!("{}", "=".repeat(70));
    println!();

    // Find baselines
    let ring_mean = results
        .iter()
        .find(|(name, _, _, _, _)| name.contains("Ring"))
        .map(|(_, mean, _, _, _)| *mean)
        .unwrap();

    let binary_tree_mean = results
        .iter()
        .find(|(name, _, _, _, _)| name.contains("Binary Tree"))
        .map(|(_, mean, _, _, _)| *mean)
        .unwrap();

    // Compare Sierpinski to Ring
    let sierpinski = results
        .iter()
        .find(|(name, _, _, _, _)| name.contains("Sierpinski"))
        .unwrap();

    println!("Sierpinski Gasket (d≈1.585) vs Ring (1D):");
    println!(
        "  Sierpinski Φ: {:.4}, Ring Φ: {:.4}",
        sierpinski.1, ring_mean
    );
    let diff_pct = ((sierpinski.1 - ring_mean) / ring_mean) * 100.0;
    println!("  Difference: {:.2}%", diff_pct);

    if diff_pct.abs() < 1.0 {
        println!("  ⚖️  RESULT: Sierpinski ≈ Ring (within 1%)");
        println!("     → Fractal dimension d≈1.585 does NOT produce intermediate Φ");
        println!("     → Integration depends on local uniformity, not fractal dimension");
    } else if sierpinski.1 > ring_mean {
        println!("  ✅ RESULT: Sierpinski > Ring");
        println!("     → Self-similarity may provide integration benefits");
    } else {
        println!("  ❌ RESULT: Sierpinski < Ring");
        println!("     → Hierarchical structure reduces integration");
    }
    println!();

    // Compare Fractal Tree to Binary Tree
    let fractal_tree = results
        .iter()
        .find(|(name, _, _, _, _)| name.contains("Fractal Tree"))
        .unwrap();

    println!("Fractal Tree vs Binary Tree:");
    println!(
        "  Fractal Tree Φ: {:.4}, Binary Tree Φ: {:.4}",
        fractal_tree.1, binary_tree_mean
    );
    let diff_pct = ((fractal_tree.1 - binary_tree_mean) / binary_tree_mean) * 100.0;
    println!("  Difference: {:.2}%", diff_pct);

    if diff_pct > 1.0 {
        println!("  ✅ CONFIRMED: Self-similarity improves integration");
        println!("     → Uniform branching at all scales enhances Φ");
    } else if diff_pct.abs() < 1.0 {
        println!("  ⚖️  NEUTRAL: Fractal Tree ≈ Binary Tree");
        println!("     → Self-similarity doesn't significantly affect integration");
    } else {
        println!("  ❌ REFUTED: Fractal Tree < Binary Tree");
        println!("     → Sibling connections may create local clusters");
    }
    println!();

    println!("{}", "=".repeat(70));
    println!("STATISTICAL ANALYSIS");
    println!("{}", "=".repeat(70));
    println!();

    // Two-sample t-tests
    for (i, (name, mean, std_dev, phi_values, _)) in results.iter().enumerate() {
        if name.contains("Sierpinski") || name.contains("Fractal Tree") {
            // Find comparison baseline
            let (baseline_name, baseline_mean, baseline_std, baseline_values) =
                if name.contains("Sierpinski") {
                    let ring = results
                        .iter()
                        .find(|(n, _, _, _, _)| n.contains("Ring"))
                        .unwrap();
                    (ring.0, ring.1, ring.2, &ring.3)
                } else {
                    let tree = results
                        .iter()
                        .find(|(n, _, _, _, _)| n.contains("Binary Tree"))
                        .unwrap();
                    (tree.0, tree.1, tree.2, &tree.3)
                };

            println!("{} vs {}:", name, baseline_name);

            let n1 = baseline_values.len() as f64;
            let n2 = phi_values.len() as f64;

            let pooled_var =
                ((n1 - 1.0) * baseline_std.powi(2) + (n2 - 1.0) * std_dev.powi(2))
                    / (n1 + n2 - 2.0);
            let se = (pooled_var * (1.0 / n1 + 1.0 / n2)).sqrt();

            let t_statistic = (mean - baseline_mean) / se;

            println!(
                "  Difference: {:.4} ({:.2}%)",
                mean - baseline_mean,
                ((mean - baseline_mean) / baseline_mean) * 100.0
            );
            println!("  t-statistic: {:.2}", t_statistic);

            let significance = if t_statistic.abs() > 3.0 {
                "HIGHLY SIGNIFICANT (p < 0.01)"
            } else if t_statistic.abs() > 2.0 {
                "Significant (p < 0.05)"
            } else {
                "Not significant (p > 0.05)"
            };
            println!("  {}", significance);
            println!();
        }
    }

    println!("{}", "=".repeat(70));
    println!("HYPOTHESIS VALIDATION");
    println!("{}", "=".repeat(70));
    println!();

    println!("Hypothesis 1: Fractal dimension determines Φ");
    println!("  Status: Testing d≈1.585 (Sierpinski) should place Φ between 1D and 2D");
    println!();

    println!("Hypothesis 2: Self-similarity enhances integration");
    println!("  Status: Fractal Tree should outperform Binary Tree");
    println!();

    println!("Next Steps:");
    println!("- Test larger fractal iterations (depth 5, 6)");
    println!("- Test higher branching factors (3, 4 for Fractal Tree)");
    println!("- Implement Koch Snowflake (d≈1.262) for lower dimension test");
    println!("- Implement Menger Sponge (d≈2.727) for higher dimension test");
    println!("- Analyze relationship between fractal dimension and Φ");
    println!();
}
