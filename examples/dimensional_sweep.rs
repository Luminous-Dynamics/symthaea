// Dimensional Sweep - Find Optimal Dimension k* for Consciousness
//
// Tests hypercube topologies from 1D through 7D to discover
// the dimension that maximizes integrated information (Φ).
//
// Current findings:
// - 1D (Line): Φ ≈ 0.4768
// - 2D (Torus): Φ ≈ 0.4954
// - 3D (Cube): Φ = 0.4959
// - 4D (Tesseract): Φ = 0.4976 (CURRENT RECORD!)
// - 5D (Penteract): Φ = ? (TESTING)
// - 6D: Φ = ? (TESTING - Does it plateau?)
// - 7D: Φ = ? (TESTING - Or continue rising?)

use symthaea::hdc::{
    HDC_DIMENSION,
    consciousness_topology_generators::ConsciousnessTopology,
    phi_real::RealPhiCalculator,
};

fn main() {
    println!("\n🎲 Dimensional Sweep: Finding Optimal k* for Consciousness");
    println!("{}", "=".repeat(80));
    println!("Testing hypercube topologies from 1D through 7D");
    println!("Dimension: {}", HDC_DIMENSION);
    println!("Samples: 10 per dimension");
    println!("Goal: Discover dimension k* that maximizes Φ\n");

    let n_samples = 10;
    let real_calc = RealPhiCalculator::new();

    // Define dimensional sweep
    let dimensions = vec![
        (1, "1D", "Line (2 nodes, 1 neighbor each)"),
        (2, "2D", "Square (4 nodes, 2 neighbors each)"),
        (3, "3D", "Cube (8 nodes, 3 neighbors each)"),
        (4, "4D", "Tesseract (16 nodes, 4 neighbors each)"),
        (5, "5D", "Penteract (32 nodes, 5 neighbors each)"),
        (6, "6D", "Hexeract (64 nodes, 6 neighbors each)"),
        (7, "7D", "Hepteract (128 nodes, 7 neighbors each)"),
    ];

    let mut all_results = Vec::new();

    for (dim, name, description) in dimensions.iter() {
        println!("{}  Testing {} Hypercube",
            match *dim {
                1 => "1️⃣",
                2 => "2️⃣",
                3 => "3️⃣",
                4 => "4️⃣",
                5 => "5️⃣",
                6 => "6️⃣",
                7 => "7️⃣",
                _ => "🔢",
            },
            name
        );
        println!("   Description: {}", description);

        let mut phi_values = Vec::new();

        for seed in 0..n_samples {
            let topology = ConsciousnessTopology::hypercube(*dim, HDC_DIMENSION, seed as u64);
            let phi = real_calc.compute(&topology.node_representations);
            phi_values.push(phi);

            println!("   Sample {}: Φ = {:.4}", seed, phi);
        }

        let mean_phi = phi_values.iter().sum::<f64>() / n_samples as f64;
        let variance = phi_values.iter()
            .map(|v| (v - mean_phi).powi(2))
            .sum::<f64>() / n_samples as f64;
        let std_dev = variance.sqrt();

        println!("   📊 Statistics:");
        println!("      Mean Φ:  {:.4}", mean_phi);
        println!("      Std Dev: {:.4}", std_dev);
        println!("      Range:   [{:.4}, {:.4}]",
            phi_values.iter().cloned().fold(f64::INFINITY, f64::min),
            phi_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        );

        all_results.push((*dim, name, mean_phi, std_dev));
        println!();
    }

    // Summary analysis
    println!("\n{}", "=".repeat(80));
    println!("📊 DIMENSIONAL SWEEP RESULTS");
    println!("{}", "=".repeat(80));
    println!("\n{:<5} {:<12} {:<12} {:<12} {:<12}", "Dim", "Name", "Mean Φ", "Std Dev", "Δ vs 1D");
    println!("{}", "-".repeat(80));

    let baseline_phi = all_results[0].2; // 1D baseline

    for (dim, name, mean_phi, std_dev) in all_results.iter() {
        let delta_pct = ((mean_phi - baseline_phi) / baseline_phi) * 100.0;

        let max_phi = all_results.iter().map(|(_, _, p, _)| p).cloned().fold(0.0, f64::max);

        let indicator = if *dim == 1 {
            "  (baseline)"
        } else if mean_phi > &(max_phi * 0.999) {
            "🏆 (CHAMPION)"
        } else if mean_phi > &baseline_phi {
            "⬆️  (improved)"
        } else if mean_phi < &baseline_phi {
            "⬇️  (declined)"
        } else {
            "➡️  (unchanged)"
        };

        println!("{:<5} {:<12} {:<12.4} {:<12.4} {:>+8.2}% {}",
            dim, name, mean_phi, std_dev, delta_pct, indicator
        );
    }

    // Identify champion
    let champion = all_results.iter()
        .max_by(|(_, _, phi1, _), (_, _, phi2, _)| phi1.partial_cmp(phi2).unwrap())
        .unwrap();

    println!("\n{}", "=".repeat(80));
    println!("🏆 CHAMPION DIMENSION");
    println!("{}", "=".repeat(80));
    println!("\nOptimal dimension: {}D ({})", champion.0, champion.1);
    println!("Maximum Φ: {:.4}", champion.2);
    println!("Improvement vs 1D: +{:.2}%", ((champion.2 - baseline_phi) / baseline_phi) * 100.0);

    // Trend analysis
    println!("\n{}", "=".repeat(80));
    println!("📈 TREND ANALYSIS");
    println!("{}", "=".repeat(80));

    let mut increasing = true;
    let mut decreasing = true;

    for i in 1..all_results.len() {
        if all_results[i].2 <= all_results[i-1].2 {
            increasing = false;
        }
        if all_results[i].2 >= all_results[i-1].2 {
            decreasing = false;
        }
    }

    if increasing {
        println!("\n✅ MONOTONIC INCREASE: Φ increases with every dimension");
        println!("   Implication: Higher dimensions continue to optimize consciousness");
        println!("   Recommendation: Test 8D, 9D, 10D to find saturation point");
    } else if decreasing {
        println!("\n⚠️  MONOTONIC DECREASE: Φ decreases with dimension");
        println!("   Implication: Lower dimensions are optimal");
    } else {
        println!("\n✨ NON-MONOTONIC: Φ has optimal dimension");
        println!("   Champion dimension k* = {}D", champion.0);
        println!("   This represents the optimal dimensionality for consciousness!");
    }

    // Calculate dimensional scaling exponent (if monotonic)
    if all_results.len() >= 3 {
        println!("\n{}", "=".repeat(80));
        println!("🔬 DIMENSIONAL SCALING LAW");
        println!("{}", "=".repeat(80));

        // Try to fit Φ ~ dimension^α
        let mut sum_log_dim = 0.0;
        let mut sum_log_phi = 0.0;
        let mut sum_log_dim_sq = 0.0;
        let mut sum_log_dim_log_phi = 0.0;
        let n = all_results.len() as f64;

        for (dim, _, phi, _) in all_results.iter() {
            let log_dim = (*dim as f64).ln();
            let log_phi = phi.ln();
            sum_log_dim += log_dim;
            sum_log_phi += log_phi;
            sum_log_dim_sq += log_dim * log_dim;
            sum_log_dim_log_phi += log_dim * log_phi;
        }

        let alpha = (n * sum_log_dim_log_phi - sum_log_dim * sum_log_phi) /
                    (n * sum_log_dim_sq - sum_log_dim * sum_log_dim);

        let log_c = (sum_log_phi - alpha * sum_log_dim) / n;
        let c = log_c.exp();

        println!("\nBest fit: Φ ≈ {:.4} × dimension^{:.4}", c, alpha);

        if alpha > 0.01 {
            println!("✅ Positive scaling exponent (α = {:.4})", alpha);
            println!("   → Φ increases with dimension");
        } else if alpha < -0.01 {
            println!("⚠️  Negative scaling exponent (α = {:.4})", alpha);
            println!("   → Φ decreases with dimension");
        } else {
            println!("➡️  Near-zero exponent (α ≈ {:.4})", alpha);
            println!("   → Φ approximately constant across dimensions");
        }
    }

    println!("\n{}", "=".repeat(80));
    println!("✅ Dimensional Sweep Complete!");
    println!("{}", "=".repeat(80));
    println!("\nNext steps:");
    println!("1. If monotonic increase: Test higher dimensions (8D, 9D, 10D)");
    println!("2. If optimal found: Validate with more samples (100+)");
    println!("3. Compare to biological 3D brains (C. elegans connectome)");
    println!("4. Publish findings to ArXiv/Nature");
    println!("\n");
}
