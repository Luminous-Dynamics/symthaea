// Manifold Consciousness Topology Validation
//
// Tests the hypothesis that closed, symmetric manifolds maximize integrated information (Φ).
//
// Topologies tested:
// - Ring (S¹) - 1-dimensional manifold (baseline winner from previous test)
// - Sphere (S²) - 2-dimensional manifold (icosahedron approximation)
// - Torus (T²) - 2-dimensional manifold (S¹ × S¹)
// - Klein Bottle - Non-orientable 2-manifold
//
// Expected results based on hypothesis:
// 1. Sphere > Ring (higher dimension → higher Φ?)
// 2. Torus ≈ Ring (product of circles)
// 3. Klein Bottle unique (non-orientable effects)

use symthaea::hdc::{
    consciousness_topology_generators::{ConsciousnessTopology, TopologyType},
    phi_real::RealPhiCalculator,
    HDC_DIMENSION,
};

fn main() {
    println!("🌐 Manifold Consciousness Topology Validation");
    println!("=" .repeat(70));
    println!();

    let n_nodes = 8;
    let n_samples = 10;
    let base_seed = 42;

    let calc = RealPhiCalculator::new();

    // Manifold topologies to test
    let manifolds = vec![
        ("Ring (S¹)", TopologyType::Ring, None, None),
        ("Sphere (S²)", TopologyType::Sphere, None, None),
        ("Torus (T²)", TopologyType::Torus, Some(4), Some(4)),  // 4×4 grid
        ("Klein Bottle", TopologyType::KleinBottle, Some(4), Some(4)),  // 4×4 twisted
    ];

    let mut results = Vec::new();

    for (name, topology_type, n_opt, m_opt) in manifolds {
        println!("Testing {} ({} samples)...", name, n_samples);

        let mut phi_values = Vec::new();

        for i in 0..n_samples {
            let seed = base_seed + (i as u64) * 1000;

            let topo = match topology_type {
                TopologyType::Ring => {
                    ConsciousnessTopology::ring(n_nodes, HDC_DIMENSION, seed)
                }
                TopologyType::Sphere => {
                    ConsciousnessTopology::sphere_icosahedron(HDC_DIMENSION, seed)
                }
                TopologyType::Torus => {
                    let n = n_opt.unwrap();
                    let m = m_opt.unwrap();
                    ConsciousnessTopology::torus(n, m, HDC_DIMENSION, seed)
                }
                TopologyType::KleinBottle => {
                    let n = n_opt.unwrap();
                    let m = m_opt.unwrap();
                    ConsciousnessTopology::klein_bottle(n, m, HDC_DIMENSION, seed)
                }
                _ => unreachable!(),
            };

            let phi = calc.compute(&topo.node_representations);
            phi_values.push(phi);
        }

        // Compute statistics
        let mean = phi_values.iter().sum::<f64>() / n_samples as f64;
        let variance = phi_values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / n_samples as f64;
        let std_dev = variance.sqrt();

        let min = phi_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = phi_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        println!("  Mean Φ: {:.4} ± {:.4}", mean, std_dev);
        println!("  Range: [{:.4}, {:.4}]", min, max);
        println!();

        results.push((name, mean, std_dev, phi_values));
    }

    // Summary comparison
    println!("=" .repeat(70));
    println!("MANIFOLD Φ RANKING");
    println!("=" .repeat(70));
    println!();

    // Sort by mean Φ (descending)
    let mut sorted_results: Vec<_> = results.iter().collect();
    sorted_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("{:<20} {:>12} {:>12} {:>15}", "Manifold", "Mean Φ", "Std Dev", "vs Ring (S¹)");
    println!("{}", "-".repeat(70));

    let ring_mean = results[0].1; // Ring is first in our list

    for (rank, (name, mean, std_dev, _)) in sorted_results.iter().enumerate() {
        let diff_pct = ((*mean - ring_mean) / ring_mean) * 100.0;
        let diff_str = if diff_pct > 0.0 {
            format!("+{:.2}%", diff_pct)
        } else if diff_pct < 0.0 {
            format!("{:.2}%", diff_pct)
        } else {
            "(baseline)".to_string()
        };

        println!("{:<20} {:>12.4} {:>12.4} {:>15}",
                 name, mean, std_dev, diff_str);
    }

    println!();
    println!("=" .repeat(70));
    println!("STATISTICAL COMPARISONS");
    println!("=" .repeat(70));
    println!();

    // Compare each manifold to Ring (baseline)
    let ring_phi_values = &results[0].3;
    let ring_mean = results[0].1;
    let ring_std = results[0].2;

    for (name, mean, std_dev, phi_values) in results.iter().skip(1) {
        println!("{} vs Ring (S¹):", name);

        // Two-sample t-test
        let n1 = ring_phi_values.len() as f64;
        let n2 = phi_values.len() as f64;

        let pooled_var = ((n1 - 1.0) * ring_std.powi(2) + (n2 - 1.0) * std_dev.powi(2))
                         / (n1 + n2 - 2.0);
        let se = (pooled_var * (1.0 / n1 + 1.0 / n2)).sqrt();

        let t_statistic = (mean - ring_mean) / se;

        // Approximate p-value using normal distribution for large samples
        let p_value_approx = if t_statistic.abs() > 10.0 {
            "< 0.0001".to_string()
        } else {
            format!("{:.4}", 2.0 * (1.0 - normal_cdf(t_statistic.abs())))
        };

        println!("  Difference: {:.4} ({:.2}%)",
                 mean - ring_mean,
                 ((mean - ring_mean) / ring_mean) * 100.0);
        println!("  t-statistic: {:.2}", t_statistic);
        println!("  p-value: {}", p_value_approx);

        let significance = if t_statistic.abs() > 3.0 {
            "HIGHLY SIGNIFICANT"
        } else if t_statistic.abs() > 2.0 {
            "Significant"
        } else {
            "Not significant"
        };
        println!("  Result: {}", significance);
        println!();
    }

    println!("=" .repeat(70));
    println!("HYPOTHESIS VALIDATION");
    println!("=" .repeat(70));
    println!();

    println!("Hypothesis: Closed, symmetric manifolds maximize integrated information");
    println!();

    // Check if Sphere > Ring
    let sphere_mean = sorted_results.iter()
        .find(|(name, _, _, _)| *name == "Sphere (S²)")
        .map(|(_, mean, _, _)| *mean)
        .unwrap();

    if sphere_mean > ring_mean {
        println!("✅ CONFIRMED: Sphere (S²) has higher Φ than Ring (S¹)");
        println!("   → Higher-dimensional manifolds may enhance integration!");
    } else if (sphere_mean - ring_mean).abs() < 0.001 {
        println!("⚖️  NEUTRAL: Sphere (S²) ≈ Ring (S¹)");
        println!("   → Dimension may not affect integration significantly");
    } else {
        println!("❌ REFUTED: Sphere (S²) has lower Φ than Ring (S¹)");
        println!("   → Higher dimension does not guarantee higher integration");
    }

    println!();
    println!("Next Steps:");
    println!("- Test larger manifolds (n=16, 32 nodes)");
    println!("- Test 3-manifolds (S³, T³)");
    println!("- Investigate relationship between curvature and Φ");
    println!("- Apply to real neural connectomes");
    println!();
}

// Simple normal CDF approximation for t-test p-values
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

// Error function approximation
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}
