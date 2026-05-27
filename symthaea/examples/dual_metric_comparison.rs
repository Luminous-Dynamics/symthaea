// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/// Dual-Metric Comparison: Algebraic Connectivity (λ₂) vs IIT-Inspired Φ
///
/// This example runs BOTH metrics on all 19 topologies to determine:
/// 1. Do the metrics correlate?
/// 2. Do they produce the same topology rankings?
/// 3. Can λ₂ serve as a proxy for IIT Φ?
///
/// Results will inform how to frame the paper's claims.
use symthaea::hdc::{
    HDC_DIMENSION, consciousness_topology_generators::ConsciousnessTopology,
    integrated_information::IntegratedInformation,
    phi_topology_validation::real_hv_to_hv16_probabilistic,
    spectral_connectivity::ConnectivityCalculator,
};

fn main() {
    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║         DUAL-METRIC COMPARISON: λ₂ (Spectral) vs Φ (IIT-Inspired)           ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");

    println!("\n📐 Configuration:");
    println!("   HDC Dimension: {}", HDC_DIMENSION);
    println!("   Nodes per topology: 8");
    println!("   Samples per topology: 5");
    println!("   Metrics: λ₂ (ConnectivityCalculator) vs Φ (IntegratedInformation)");

    let n_nodes = 8;
    let n_samples = 5;

    // Results: (name, λ₂_mean, λ₂_std, Φ_mean, Φ_std)
    let mut results: Vec<(String, f64, f64, f64, f64)> = Vec::new();

    println!("\n{}", "=".repeat(80));
    println!("Running measurements on all 19 topologies...\n");

    // ===== All 19 Topologies =====

    // 1. Random
    measure_topology("Random", &mut results, n_nodes, n_samples, |seed| {
        ConsciousnessTopology::random(n_nodes, HDC_DIMENSION, seed)
    });

    // 2. Star
    measure_topology("Star", &mut results, n_nodes, n_samples, |seed| {
        ConsciousnessTopology::star(n_nodes, HDC_DIMENSION, seed)
    });

    // 3. Ring
    measure_topology("Ring", &mut results, n_nodes, n_samples, |seed| {
        ConsciousnessTopology::ring(n_nodes, HDC_DIMENSION, seed)
    });

    // 4. Line
    measure_topology("Line", &mut results, n_nodes, n_samples, |seed| {
        ConsciousnessTopology::line(n_nodes, HDC_DIMENSION, seed)
    });

    // 5. Binary Tree
    measure_topology("Binary Tree", &mut results, n_nodes, n_samples, |seed| {
        ConsciousnessTopology::binary_tree(n_nodes, HDC_DIMENSION, seed)
    });

    // 6. Dense Network
    measure_topology("Dense Network", &mut results, n_nodes, n_samples, |seed| {
        ConsciousnessTopology::dense_network(n_nodes, HDC_DIMENSION, None, seed)
    });

    // 7. Modular
    measure_topology("Modular", &mut results, n_nodes, n_samples, |seed| {
        ConsciousnessTopology::modular(n_nodes, HDC_DIMENSION, 2, seed)
    });

    // 8. Lattice
    measure_topology("Lattice", &mut results, n_nodes, n_samples, |seed| {
        ConsciousnessTopology::lattice(n_nodes, HDC_DIMENSION, seed)
    });

    // 9. Small-World
    measure_topology("Small-World", &mut results, n_nodes, n_samples, |seed| {
        ConsciousnessTopology::small_world(n_nodes, HDC_DIMENSION, 4, 0.1, seed)
    });

    // 10. Möbius Strip
    measure_topology("Möbius Strip", &mut results, n_nodes, n_samples, |seed| {
        ConsciousnessTopology::mobius_strip(n_nodes, HDC_DIMENSION, seed)
    });

    // 11. Torus
    measure_topology("Torus (3×3)", &mut results, n_nodes, n_samples, |seed| {
        ConsciousnessTopology::torus(3, 3, HDC_DIMENSION, seed)
    });

    // 12. Klein Bottle
    measure_topology("Klein Bottle", &mut results, n_nodes, n_samples, |seed| {
        ConsciousnessTopology::klein_bottle(3, 3, HDC_DIMENSION, seed)
    });

    // 13. Hyperbolic
    measure_topology("Hyperbolic", &mut results, n_nodes, n_samples, |seed| {
        ConsciousnessTopology::hyperbolic(n_nodes, HDC_DIMENSION, 2, seed)
    });

    // 14. Scale-Free
    measure_topology("Scale-Free", &mut results, n_nodes, n_samples, |seed| {
        ConsciousnessTopology::scale_free(n_nodes, HDC_DIMENSION, 2, seed)
    });

    // 15. Fractal
    measure_topology("Fractal", &mut results, n_nodes, n_samples, |seed| {
        ConsciousnessTopology::fractal(n_nodes, HDC_DIMENSION, seed)
    });

    // 16. Hypercube 3D
    measure_topology("Hypercube 3D", &mut results, n_nodes, n_samples, |seed| {
        ConsciousnessTopology::hypercube(3, HDC_DIMENSION, seed)
    });

    // 17. Hypercube 4D
    measure_topology("Hypercube 4D", &mut results, n_nodes, n_samples, |seed| {
        ConsciousnessTopology::hypercube(4, HDC_DIMENSION, seed)
    });

    // 18. Quantum (1:1:1)
    measure_topology(
        "Quantum (1:1:1)",
        &mut results,
        n_nodes,
        n_samples,
        |seed| ConsciousnessTopology::quantum(n_nodes, HDC_DIMENSION, (1.0, 1.0, 1.0), seed),
    );

    // 19. Quantum (Ring-biased)
    measure_topology(
        "Quantum (3:1:1)",
        &mut results,
        n_nodes,
        n_samples,
        |seed| ConsciousnessTopology::quantum(n_nodes, HDC_DIMENSION, (3.0, 1.0, 1.0), seed),
    );

    // ===== RESULTS ANALYSIS =====

    println!("\n{}", "═".repeat(80));
    println!("                           RESULTS SUMMARY");
    println!("{}", "═".repeat(80));

    // Sort by λ₂ for first table
    let mut lambda_sorted = results.clone();
    lambda_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\n📊 RANKED BY λ₂ (Algebraic Connectivity / Spectral):");
    println!("{}", "-".repeat(80));
    println!(
        "{:<20} {:>10} {:>8} {:>10} {:>8} {:>8}",
        "Topology", "λ₂", "±", "Φ (IIT)", "±", "λ₂ Rank"
    );
    println!("{}", "-".repeat(80));

    for (rank, (name, l2, l2_std, phi, phi_std)) in lambda_sorted.iter().enumerate() {
        let medal = match rank {
            0 => "🥇",
            1 => "🥈",
            2 => "🥉",
            _ => "  ",
        };
        println!(
            "{} {:<18} {:>10.4} {:>8.4} {:>10.4} {:>8.4} {:>8}",
            medal,
            name,
            l2,
            l2_std,
            phi,
            phi_std,
            rank + 1
        );
    }

    // Sort by Φ for second table
    let mut phi_sorted = results.clone();
    phi_sorted.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());

    println!("\n📊 RANKED BY Φ (IIT-Inspired / Information-Theoretic):");
    println!("{}", "-".repeat(80));
    println!(
        "{:<20} {:>10} {:>8} {:>10} {:>8} {:>8}",
        "Topology", "Φ (IIT)", "±", "λ₂", "±", "Φ Rank"
    );
    println!("{}", "-".repeat(80));

    for (rank, (name, l2, l2_std, phi, phi_std)) in phi_sorted.iter().enumerate() {
        let medal = match rank {
            0 => "🥇",
            1 => "🥈",
            2 => "🥉",
            _ => "  ",
        };
        println!(
            "{} {:<18} {:>10.4} {:>8.4} {:>10.4} {:>8.4} {:>8}",
            medal,
            name,
            phi,
            phi_std,
            l2,
            l2_std,
            rank + 1
        );
    }

    // ===== CORRELATION ANALYSIS =====

    println!("\n{}", "═".repeat(80));
    println!("                        CORRELATION ANALYSIS");
    println!("{}", "═".repeat(80));

    // Extract values for correlation
    let l2_values: Vec<f64> = results.iter().map(|r| r.1).collect();
    let phi_values: Vec<f64> = results.iter().map(|r| r.3).collect();

    // Compute Pearson correlation
    let correlation = pearson_correlation(&l2_values, &phi_values);

    // Compute Spearman rank correlation
    let spearman = spearman_correlation(&l2_values, &phi_values);

    println!("\n📈 Pearson correlation (r):  {:.4}", correlation);
    println!("📈 Spearman correlation (ρ): {:.4}", spearman);

    // Interpretation
    println!("\n🔬 INTERPRETATION:");
    if correlation.abs() > 0.8 {
        println!("   ✅ STRONG correlation - λ₂ can serve as proxy for Φ");
        println!("   → Paper can claim λ₂ predicts IIT-like integration");
    } else if correlation.abs() > 0.5 {
        println!("   ⚠️  MODERATE correlation - metrics capture related but distinct properties");
        println!("   → Paper should report both metrics, note differences");
    } else {
        println!("   ❌ WEAK correlation - metrics measure fundamentally different things");
        println!("   → Paper must be reframed OR focus on the divergence as a finding");
    }

    // Rank comparison
    println!("\n📊 RANK COMPARISON:");
    let mut rank_diff_sum = 0;
    for (name, l2, _, phi, _) in &results {
        let l2_rank = lambda_sorted.iter().position(|r| r.0 == *name).unwrap() + 1;
        let phi_rank = phi_sorted.iter().position(|r| r.0 == *name).unwrap() + 1;
        let diff = (l2_rank as i32 - phi_rank as i32).abs();
        rank_diff_sum += diff;

        if diff >= 5 {
            println!(
                "   ⚠️  {}: λ₂ rank {} vs Φ rank {} (diff: {})",
                name, l2_rank, phi_rank, diff
            );
        }
    }

    let avg_rank_diff = rank_diff_sum as f64 / results.len() as f64;
    println!("\n   Average rank difference: {:.2}", avg_rank_diff);

    // Key topology check: Star vs Random
    println!("\n🔑 KEY DIAGNOSTIC - Star vs Random:");
    let star_l2 = results
        .iter()
        .find(|r| r.0 == "Star")
        .map(|r| r.1)
        .unwrap_or(0.0);
    let star_phi = results
        .iter()
        .find(|r| r.0 == "Star")
        .map(|r| r.3)
        .unwrap_or(0.0);
    let random_l2 = results
        .iter()
        .find(|r| r.0 == "Random")
        .map(|r| r.1)
        .unwrap_or(0.0);
    let random_phi = results
        .iter()
        .find(|r| r.0 == "Random")
        .map(|r| r.3)
        .unwrap_or(0.0);

    println!("   Star:   λ₂ = {:.4}, Φ = {:.4}", star_l2, star_phi);
    println!("   Random: λ₂ = {:.4}, Φ = {:.4}", random_l2, random_phi);

    if star_l2 < random_l2 && star_phi > random_phi {
        println!("   ❌ OPPOSITE ORDERING - metrics fundamentally differ");
        println!("      λ₂: Random > Star (spectral prefers uniform degree)");
        println!("      Φ:  Star > Random (IIT prefers hub integration)");
    } else if star_l2 > random_l2 && star_phi > random_phi {
        println!("   ✅ SAME ORDERING - metrics agree on Star > Random");
    } else {
        println!("   ⚠️  MIXED RESULTS - need further analysis");
    }

    println!("\n{}", "═".repeat(80));
    println!("                         ANALYSIS COMPLETE");
    println!("{}", "═".repeat(80));
    println!("\nResults saved. Use these findings to update the paper.\n");
}

/// Measure both metrics for a topology
fn measure_topology<F>(
    name: &str,
    results: &mut Vec<(String, f64, f64, f64, f64)>,
    n_nodes: usize,
    n_samples: usize,
    generator: F,
) where
    F: Fn(u64) -> ConsciousnessTopology,
{
    print!("   Testing {:<20}", name);

    let real_calc = ConnectivityCalculator::new();
    let mut l2_values = Vec::new();
    let mut phi_values = Vec::new();

    for seed in 0..n_samples as u64 {
        let topology = generator(seed);

        // λ₂ (Spectral / Algebraic Connectivity)
        let l2 = real_calc.algebraic_connectivity(&topology.node_representations);
        l2_values.push(l2);

        // Φ (IIT-inspired via binary conversion)
        let binary_components: Vec<_> = topology
            .node_representations
            .iter()
            .map(|real_hv| real_hv_to_hv16_probabilistic(real_hv, seed))
            .collect();

        let mut phi_calc = IntegratedInformation::new();
        let phi = phi_calc.compute_phi(&binary_components);
        phi_values.push(phi);
    }

    // Statistics
    let l2_mean = l2_values.iter().sum::<f64>() / l2_values.len() as f64;
    let l2_std = std_dev(&l2_values, l2_mean);

    let phi_mean = phi_values.iter().sum::<f64>() / phi_values.len() as f64;
    let phi_std = std_dev(&phi_values, phi_mean);

    println!(
        "λ₂={:.4}±{:.3}  Φ={:.4}±{:.3}",
        l2_mean, l2_std, phi_mean, phi_std
    );

    results.push((name.to_string(), l2_mean, l2_std, phi_mean, phi_std));
}

fn std_dev(values: &[f64], mean: f64) -> f64 {
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
    variance.sqrt()
}

fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let x_mean = x.iter().sum::<f64>() / n;
    let y_mean = y.iter().sum::<f64>() / n;

    let mut numerator = 0.0;
    let mut x_var = 0.0;
    let mut y_var = 0.0;

    for i in 0..x.len() {
        let x_diff = x[i] - x_mean;
        let y_diff = y[i] - y_mean;
        numerator += x_diff * y_diff;
        x_var += x_diff.powi(2);
        y_var += y_diff.powi(2);
    }

    if x_var == 0.0 || y_var == 0.0 {
        return 0.0;
    }

    numerator / (x_var.sqrt() * y_var.sqrt())
}

fn spearman_correlation(x: &[f64], y: &[f64]) -> f64 {
    // Convert to ranks
    let x_ranks = to_ranks(x);
    let y_ranks = to_ranks(y);

    // Pearson on ranks = Spearman
    pearson_correlation(&x_ranks, &y_ranks)
}

fn to_ranks(values: &[f64]) -> Vec<f64> {
    let mut indexed: Vec<(usize, f64)> = values.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut ranks = vec![0.0; values.len()];
    for (rank, (original_idx, _)) in indexed.iter().enumerate() {
        ranks[*original_idx] = (rank + 1) as f64;
    }
    ranks
}
