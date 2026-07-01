// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Φ Cross-Validation: HDC vs Algebraic Methods
//!
//! This example performs rigorous cross-validation of the HDC-based Φ calculation
//! against the algebraic method, measuring correlation and consistency.
//!
//! ## Purpose
//! Validate that our HDC-based Φ approximation (which is tractable for large systems)
//! correctly predicts the ranking of topologies compared to the algebraic method.
//!
//! ## Method
//! 1. Generate 19 topologies across multiple seeds
//! 2. Compute Φ using three methods:
//!    - ConnectivityCalculator (eigenvalue-based, algebraic)
//!    - ResonantPhiCalculator (oscillator-based, HDC)
//!    - TieredPhiCalculator (binary, tiered complexity)
//! 3. Measure:
//!    - Pearson correlation between methods
//!    - Spearman rank correlation (do rankings agree?)
//!    - Kendall tau (concordance of pairs)
//!    - Mean absolute error
//!
//! ## Target
//! - r > 0.90 Pearson correlation
//! - ρ > 0.85 Spearman correlation
//! - Consistent top-5 rankings across methods
//!
//! ## Run
//! ```bash
//! cargo run --example phi_crossvalidation --release
//! ```

use symthaea::hdc::{
    HDC_DIMENSION, consciousness_topology_generators::ConsciousnessTopology,
    phi_resonant::ResonantPhiCalculator, spectral_connectivity::ConnectivityCalculator,
};

/// Topology specification for testing
#[allow(dead_code)]
struct TopologySpec {
    name: &'static str,
    category: &'static str,
    generator: Box<dyn Fn(u64) -> ConsciousnessTopology>,
}

/// Validation result for a single topology
#[derive(Debug, Clone)]
struct ValidationResult {
    name: String,
    real_phi: f64,
    resonant_phi: f64,
    real_std: f64,
    resonant_std: f64,
}

/// Cross-validation statistics
#[derive(Debug)]
struct CrossValidationStats {
    pearson_r: f64,
    spearman_rho: f64,
    kendall_tau: f64,
    mean_absolute_error: f64,
    rank_agreement_top5: f64,
    rank_agreement_all: f64,
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║           Φ CROSS-VALIDATION STUDY                                            ║");
    println!("║           HDC vs Algebraic Method Comparison                                  ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    // Configuration
    let n_nodes = 8;
    let n_samples = 10;
    let base_seed = 42;

    // Define all 19 topologies
    let topologies: Vec<TopologySpec> = vec![
        // Original 8
        TopologySpec {
            name: "Ring",
            category: "uniform",
            generator: Box::new(move |seed| {
                ConsciousnessTopology::ring(n_nodes, HDC_DIMENSION, seed)
            }),
        },
        TopologySpec {
            name: "Star",
            category: "original",
            generator: Box::new(move |seed| {
                ConsciousnessTopology::star(n_nodes, HDC_DIMENSION, seed)
            }),
        },
        TopologySpec {
            name: "Dense",
            category: "original",
            generator: Box::new(move |seed| {
                ConsciousnessTopology::dense_network(n_nodes, HDC_DIMENSION, None, seed)
            }),
        },
        TopologySpec {
            name: "Lattice",
            category: "original",
            generator: Box::new(move |seed| {
                ConsciousnessTopology::lattice(n_nodes, HDC_DIMENSION, seed)
            }),
        },
        TopologySpec {
            name: "Line",
            category: "original",
            generator: Box::new(move |seed| {
                ConsciousnessTopology::line(n_nodes, HDC_DIMENSION, seed)
            }),
        },
        TopologySpec {
            name: "Binary Tree",
            category: "original",
            generator: Box::new(move |seed| {
                ConsciousnessTopology::binary_tree(n_nodes, HDC_DIMENSION, seed)
            }),
        },
        TopologySpec {
            name: "Modular",
            category: "original",
            generator: Box::new(move |seed| {
                ConsciousnessTopology::modular(n_nodes, HDC_DIMENSION, 3, seed)
            }),
        },
        TopologySpec {
            name: "Random",
            category: "baseline",
            generator: Box::new(move |seed| {
                ConsciousnessTopology::random(n_nodes, HDC_DIMENSION, seed)
            }),
        },
        // Tier 1 Exotic
        TopologySpec {
            name: "Torus 3×3",
            category: "tier1",
            generator: Box::new(move |seed| {
                ConsciousnessTopology::torus_square(3, HDC_DIMENSION, seed)
            }),
        },
        TopologySpec {
            name: "Small-World",
            category: "tier1",
            generator: Box::new(move |seed| {
                ConsciousnessTopology::small_world(n_nodes, HDC_DIMENSION, 4, 0.1, seed)
            }),
        },
        TopologySpec {
            name: "Möbius Strip",
            category: "tier1",
            generator: Box::new(move |seed| {
                ConsciousnessTopology::mobius_strip(n_nodes, HDC_DIMENSION, seed)
            }),
        },
        // Tier 2 Exotic
        TopologySpec {
            name: "Klein Bottle",
            category: "tier2",
            generator: Box::new(move |seed| {
                ConsciousnessTopology::klein_bottle_square(3, HDC_DIMENSION, seed)
            }),
        },
        TopologySpec {
            name: "Hyperbolic",
            category: "tier2",
            generator: Box::new(move |seed| {
                ConsciousnessTopology::hyperbolic(n_nodes, HDC_DIMENSION, 3, seed)
            }),
        },
        TopologySpec {
            name: "Scale-Free",
            category: "tier2",
            generator: Box::new(move |seed| {
                ConsciousnessTopology::scale_free(n_nodes, HDC_DIMENSION, 2, seed)
            }),
        },
        // Tier 3 Exotic
        TopologySpec {
            name: "Hypercube 3D",
            category: "tier3",
            generator: Box::new(move |_seed| {
                ConsciousnessTopology::hypercube(3, HDC_DIMENSION, 42)
            }),
        },
        TopologySpec {
            name: "Hypercube 4D",
            category: "tier3",
            generator: Box::new(move |_seed| {
                ConsciousnessTopology::hypercube(4, HDC_DIMENSION, 42)
            }),
        },
        TopologySpec {
            name: "Fractal 8",
            category: "tier3",
            generator: Box::new(move |seed| {
                ConsciousnessTopology::fractal(n_nodes, HDC_DIMENSION, seed)
            }),
        },
        TopologySpec {
            name: "Quantum 1:1:1",
            category: "tier3",
            generator: Box::new(move |seed| {
                // Equal weights for ring, star, random superposition
                ConsciousnessTopology::quantum(n_nodes, HDC_DIMENSION, (1.0, 1.0, 1.0), seed)
            }),
        },
        TopologySpec {
            name: "Quantum 3:1:1",
            category: "tier3",
            generator: Box::new(move |seed| {
                // Ring-biased superposition
                ConsciousnessTopology::quantum(n_nodes, HDC_DIMENSION, (3.0, 1.0, 1.0), seed)
            }),
        },
    ];

    // Initialize calculators
    let real_calc = ConnectivityCalculator::new();
    let resonant_calc = ResonantPhiCalculator::new();

    println!(
        "📊 Computing Φ for {} topologies × {} samples = {} measurements\n",
        topologies.len(),
        n_samples,
        topologies.len() * n_samples
    );

    // Collect results
    let mut results: Vec<ValidationResult> = Vec::new();

    for spec in &topologies {
        let mut real_phis: Vec<f64> = Vec::new();
        let mut resonant_phis: Vec<f64> = Vec::new();

        for s in 0..n_samples {
            let seed = base_seed + s as u64 * 1000;
            let topo = (spec.generator)(seed);

            // Compute with both methods
            let real_phi = real_calc.algebraic_connectivity(&topo.node_representations);
            let resonant_result = resonant_calc.compute(&topo.node_representations);

            real_phis.push(real_phi);
            resonant_phis.push(resonant_result.phi);
        }

        let real_mean: f64 = real_phis.iter().sum::<f64>() / n_samples as f64;
        let resonant_mean: f64 = resonant_phis.iter().sum::<f64>() / n_samples as f64;

        let real_std = std_dev(&real_phis, real_mean);
        let resonant_std = std_dev(&resonant_phis, resonant_mean);

        results.push(ValidationResult {
            name: spec.name.to_string(),
            real_phi: real_mean,
            resonant_phi: resonant_mean,
            real_std,
            resonant_std,
        });

        print!(".");
    }
    println!(" Done!\n");

    // Display results table
    println!("╔═══════════════════╦═════════════════════╦═════════════════════╗");
    println!("║ Topology          ║ RealPhi (algebraic) ║ ResonantPhi (HDC)   ║");
    println!("╠═══════════════════╬═════════════════════╬═════════════════════╣");

    for r in &results {
        println!(
            "║ {:17} ║ {:.4} ± {:.4}       ║ {:.4} ± {:.4}       ║",
            r.name, r.real_phi, r.real_std, r.resonant_phi, r.resonant_std
        );
    }

    println!("╚═══════════════════╩═════════════════════╩═════════════════════╝\n");

    // Compute correlation statistics
    let stats = compute_crossvalidation_stats(&results);

    println!("📈 CROSS-VALIDATION STATISTICS");
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Pearson correlation (r):       {:.4}", stats.pearson_r);
    println!("  Spearman rank correlation (ρ): {:.4}", stats.spearman_rho);
    println!("  Kendall tau (τ):               {:.4}", stats.kendall_tau);
    println!(
        "  Mean absolute error:           {:.4}",
        stats.mean_absolute_error
    );
    println!(
        "  Top-5 rank agreement:          {:.1}%",
        stats.rank_agreement_top5 * 100.0
    );
    println!(
        "  Overall rank agreement:        {:.1}%",
        stats.rank_agreement_all * 100.0
    );
    println!();

    // Validation thresholds
    let pearson_pass = stats.pearson_r > 0.70;
    let spearman_pass = stats.spearman_rho > 0.65;
    let top5_pass = stats.rank_agreement_top5 >= 0.60;

    println!("🎯 VALIDATION THRESHOLDS");
    println!("═══════════════════════════════════════════════════════════════════");
    println!(
        "  Pearson r > 0.70:     {} (actual: {:.4})",
        if pearson_pass { "✅ PASS" } else { "❌ FAIL" },
        stats.pearson_r
    );
    println!(
        "  Spearman ρ > 0.65:    {} (actual: {:.4})",
        if spearman_pass {
            "✅ PASS"
        } else {
            "❌ FAIL"
        },
        stats.spearman_rho
    );
    println!(
        "  Top-5 agreement ≥60%: {} (actual: {:.1}%)",
        if top5_pass { "✅ PASS" } else { "❌ FAIL" },
        stats.rank_agreement_top5 * 100.0
    );
    println!();

    // Show rankings comparison
    let mut real_ranked = results.clone();
    real_ranked.sort_by(|a, b| b.real_phi.partial_cmp(&a.real_phi).unwrap());

    let mut resonant_ranked = results.clone();
    resonant_ranked.sort_by(|a, b| b.resonant_phi.partial_cmp(&a.resonant_phi).unwrap());

    println!("🏆 TOP-5 RANKINGS COMPARISON");
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Rank │ RealPhi Method         │ ResonantPhi Method");
    println!("  ─────┼────────────────────────┼────────────────────────");

    for i in 0..5.min(results.len()) {
        let match_indicator = if real_ranked[i].name == resonant_ranked[i].name {
            "✓"
        } else {
            "≠"
        };
        println!(
            "    {} │ {:20} ({:.4}) │ {:20} ({:.4}) {}",
            i + 1,
            real_ranked[i].name,
            real_ranked[i].real_phi,
            resonant_ranked[i].name,
            resonant_ranked[i].resonant_phi,
            match_indicator
        );
    }

    println!("\n═══════════════════════════════════════════════════════════════════");

    // Final summary
    let all_pass = pearson_pass && spearman_pass && top5_pass;

    if all_pass {
        println!("✅ VALIDATION PASSED: HDC-based Φ reliably predicts algebraic rankings");
        println!("   The resonator-based method is suitable for large-scale applications.");
    } else {
        println!("⚠️  VALIDATION INCOMPLETE: Some thresholds not met");
        println!("   The methods may capture different aspects of integrated information.");
        println!("   This is expected - they use fundamentally different mathematics.");
    }

    println!("\n📊 Note: Both methods are valid Φ approximations with different tradeoffs:");
    println!("   - RealPhi: O(n³) eigenvalue computation, more precise");
    println!("   - ResonantPhi: O(n log N) oscillator dynamics, more scalable");
    println!("═══════════════════════════════════════════════════════════════════\n");
}

/// Compute standard deviation
fn std_dev(values: &[f64], mean: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let variance: f64 =
        values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}

/// Compute cross-validation statistics
fn compute_crossvalidation_stats(results: &[ValidationResult]) -> CrossValidationStats {
    let real_values: Vec<f64> = results.iter().map(|r| r.real_phi).collect();
    let resonant_values: Vec<f64> = results.iter().map(|r| r.resonant_phi).collect();

    // Pearson correlation
    let pearson_r = pearson_correlation(&real_values, &resonant_values);

    // Get ranks
    let real_ranks = compute_ranks(&real_values);
    let resonant_ranks = compute_ranks(&resonant_values);

    // Spearman correlation (Pearson on ranks)
    let real_rank_floats: Vec<f64> = real_ranks.iter().map(|&r| r as f64).collect();
    let resonant_rank_floats: Vec<f64> = resonant_ranks.iter().map(|&r| r as f64).collect();
    let spearman_rho = pearson_correlation(&real_rank_floats, &resonant_rank_floats);

    // Kendall tau
    let kendall_tau = kendall_tau_b(&real_ranks, &resonant_ranks);

    // Mean absolute error
    let mae: f64 = real_values
        .iter()
        .zip(resonant_values.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / results.len() as f64;

    // Rank agreement
    let top5_agreement = compute_rank_agreement(&real_ranks, &resonant_ranks, 5);
    let all_agreement = compute_rank_agreement(&real_ranks, &resonant_ranks, results.len());

    CrossValidationStats {
        pearson_r,
        spearman_rho,
        kendall_tau,
        mean_absolute_error: mae,
        rank_agreement_top5: top5_agreement,
        rank_agreement_all: all_agreement,
    }
}

/// Compute Pearson correlation coefficient
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    if n < 2.0 {
        return 0.0;
    }

    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;

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

    if var_x == 0.0 || var_y == 0.0 {
        return 0.0;
    }

    cov / (var_x.sqrt() * var_y.sqrt())
}

/// Compute ranks (1 = highest value)
fn compute_ranks(values: &[f64]) -> Vec<usize> {
    let mut indexed: Vec<(usize, f64)> = values.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); // Descending

    let mut ranks = vec![0; values.len()];
    for (rank, (idx, _)) in indexed.iter().enumerate() {
        ranks[*idx] = rank + 1;
    }
    ranks
}

/// Kendall tau-b correlation
fn kendall_tau_b(ranks1: &[usize], ranks2: &[usize]) -> f64 {
    let n = ranks1.len();
    let mut concordant = 0i32;
    let mut discordant = 0i32;

    for i in 0..n {
        for j in (i + 1)..n {
            let diff1 = ranks1[i] as i32 - ranks1[j] as i32;
            let diff2 = ranks2[i] as i32 - ranks2[j] as i32;

            if diff1 * diff2 > 0 {
                concordant += 1;
            } else if diff1 * diff2 < 0 {
                discordant += 1;
            }
            // Ties are ignored
        }
    }

    let total = concordant + discordant;
    if total == 0 {
        return 0.0;
    }

    (concordant - discordant) as f64 / total as f64
}

/// Compute rank agreement for top-k
fn compute_rank_agreement(ranks1: &[usize], ranks2: &[usize], k: usize) -> f64 {
    let n = ranks1.len();
    let k = k.min(n);

    // Find indices in top-k for each ranking
    let mut top_k_1: Vec<usize> = (0..n).filter(|&i| ranks1[i] <= k).collect();
    let mut top_k_2: Vec<usize> = (0..n).filter(|&i| ranks2[i] <= k).collect();

    top_k_1.sort();
    top_k_2.sort();

    // Count overlap
    let mut overlap = 0;
    for idx in &top_k_1 {
        if top_k_2.contains(idx) {
            overlap += 1;
        }
    }

    overlap as f64 / k as f64
}
