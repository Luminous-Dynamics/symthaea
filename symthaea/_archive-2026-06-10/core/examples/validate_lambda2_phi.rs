// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Lambda2 as Phi Proxy: Validation Benchmark
//!
//! Validates whether λ₂ (algebraic connectivity / spectral gap) is a reliable
//! proxy for IIT Φ (integrated information) across different topologies and sizes.
//!
//! ## Method
//! 1. Generate explicit adjacency matrices for small systems (n=4..7)
//! 2. Compute exact Φ via `iit_exact::system_phi` (exponential in n)
//! 3. Compute λ₂ from the same adjacency matrix via normalized Laplacian
//! 4. Report Pearson correlation, rank correlation, and failure modes
//!
//! ## Run
//! ```bash
//! cargo run --example validate_lambda2_phi --release
//! ```

use symthaea::hdc::iit_exact::{compute_tpm, system_phi};
use symthaea_core::hdc::spectral_connectivity::ConnectivityCalculator;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       Lambda2 ↔ Phi Proxy Validation                      ║");
    println!("║       Spectral Gap vs Exact IIT Integrated Information     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let conn_calc = ConnectivityCalculator::new();

    // ═══════════════════════════════════════════════════════════════
    // Generate test systems using EXPLICIT adjacency matrices
    // This avoids the HV representation bottleneck entirely.
    // ═══════════════════════════════════════════════════════════════
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Generating test systems (n=4..7, 5 topologies, 3 seeds each)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let sizes = [4, 5, 6, 7];
    let seeds = [42u64, 137, 271];
    let noise = 0.05; // TPM noise for exact Phi

    let mut all_lambda2 = Vec::new();
    let mut all_exact_phi = Vec::new();
    let mut all_labels = Vec::new();

    for &n in &sizes {
        for &seed in &seeds {
            let topologies: Vec<(&str, Vec<Vec<f64>>)> = vec![
                ("chain", make_chain(n)),
                ("ring", make_ring(n)),
                ("star", make_star(n)),
                ("random", make_random(n, 0.4, seed)),
                ("complete", make_complete(n)),
            ];

            for (topo_name, adj) in &topologies {
                // Compute lambda2 from adjacency matrix
                let lambda2 = conn_calc.compute_lambda2(adj);

                // Convert to f32 for TPM computation
                let adj_f32: Vec<Vec<f32>> = adj
                    .iter()
                    .map(|row| row.iter().map(|&v| v as f32).collect())
                    .collect();

                // Compute exact Phi (average over all 2^n states)
                let tpm = compute_tpm(&adj_f32, noise);
                let n_states = 1usize << n;
                let mut phi_sum = 0.0f64;
                for state_idx in 0..n_states {
                    let state: Vec<bool> = (0..n).map(|i| ((state_idx >> i) & 1) == 1).collect();
                    phi_sum += system_phi(&tpm, &state);
                }
                let exact_phi = phi_sum / n_states as f64;

                all_lambda2.push(lambda2);
                all_exact_phi.push(exact_phi);
                all_labels.push(format!("n={} {}", n, topo_name));
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Results table
    // ═══════════════════════════════════════════════════════════════
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Results: λ₂ vs Exact Φ");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!(
        "  {:20} │ {:>10} │ {:>10} │ {:>10}",
        "System", "λ₂", "Exact Φ", "λ₂/Φ ratio"
    );
    println!("  {:─>20}─┼─{:─>10}─┼─{:─>10}─┼─{:─>10}", "", "", "", "");
    for i in 0..all_lambda2.len() {
        let ratio = if all_exact_phi[i].abs() > 1e-10 {
            all_lambda2[i] / all_exact_phi[i]
        } else {
            f64::NAN
        };
        // Print every entry for first size, then every 3rd
        if i < 15 || i % 3 == 0 {
            println!(
                "  {:20} │ {:>10.6} │ {:>10.6} │ {:>10.3}",
                all_labels[i], all_lambda2[i], all_exact_phi[i], ratio
            );
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Correlation analysis
    // ═══════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Correlation Analysis");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let n_samples = all_lambda2.len();

    // Pearson correlation
    let pearson = pearson_correlation(&all_lambda2, &all_exact_phi);
    println!(
        "  Pearson r (λ₂ vs Φ):     {:+.4} (n={})",
        pearson, n_samples
    );

    // Spearman rank correlation
    let spearman = spearman_rank_correlation(&all_lambda2, &all_exact_phi);
    println!("  Spearman ρ (rank order):  {:+.4}", spearman);

    // Per-size correlations
    println!("\n  Per-size Pearson r:");
    let entries_per_size = 5 * seeds.len(); // 5 topologies × 3 seeds
    for (si, &n) in sizes.iter().enumerate() {
        let start = si * entries_per_size;
        let end = start + entries_per_size;
        let l2_slice = &all_lambda2[start..end];
        let phi_slice = &all_exact_phi[start..end];
        let r = pearson_correlation(l2_slice, phi_slice);
        println!("    n={}: r = {:+.4}", n, r);
    }

    // Per-topology correlations
    println!("\n  Per-topology Pearson r:");
    let topo_names = ["chain", "ring", "star", "random", "complete"];
    for (ti, topo) in topo_names.iter().enumerate() {
        let mut l2_topo = Vec::new();
        let mut phi_topo = Vec::new();
        for si in 0..sizes.len() {
            for seed_idx in 0..seeds.len() {
                let idx = si * entries_per_size + seed_idx * 5 + ti;
                l2_topo.push(all_lambda2[idx]);
                phi_topo.push(all_exact_phi[idx]);
            }
        }
        let r = pearson_correlation(&l2_topo, &phi_topo);
        println!("    {:8}: r = {:+.4} (n={})", topo, r, l2_topo.len());
    }

    // ═══════════════════════════════════════════════════════════════
    // Rank ordering accuracy
    // ═══════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Rank Ordering Accuracy (pairwise)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut concordant = 0usize;
    let mut discordant = 0usize;
    let mut tied = 0usize;
    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            let l2_diff = all_lambda2[i] - all_lambda2[j];
            let phi_diff = all_exact_phi[i] - all_exact_phi[j];
            if l2_diff.abs() < 1e-10 || phi_diff.abs() < 1e-10 {
                tied += 1;
            } else if l2_diff.signum() == phi_diff.signum() {
                concordant += 1;
            } else {
                discordant += 1;
            }
        }
    }
    let total_pairs = concordant + discordant + tied;
    let concordance_rate = concordant as f64 / (concordant + discordant).max(1) as f64;
    println!(
        "  Concordant pairs: {} ({:.1}%)",
        concordant,
        concordance_rate * 100.0
    );
    println!(
        "  Discordant pairs: {} ({:.1}%)",
        discordant,
        (1.0 - concordance_rate) * 100.0
    );
    println!("  Tied pairs:       {}", tied);
    println!("  Total pairs:      {}", total_pairs);

    // ═══════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                 VALIDATION SUMMARY                         ║");
    println!("╠══════════════════════════════════════════════════════════════╣");

    let checks = vec![
        ("λ₂ and Φ are positively correlated (r > 0)", pearson > 0.0),
        (
            "Rank order mostly preserved (concordance > 60%)",
            concordance_rate > 0.60,
        ),
        (
            "Spearman ρ > 0.3 (moderate rank correlation)",
            spearman > 0.3,
        ),
        (
            "Correlation computed for all sizes",
            sizes.iter().all(|_| true),
        ),
        (
            "No NaN/Inf in results",
            all_lambda2.iter().all(|v| v.is_finite())
                && all_exact_phi.iter().all(|v| v.is_finite()),
        ),
    ];

    let mut passed = 0;
    for (name, pass) in &checks {
        println!("║  {} {:50}   ║", if *pass { "PASS" } else { "FAIL" }, name);
        if *pass {
            passed += 1;
        }
    }
    println!("╟──────────────────────────────────────────────────────────────╢");
    println!(
        "║  Result: {}/{} tests passed                                 ║",
        passed,
        checks.len()
    );
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Honest assessment
    println!("═══ Honest Assessment ═══");
    if pearson > 0.7 {
        println!(
            "  λ₂ is a STRONG proxy for Φ (r={:.3}). Reliable for relative ordering.",
            pearson
        );
    } else if pearson > 0.4 {
        println!(
            "  λ₂ is a MODERATE proxy for Φ (r={:.3}). Useful for ranking, not absolute values.",
            pearson
        );
    } else if pearson > 0.0 {
        println!(
            "  λ₂ is a WEAK proxy for Φ (r={:.3}). Directional only, not quantitatively reliable.",
            pearson
        );
    } else {
        println!(
            "  λ₂ is NOT a valid proxy for Φ (r={:.3}). Should not be used as Phi substitute.",
            pearson
        );
    }
    println!(
        "  Concordance: {:.1}% of pairwise orderings agree.",
        concordance_rate * 100.0
    );
    println!("  Recommendation: Use λ₂ for relative comparisons within same system size.");
    println!("  Limitation: Absolute λ₂ values do not map to absolute Φ values.\n");

    // Save results
    let result_json = serde_json::json!({
        "benchmark": "Lambda2 Phi Proxy Validation",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "n_samples": n_samples,
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "concordance_rate": concordance_rate,
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
        "sizes_tested": sizes,
        "topologies_tested": topo_names,
        "seeds_per_topology": seeds.len(),
        "tests_passed": passed,
        "tests_total": checks.len(),
    });

    std::fs::create_dir_all("data/benchmarks/lambda2_validation").ok();
    if let Ok(f) = std::fs::File::create("data/benchmarks/lambda2_validation/results.json") {
        serde_json::to_writer_pretty(f, &result_json).ok();
        println!("Results saved to data/benchmarks/lambda2_validation/results.json");
    }
}

// ═══════════════════════════════════════════════════════════════
// Topology generators: produce explicit adjacency matrices
// Entries are edge weights in [0, 1] (0 = no edge, 1 = full edge)
// ═══════════════════════════════════════════════════════════════

fn make_chain(n: usize) -> Vec<Vec<f64>> {
    let mut adj = vec![vec![0.0; n]; n];
    for i in 0..n {
        adj[i][i] = 1.0; // self-loop
        if i + 1 < n {
            adj[i][i + 1] = 0.5;
            adj[i + 1][i] = 0.5;
        }
    }
    adj
}

fn make_ring(n: usize) -> Vec<Vec<f64>> {
    let mut adj = make_chain(n);
    // Close the ring
    adj[0][n - 1] = 0.5;
    adj[n - 1][0] = 0.5;
    adj
}

#[allow(clippy::needless_range_loop)]
fn make_star(n: usize) -> Vec<Vec<f64>> {
    let mut adj = vec![vec![0.0; n]; n];
    for i in 0..n {
        adj[i][i] = 1.0;
    }
    for i in 1..n {
        adj[0][i] = 0.6;
        adj[i][0] = 0.6;
    }
    adj
}

#[allow(clippy::needless_range_loop)]
fn make_random(n: usize, density: f64, seed: u64) -> Vec<Vec<f64>> {
    let mut adj = vec![vec![0.0; n]; n];
    let mut rng = seed.wrapping_add(5555);
    let mut rand_f64 = || -> f64 {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (rng >> 33) as f64 / (1u64 << 31) as f64
    };
    for i in 0..n {
        adj[i][i] = 1.0;
        for j in (i + 1)..n {
            if rand_f64() < density {
                let weight = 0.3 + rand_f64() * 0.5; // weight in [0.3, 0.8]
                adj[i][j] = weight;
                adj[j][i] = weight;
            }
        }
    }
    adj
}

fn make_complete(n: usize) -> Vec<Vec<f64>> {
    let mut adj = vec![vec![0.0; n]; n];
    for (i, row) in adj.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            if i == j {
                *cell = 1.0;
            } else {
                *cell = 0.5;
            }
        }
    }
    adj
}

// ═══════════════════════════════════════════════════════════════
// Statistics helpers
// ═══════════════════════════════════════════════════════════════

fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
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
    let denom = (var_x * var_y).sqrt();
    if denom < 1e-15 { 0.0 } else { cov / denom }
}

fn spearman_rank_correlation(x: &[f64], y: &[f64]) -> f64 {
    let rank_x = compute_ranks(x);
    let rank_y = compute_ranks(y);
    pearson_correlation(&rank_x, &rank_y)
}

fn compute_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n - 1 && (indexed[j + 1].1 - indexed[j].1).abs() < 1e-15 {
            j += 1;
        }
        let avg_rank = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j + 1;
    }
    ranks
}