// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Debug example to compare eigenvalue calculations
// Run with: cargo run --example debug_eigenvalues --release

use nalgebra::DMatrix;
use symthaea::hdc::{
    HDC_DIMENSION, consciousness_topology_generators::ConsciousnessTopology,
    spectral_connectivity::ConnectivityCalculator,
};

fn main() {
    println!("\n🔬 DEBUG: Eigenvalue Calculation Analysis");
    println!("============================================================\n");

    let calc = ConnectivityCalculator::new();

    // Test with Star and Random topologies
    let n_nodes = 8;
    let dim = HDC_DIMENSION;
    let seed = 42u64;

    println!("Testing with {} nodes, {} dimensions\n", n_nodes, dim);

    // Generate topologies
    let star = ConsciousnessTopology::star(n_nodes, dim, seed);
    let random = ConsciousnessTopology::random(n_nodes, dim, seed);

    // Build similarity matrices
    let star_sim = calc.build_similarity_matrix(&star.node_representations);
    let random_sim = calc.build_similarity_matrix(&random.node_representations);

    println!("📊 Star Topology Analysis:");
    analyze_matrix(&star_sim, "Star");

    println!("\n📊 Random Topology Analysis:");
    analyze_matrix(&random_sim, "Random");

    // Compute Φ values
    let phi_star = calc.algebraic_connectivity(&star.node_representations);
    let phi_random = calc.algebraic_connectivity(&random.node_representations);

    println!("\n============================================================");
    println!("🎯 FINAL Φ VALUES:");
    println!("  Star:   Φ = {:.4}", phi_star);
    println!("  Random: Φ = {:.4}", phi_random);
    println!(
        "  Δ = {:.4} ({:+.2}%)",
        phi_star - phi_random,
        ((phi_star - phi_random) / phi_random) * 100.0
    );
    println!("============================================================\n");
}

fn analyze_matrix(similarity_matrix: &[Vec<f64>], _name: &str) {
    let n = similarity_matrix.len();

    // Print similarity matrix stats
    let mut min_sim = f64::MAX;
    let mut max_sim = f64::MIN;
    let mut sum_sim = 0.0;
    let mut off_diag_count = 0;

    for (i, row) in similarity_matrix.iter().enumerate().take(n) {
        for (j, &v) in row.iter().enumerate().take(n) {
            if i != j {
                min_sim = min_sim.min(v);
                max_sim = max_sim.max(v);
                sum_sim += v;
                off_diag_count += 1;
            }
        }
    }
    let avg_sim = sum_sim / off_diag_count as f64;

    println!("  Similarity Matrix Stats:");
    println!("    Min off-diagonal: {:.4}", min_sim);
    println!("    Max off-diagonal: {:.4}", max_sim);
    println!("    Avg off-diagonal: {:.4}", avg_sim);

    // Compute degrees
    let mut degrees: Vec<f64> = vec![0.0; n];
    for (i, degree) in degrees.iter_mut().enumerate().take(n) {
        for (j, &val) in similarity_matrix[i].iter().enumerate().take(n) {
            if i != j {
                *degree += val;
            }
        }
    }

    let min_deg = degrees.iter().cloned().fold(f64::MAX, f64::min);
    let max_deg = degrees.iter().cloned().fold(f64::MIN, f64::max);
    let avg_deg: f64 = degrees.iter().sum::<f64>() / n as f64;

    println!("  Degree Stats:");
    println!("    Min degree: {:.4}", min_deg);
    println!("    Max degree: {:.4}", max_deg);
    println!("    Avg degree: {:.4}", avg_deg);

    // Build and analyze normalized Laplacian
    let inv_sqrt_degrees: Vec<f64> = degrees
        .iter()
        .map(|&d| if d > 1e-10 { 1.0 / d.sqrt() } else { 0.0 })
        .collect();

    let mut laplacian_data = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let idx = i * n + j;
            if i == j {
                laplacian_data[idx] = if degrees[i] > 1e-10 { 1.0 } else { 0.0 };
            } else {
                let normalization = inv_sqrt_degrees[i] * inv_sqrt_degrees[j];
                laplacian_data[idx] = -similarity_matrix[i][j] * normalization;
            }
        }
    }

    // Compute eigenvalues using nalgebra
    let laplacian = DMatrix::from_row_slice(n, n, &laplacian_data);
    let eigen = nalgebra::SymmetricEigen::new(laplacian);

    let mut eigenvalues: Vec<f64> = eigen.eigenvalues.iter().cloned().collect();
    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    println!("  Normalized Laplacian Eigenvalues (sorted):");
    for (i, ev) in eigenvalues.iter().enumerate() {
        let label = if i == 0 {
            "λ₁ (should be ~0)"
        } else if i == 1 {
            "λ₂ (algebraic connectivity)"
        } else {
            ""
        };
        println!("    λ_{} = {:.6}  {}", i + 1, ev, label);
    }

    println!(
        "  → Algebraic Connectivity (λ₂): {:.6}",
        eigenvalues[1].max(0.0)
    );
    println!(
        "  → Normalized Φ (λ₂/2): {:.6}",
        (eigenvalues[1].max(0.0) / 2.0).clamp(0.0, 1.0)
    );
}