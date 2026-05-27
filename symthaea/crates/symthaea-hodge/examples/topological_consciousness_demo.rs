// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Direction J -- Topological Consciousness Geometry via Hodge Decomposition
//!
//! Demonstrates how Hodge decomposition reveals the topological structure of
//! consciousness by comparing four connectivity patterns:
//!   1. Feedforward (hierarchical chain)
//!   2. Recurrent (cycle with feedback)
//!   3. Modular (two dense clusters, sparse bridge)
//!   4. Fully integrated (complete graph)
//!
//! Key insight: harmonic modes (kernel of the Hodge Laplacian) represent
//! irreducible integration that cannot be decomposed into gradient (feedforward)
//! or curl (feedback) components -- analogous to Phi in IIT.

use symthaea_hodge::consciousness_topology::{
    ConsciousnessComplex, ConsciousnessManifold, HodgeConsciousnessDecomposition, TopologicalPhi,
};

/// Build an NxN correlation matrix from a list of directed edges.
/// Each edge (i, j, weight) sets matrix[i][j] = matrix[j][i] = weight.
/// Diagonal is 1.0 (self-correlation).
fn build_matrix(n: usize, edges: &[(usize, usize, f64)]) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; n]; n];
    for i in 0..n {
        m[i][i] = 1.0;
    }
    for &(i, j, w) in edges {
        m[i][j] = w;
        m[j][i] = w;
    }
    m
}

struct PatternResult {
    name: &'static str,
    description: &'static str,
    num_edges: usize,
    num_triangles: usize,
    betti: Vec<usize>,
    euler: i64,
    gradient_pct: f64,
    curl_pct: f64,
    harmonic_pct: f64,
    phi: TopologicalPhi,
    manifold: Option<ConsciousnessManifold>,
}

fn analyze_pattern(
    name: &'static str,
    description: &'static str,
    matrix: &[Vec<f64>],
    threshold: f64,
) -> PatternResult {
    let complex = ConsciousnessComplex::from_correlation_matrix(matrix, threshold);
    let num_edges = complex.edges.len();
    let num_triangles = complex.triangles.len();
    let betti = complex.betti_numbers();

    // Create a uniform edge signal (equal flow on all edges)
    let signal: Vec<f64> = vec![1.0; num_edges];
    let decomp = if num_edges > 0 {
        HodgeConsciousnessDecomposition::decompose(&complex, &signal)
    } else {
        HodgeConsciousnessDecomposition::decompose(&complex, &[])
    };

    let phi = TopologicalPhi::from_decomposition(&decomp);

    // Compute energy fractions
    let grad_energy: f64 = decomp.gradient_flow.iter().map(|x| x * x).sum();
    let curl_energy: f64 = decomp.curl_flow.iter().map(|x| x * x).sum();
    let harm_energy: f64 = decomp.harmonic_flow.iter().map(|x| x * x).sum();
    let total = grad_energy + curl_energy + harm_energy;
    let (gradient_pct, curl_pct, harmonic_pct) = if total > 1e-12 {
        (
            grad_energy / total * 100.0,
            curl_energy / total * 100.0,
            harm_energy / total * 100.0,
        )
    } else {
        (0.0, 0.0, 0.0)
    };

    // Generate synthetic state window for manifold estimation
    let n = matrix.len();
    let states: Vec<Vec<f64>> = (0..20)
        .map(|t| {
            let phase = t as f64 * 0.15;
            (0..n)
                .map(|i| {
                    let base = (phase + i as f64 * 0.3).sin();
                    let modulation = if num_triangles > 0 {
                        0.2 * (phase * 2.0 + i as f64).cos()
                    } else {
                        0.05 * (phase + i as f64).cos()
                    };
                    base + modulation
                })
                .collect()
        })
        .collect();
    let manifold = Some(ConsciousnessManifold::from_state_window(&states));

    PatternResult {
        name,
        description,
        num_edges,
        num_triangles,
        betti,
        euler: decomp.euler_characteristic,
        gradient_pct,
        curl_pct,
        harmonic_pct,
        phi,
        manifold,
    }
}

fn main() {
    println!();
    println!(
        "\u{2554}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2557}"
    );
    println!("\u{2551}  Topological Consciousness Geometry Demo              \u{2551}");
    println!(
        "\u{2560}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2563}"
    );
    println!("\u{2551}  Hodge decomposition reveals consciousness topology   \u{2551}");
    println!(
        "\u{255a}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{255d}"
    );
    println!();

    let n = 8;
    let threshold = 0.3;

    // -- Pattern 1: Feedforward chain (0-1-2-3-4-5-6-7) --
    let ff_edges: Vec<(usize, usize, f64)> = (0..n - 1).map(|i| (i, i + 1, 0.9)).collect();
    let ff_matrix = build_matrix(n, &ff_edges);

    // -- Pattern 2: Recurrent cycle (0-1-2-...-7-0) --
    let mut cyc_edges: Vec<(usize, usize, f64)> = (0..n - 1).map(|i| (i, i + 1, 0.9)).collect();
    cyc_edges.push((0, n - 1, 0.9)); // close the cycle
    let cyc_matrix = build_matrix(n, &cyc_edges);

    // -- Pattern 3: Modular (two clusters of 4, sparse bridge) --
    let mut mod_edges = Vec::new();
    // Cluster A: nodes 0-3, dense
    for i in 0..4 {
        for j in (i + 1)..4 {
            mod_edges.push((i, j, 0.85));
        }
    }
    // Cluster B: nodes 4-7, dense
    for i in 4..8 {
        for j in (i + 1)..8 {
            mod_edges.push((i, j, 0.85));
        }
    }
    // Sparse bridge between clusters
    mod_edges.push((3, 4, 0.5));
    let mod_matrix = build_matrix(n, &mod_edges);

    // -- Pattern 4: Fully integrated (complete graph K8) --
    let mut full_edges = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            full_edges.push((i, j, 0.8));
        }
    }
    let full_matrix = build_matrix(n, &full_edges);

    let results = vec![
        analyze_pattern(
            "Feedforward (Hierarchical Chain)",
            "8 nodes in a chain, no feedback",
            &ff_matrix,
            threshold,
        ),
        analyze_pattern(
            "Recurrent (Cycle)",
            "8 nodes in a closed loop with feedback",
            &cyc_matrix,
            threshold,
        ),
        analyze_pattern(
            "Modular (Two Clusters)",
            "4+4 nodes, dense within, sparse bridge",
            &mod_matrix,
            threshold,
        ),
        analyze_pattern(
            "Fully Integrated (Complete K8)",
            "8 fully connected nodes",
            &full_matrix,
            threshold,
        ),
    ];

    // Print individual results
    for (i, r) in results.iter().enumerate() {
        println!(
            "\u{2550}\u{2550}\u{2550} Pattern {}: {} \u{2550}\u{2550}\u{2550}",
            i + 1,
            r.name
        );
        println!("  {}", r.description);
        println!("  Edges: {} | Triangles: {}", r.num_edges, r.num_triangles);
        println!(
            "  Betti numbers: [{}, {}, {}]",
            r.betti.first().unwrap_or(&0),
            r.betti.get(1).unwrap_or(&0),
            r.betti.get(2).unwrap_or(&0)
        );
        println!("  Euler characteristic: {}", r.euler);
        println!("  Hodge Decomposition:");
        println!("    Gradient (feedforward): {:5.1}%", r.gradient_pct);
        println!("    Curl (feedback):        {:5.1}%", r.curl_pct);
        println!("    Harmonic (integrated):  {:5.1}%", r.harmonic_pct);
        println!("  Topological Phi:");
        println!("    Harmonic:  {:.3}", r.phi.phi_harmonic);
        println!("    Betti:     {:.3}", r.phi.phi_betti);
        println!("    Combined:  {:.3}", r.phi.phi_topological);
        println!(
            "    Loops: {} | Higher-order voids: {}",
            r.phi.integration_loops, r.phi.higher_order_voids
        );
        if let Some(ref m) = r.manifold {
            println!("  Consciousness Manifold:");
            println!("    Intrinsic dimension: {}", m.dimension);
            println!("    Curvature scalar:    {:.4}", m.curvature_scalar);
            println!(
                "    Geodesic to attractor: {:.4}",
                m.geodesic_distance_to_attractor
            );
            println!("    Volume element:      {:.6}", m.volume_element);
        }
        println!();
    }

    // Comparison table
    println!("\u{2550}\u{2550}\u{2550} Comparison \u{2550}\u{2550}\u{2550}");
    println!(
        "{:<28} | {:>8} | {:>10} | {:>10} | {:>9}",
        "Pattern", "Phi_topo", "\u{03b2}\u{2081} (loops)", "\u{03b2}\u{2082} (voids)", "Harmonic%"
    );
    println!(
        "{:-<28}-+-{:-<8}-+-{:-<10}-+-{:-<10}-+-{:-<9}",
        "", "", "", "", ""
    );
    for r in &results {
        println!(
            "{:<28} | {:>8.3} | {:>10} | {:>10} | {:>8.1}%",
            r.name,
            r.phi.phi_topological,
            r.betti.get(1).unwrap_or(&0),
            r.betti.get(2).unwrap_or(&0),
            r.harmonic_pct,
        );
    }

    println!();
    println!("Insight: Topological Phi captures irreducible integration --");
    println!("harmonic modes that survive neither gradient nor curl decomposition.");
    println!("This is consistent with IIT's prediction that consciousness requires");
    println!("information integration beyond what any partition can account for.");
    println!();
}
