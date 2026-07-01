// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Phi Proxy vs. Exact Analysis
//!
//! Computes spectral connectivity (λ₂ proxy) and PhiEngine Phi across
//! multiple topologies, outputs a comparison table, and calculates
//! Spearman rank correlation.
//!
//! This supports the paper's claim that the Phi proxy is informative
//! by showing how well λ₂ rank-orders topologies relative to exact Phi.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example phi_proxy_analysis
//! ```
//!
//! ## Output
//!
//! A table of (topology, λ₂ proxy, PhiEngine Phi, rank_proxy, rank_exact)
//! followed by the Spearman rho correlation coefficient.

use symthaea::hdc::consciousness_topology_generators::ConsciousnessTopology;
use symthaea::hdc::spectral_connectivity::ConnectivityCalculator;
use symthaea::hdc::unified_hv::ContinuousHV;
use symthaea::phi_engine::PhiEngine;

fn to_continuous(topology: &ConsciousnessTopology) -> Vec<ContinuousHV> {
    topology
        .node_representations
        .iter()
        .map(|rhv| ContinuousHV::from_vec(rhv.values.clone()))
        .collect()
}

/// Spearman rank correlation between two slices.
fn spearman_rho(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len() as f64;
    if n < 2.0 {
        return 0.0;
    }

    let rank = |vals: &[f64]| -> Vec<f64> {
        let mut indexed: Vec<(usize, f64)> = vals.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut ranks = vec![0.0; vals.len()];
        for (rank, (orig_idx, _)) in indexed.iter().enumerate() {
            ranks[*orig_idx] = rank as f64 + 1.0;
        }
        ranks
    };

    let rx = rank(x);
    let ry = rank(y);

    let d_sq_sum: f64 = rx.iter().zip(ry.iter()).map(|(a, b)| (a - b).powi(2)).sum();

    1.0 - (6.0 * d_sq_sum) / (n * (n * n - 1.0))
}

fn main() {
    let n_nodes = 8;
    let dim = 512;
    let seed = 42_u64;

    println!("\nPhi Proxy Analysis (n_nodes={n_nodes}, dim={dim})");
    println!("===================================================\n");

    // Build all available Tier 1 + Tier 2 topologies
    let topologies: Vec<(&str, ConsciousnessTopology)> = vec![
        ("Random", ConsciousnessTopology::random(n_nodes, dim, seed)),
        ("Star", ConsciousnessTopology::star(n_nodes, dim, seed)),
        ("Ring", ConsciousnessTopology::ring(n_nodes, dim, seed)),
        ("Line", ConsciousnessTopology::line(n_nodes, dim, seed)),
        (
            "BinaryTree",
            ConsciousnessTopology::binary_tree(n_nodes, dim, seed),
        ),
        (
            "DenseNetwork",
            ConsciousnessTopology::dense_network(n_nodes, dim, None, seed),
        ),
        (
            "Modular(2)",
            ConsciousnessTopology::modular(n_nodes, dim, 2, seed),
        ),
        (
            "Modular(4)",
            ConsciousnessTopology::modular(n_nodes, dim, 4, seed),
        ),
        (
            "Lattice",
            ConsciousnessTopology::lattice(n_nodes, dim, seed),
        ),
    ];

    // Compute proxy (spectral λ₂) and exact (PhiEngine) for each
    let calc = ConnectivityCalculator::new();
    let engine = PhiEngine::auto();

    let mut results: Vec<(&str, f64, f64)> = Vec::new();

    println!(
        "{:<15} {:>10} {:>10} {:>10}",
        "Topology", "λ₂ Proxy", "Phi", "Time(μs)"
    );
    println!("{:-<15} {:-<10} {:-<10} {:-<10}", "", "", "", "");

    for (name, topo) in &topologies {
        let hvs = to_continuous(topo);
        let proxy = calc.algebraic_connectivity(&hvs);
        let phi_result = engine.compute(&hvs);

        println!(
            "{:<15} {:>10.4} {:>10.4} {:>10}",
            name,
            proxy,
            phi_result.phi,
            phi_result.computation_time.as_micros()
        );

        results.push((name, proxy, phi_result.phi));
    }

    // Compute Spearman rank correlation
    let proxies: Vec<f64> = results.iter().map(|(_, p, _)| *p).collect();
    let phis: Vec<f64> = results.iter().map(|(_, _, p)| *p).collect();
    let rho = spearman_rho(&proxies, &phis);

    println!("\n---");
    println!("Spearman rho = {:.4}", rho);
    println!("N topologies  = {}", results.len());

    // Interpretation
    let interpretation = if rho.abs() > 0.7 {
        "Strong correlation — proxy is a reliable rank-ordering of topologies"
    } else if rho.abs() > 0.4 {
        "Moderate correlation — proxy captures some structure but not all"
    } else {
        "Weak correlation — proxy and exact Phi measure different properties"
    };
    println!("Interpretation: {}", interpretation);

    // Discussion of theoretical basis
    println!("\n## Why CfC Tau Diversity Captures Causal Architecture");
    println!("-----------------------------------------------------");
    println!("CfC networks assign time constants (τ) per neuron.");
    println!("Uniform τ = low differentiation = low integration = low Φ.");
    println!("Diverse τ = multi-scale dynamics = richer causal structure = higher Φ.");
    println!("The spectral proxy (λ₂) measures graph connectivity, which");
    println!("correlates with τ diversity because well-connected topologies");
    println!("naturally develop differentiated temporal dynamics.");
    println!();
}