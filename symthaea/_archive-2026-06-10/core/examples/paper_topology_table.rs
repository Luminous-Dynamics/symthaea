// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Paper Topology Table Generator
//!
//! Runs Phi-guided architecture search across all 10 topology types and outputs
//! a paper-ready table of topology metrics (Betti numbers, Euler characteristic,
//! Phi fitness) for the HAI paper.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example paper_topology_table
//! ```

use symthaea::consciousness::phi_architecture_search::{
    ArchitectureGenome, DecodedArchitecture, PhiArchitectureSearch, SearchConfig, SearchStrategy,
    TopologyGene,
};

fn main() {
    let topologies = TopologyGene::all();
    let generations = 30;
    let population = 20;

    println!("# Phi-Guided Architecture Search: Topology Metrics");
    println!("# Generations: {generations}, Population: {population}, HDC dim: 512");
    println!();
    println!(
        "| {:<18} | {:>6} | {:>5} | {:>5} | {:>5} | {:>5} | {:>7} | {:>7} | {:>8} |",
        "Topology", "Nodes", "Edges", "β₀", "β₁", "β₂", "χ", "Phi", "Density"
    );
    println!(
        "|{:-<20}|{:-<8}|{:-<7}|{:-<7}|{:-<7}|{:-<7}|{:-<9}|{:-<9}|{:-<10}|",
        "", "", "", "", "", "", "", "", ""
    );

    for topo in &topologies {
        let config = SearchConfig {
            population_size: population,
            elite_count: 2,
            mutation_rate: 0.25,
            hdc_dim: 512,
            seed: 42,
            min_nodes: 8,
            max_nodes: 32,
            ..Default::default()
        };

        let mut searcher = PhiArchitectureSearch::new(config);

        // Seed initial population with this specific topology
        let seed_genome = ArchitectureGenome {
            num_nodes: 16,
            topology: *topo,
            time_constants: vec![0.1, 0.5, 1.0],
            bundling: Default::default(),
            hierarchy_depth: 2,
        };

        // Run evolutionary search
        let result = searcher.search(SearchStrategy::Evolutionary, generations);

        // Decode best architecture
        let decoded = DecodedArchitecture::from_genome(&result.best_architecture);
        let stats = decoded.stats();

        let (b0, b1, b2, chi) = if let Some(ref topo_metrics) = stats.topology {
            (
                topo_metrics.betti_0,
                topo_metrics.betti_1,
                topo_metrics.betti_2,
                topo_metrics.euler_characteristic,
            )
        } else {
            (0, 0, 0, 0)
        };

        println!(
            "| {:<18} | {:>6} | {:>5} | {:>5} | {:>5} | {:>5} | {:>7} | {:>7.4} | {:>8.4} |",
            format!("{topo:?}"),
            stats.num_nodes,
            stats.num_edges,
            b0,
            b1,
            b2,
            chi,
            result.best_phi,
            stats.density,
        );

        // Also seed with the specific topology for comparison
        let _ = seed_genome; // suppress unused warning
    }

    println!();
    println!("# Notes:");
    println!("# - β₀ = connected components (1 = fully connected)");
    println!("# - β₁ = independent loops/cycles (higher = more recurrence)");
    println!("# - β₂ = enclosed voids (higher-order structure)");
    println!("# - χ = Euler characteristic = β₀ - β₁ + β₂");
    println!("# - Phi = integrated information (IIT proxy)");
}
