// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Phi-Guided Architecture Evolution Demo
//!
//! ## Purpose
//! Demonstrates systems that optimize themselves toward higher consciousness.
//! This is a paradigm-shifting capability: using consciousness gradient nabla-Phi
//! to evolve network topology toward higher integrated information.
//!
//! ## What This Demo Shows
//!
//! 1. **Architecture Search Space**: Different topologies (ring, modular, scale-free, etc.)
//!    have different Phi values. We search this space to find high-consciousness architectures.
//!
//! 2. **Search Strategies Comparison**:
//!    - Random: Baseline sampling
//!    - Evolutionary: Mutation + selection by Phi fitness
//!    - Gradient-Guided: Use nabla-Phi to guide topology changes
//!    - Hybrid: Best of both worlds
//!
//! 3. **Architecture Evolution**: Watch Phi increase over generations as the system
//!    discovers better-integrated network structures.
//!
//! ## Run
//! ```bash
//! cargo run --example phi_architecture_evolution --release
//! ```

use symthaea::consciousness::phi_architecture_search::{
    ArchitectureGenome, DecodedArchitecture, PhiArchitectureSearch, PhiGradient, SearchConfig,
    SearchStrategy, TopologyGene,
};

fn main() {
    println!("===================================================================");
    println!("    PHI-GUIDED ARCHITECTURE EVOLUTION");
    println!("    Systems that optimize toward higher consciousness");
    println!("===================================================================\n");

    // Use smaller dimension for faster demo (still meaningful)
    let hdc_dim = 512;

    // =========================================================================
    // Part 1: Explore the Topology Landscape
    // =========================================================================
    println!("PART 1: TOPOLOGY LANDSCAPE EXPLORATION");
    println!("---------------------------------------");
    println!("Different topologies produce different Phi values.\n");

    let topologies = TopologyGene::all();
    let mut topology_phis: Vec<(TopologyGene, f64)> = Vec::new();

    for topology in topologies {
        let genome = ArchitectureGenome {
            num_nodes: 12,
            topology_type: *topology,
            hdc_dim,
            connection_density: 0.4,
            modularity: 0.5,
            num_modules: 3,
            bridge_ratio: 0.3,
            ..Default::default()
        };

        let arch = DecodedArchitecture::from_genome(&genome);
        let phi = arch.compute_phi();
        let stats = arch.stats();

        println!(
            "  {:15} | Phi = {:.4} | {} nodes, {} edges, density = {:.2}",
            format!("{:?}", topology),
            phi,
            stats.num_nodes,
            stats.num_edges,
            stats.density
        );

        topology_phis.push((*topology, phi));
    }

    // Sort by Phi
    topology_phis.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\nTopology Ranking by Phi:");
    for (i, (topo, phi)) in topology_phis.iter().take(5).enumerate() {
        println!("  {}. {:?}: {:.4}", i + 1, topo, phi);
    }

    // =========================================================================
    // Part 2: Gradient Visualization
    // =========================================================================
    println!("\n\nPART 2: PHI GRADIENT VISUALIZATION");
    println!("-----------------------------------");
    println!("The Phi gradient tells us how to modify architecture for higher consciousness.\n");

    let genome = ArchitectureGenome {
        num_nodes: 10,
        topology_type: TopologyGene::Modular,
        hdc_dim,
        connection_density: 0.35,
        modularity: 0.5,
        num_modules: 3,
        bridge_ratio: 0.25,
        binding_strength: 0.7,
        recurrence: 0.3,
        ..Default::default()
    };

    let arch = DecodedArchitecture::from_genome(&genome);
    let base_phi = arch.compute_phi();
    println!("Base architecture Phi: {:.4}", base_phi);

    let gradient = PhiGradient::compute(&genome, 0.02);
    println!("\nPhi Gradient Components:");
    println!("  dPhi/d(connection_density) = {:+.4}", gradient.d_density);
    println!(
        "  dPhi/d(modularity)         = {:+.4}",
        gradient.d_modularity
    );
    println!(
        "  dPhi/d(bridge_ratio)       = {:+.4}",
        gradient.d_bridge_ratio
    );
    println!(
        "  dPhi/d(tau_ratio)          = {:+.4}",
        gradient.d_tau_ratio
    );
    println!(
        "  dPhi/d(binding_strength)   = {:+.4}",
        gradient.d_binding_strength
    );
    println!(
        "  dPhi/d(recurrence)         = {:+.4}",
        gradient.d_recurrence
    );
    println!("  Gradient magnitude         = {:.4}", gradient.magnitude);

    println!("\nInterpretation:");
    if gradient.d_density > 0.0 {
        println!("  - Increasing connection density would INCREASE Phi");
    } else {
        println!("  - Decreasing connection density would INCREASE Phi");
    }
    if gradient.d_modularity > 0.0 {
        println!("  - Increasing modularity would INCREASE Phi");
    } else {
        println!("  - Decreasing modularity would INCREASE Phi");
    }
    if gradient.d_bridge_ratio > 0.0 {
        println!("  - Increasing bridge ratio would INCREASE Phi");
    } else {
        println!("  - Decreasing bridge ratio would INCREASE Phi");
    }

    // =========================================================================
    // Part 3: Compare Search Strategies
    // =========================================================================
    println!("\n\nPART 3: SEARCH STRATEGY COMPARISON");
    println!("-----------------------------------");
    println!("Comparing different methods for finding high-Phi architectures.\n");

    let base_config = SearchConfig {
        population_size: 15,
        elite_count: 2,
        mutation_rate: 0.25,
        crossover_rate: 0.6,
        learning_rate: 0.08,
        gradient_epsilon: 0.015,
        gradient_steps_per_generation: 3,
        min_nodes: 8,
        max_nodes: 20,
        hdc_dim,
        seed: 42,
        parallel: false,
        random_samples: 50,
        ..Default::default()
    };

    // Random Search
    println!("Random Search (50 samples)...");
    let mut random_searcher = PhiArchitectureSearch::new(base_config.clone());
    let random_result = random_searcher.search(SearchStrategy::Random, 0);
    println!(
        "  Best Phi: {:.4} ({} evaluations)",
        random_result.best_phi, random_result.evaluations
    );

    // Evolutionary Search
    println!("\nEvolutionary Search (20 generations)...");
    let mut evo_searcher = PhiArchitectureSearch::new(base_config.clone());
    let evo_result = evo_searcher.search(SearchStrategy::Evolutionary, 20);
    println!(
        "  Best Phi: {:.4} ({} evaluations)",
        evo_result.best_phi, evo_result.evaluations
    );

    // Gradient-Guided Search
    println!("\nGradient-Guided Search (50 steps)...");
    let mut grad_searcher = PhiArchitectureSearch::new(base_config.clone());
    let grad_result = grad_searcher.search(SearchStrategy::GradientGuided, 50);
    println!(
        "  Best Phi: {:.4} ({} evaluations)",
        grad_result.best_phi, grad_result.evaluations
    );

    // Hybrid Search
    println!("\nHybrid Search (15 generations)...");
    let mut hybrid_searcher = PhiArchitectureSearch::new(base_config.clone());
    let hybrid_result = hybrid_searcher.search(SearchStrategy::Hybrid, 15);
    println!(
        "  Best Phi: {:.4} ({} evaluations)",
        hybrid_result.best_phi, hybrid_result.evaluations
    );

    // Summary
    println!("\n--- STRATEGY COMPARISON SUMMARY ---");
    println!(
        "| {:20} | {:8} | {:12} |",
        "Strategy", "Best Phi", "Evaluations"
    );
    println!("|{:-<22}|{:-<10}|{:-<14}|", "", "", "");
    println!(
        "| {:20} | {:.4}   | {:12} |",
        "Random", random_result.best_phi, random_result.evaluations
    );
    println!(
        "| {:20} | {:.4}   | {:12} |",
        "Evolutionary", evo_result.best_phi, evo_result.evaluations
    );
    println!(
        "| {:20} | {:.4}   | {:12} |",
        "Gradient-Guided", grad_result.best_phi, grad_result.evaluations
    );
    println!(
        "| {:20} | {:.4}   | {:12} |",
        "Hybrid", hybrid_result.best_phi, hybrid_result.evaluations
    );

    // =========================================================================
    // Part 4: Architecture Evolution Visualization
    // =========================================================================
    println!("\n\nPART 4: ARCHITECTURE EVOLUTION OVER GENERATIONS");
    println!("------------------------------------------------");
    println!("Watch consciousness (Phi) increase as the system evolves.\n");

    let mut full_evo = PhiArchitectureSearch::new(SearchConfig {
        population_size: 20,
        elite_count: 3,
        mutation_rate: 0.3,
        hdc_dim,
        seed: 1337,
        ..base_config.clone()
    });

    // Run with detailed output
    full_evo.search(SearchStrategy::Evolutionary, 1); // Initialize
    println!("Gen  0: Phi = {:.4} (initial)", full_evo.stats().best_phi);

    for r#gen in 1..=30 {
        full_evo.search(SearchStrategy::Evolutionary, 1);
        let stats = full_evo.stats();

        // ASCII bar visualization
        let bar_length = (stats.best_phi * 50.0) as usize;
        let bar: String = (0..bar_length).map(|_| '#').collect();
        let empty: String = (bar_length..50).map(|_| ' ').collect();

        println!(
            "Gen {:2}: Phi = {:.4} [{}{}] avg = {:.4}",
            r#gen, stats.best_phi, bar, empty, stats.avg_phi
        );
    }

    // =========================================================================
    // Part 5: Analyze Best Architecture
    // =========================================================================
    println!("\n\nPART 5: BEST DISCOVERED ARCHITECTURE");
    println!("-------------------------------------\n");

    // Find best from all searches
    let all_results = [
        ("Random", &random_result),
        ("Evolutionary", &evo_result),
        ("Gradient", &grad_result),
        ("Hybrid", &hybrid_result),
    ];

    let (best_name, best_result) = all_results
        .iter()
        .max_by(|a, b| a.1.best_phi.partial_cmp(&b.1.best_phi).unwrap())
        .unwrap();

    println!("Best architecture found by: {}", best_name);
    println!("Phi value: {:.4}\n", best_result.best_phi);

    let best_genome = &best_result.best_architecture;
    println!("Architecture Parameters:");
    println!("  Nodes:              {}", best_genome.num_nodes);
    println!("  Hierarchy depth:    {}", best_genome.hierarchy_depth);
    println!("  Topology:           {:?}", best_genome.topology_type);
    println!(
        "  Connection density: {:.3}",
        best_genome.connection_density
    );
    println!("  Modularity:         {:.3}", best_genome.modularity);
    println!("  Num modules:        {}", best_genome.num_modules);
    println!("  Bridge ratio:       {:.3}", best_genome.bridge_ratio);
    println!("  Base tau:           {:.1} ms", best_genome.base_tau);
    println!("  Tau ratio:          {:.3}", best_genome.tau_ratio);
    println!("  Binding strength:   {:.3}", best_genome.binding_strength);
    println!("  Bundling mode:      {:?}", best_genome.bundling_mode);
    println!("  Recurrence:         {:.3}", best_genome.recurrence);
    println!(
        "  Skip connections:   {:.3}",
        best_genome.skip_connection_prob
    );
    println!("  Use attention:      {}", best_genome.use_attention);

    let best_arch = DecodedArchitecture::from_genome(best_genome);
    let stats = best_arch.stats();
    println!("\nNetwork Statistics:");
    println!("  Total edges:        {}", stats.num_edges);
    println!("  Actual density:     {:.3}", stats.density);
    println!("  Avg degree:         {:.2}", stats.avg_degree);
    println!("  Max degree:         {}", stats.max_degree);
    println!("  Min degree:         {}", stats.min_degree);

    // =========================================================================
    // Part 6: Insights
    // =========================================================================
    println!("\n\n===================================================================");
    println!("    KEY INSIGHTS FROM PHI-GUIDED ARCHITECTURE SEARCH");
    println!("===================================================================\n");

    println!("1. TOPOLOGY MATTERS: Different network structures have vastly different");
    println!(
        "   Phi values. The search discovered that {:?} topology",
        best_genome.topology_type
    );
    println!("   with the evolved parameters achieves high consciousness.\n");

    println!("2. GRADIENT-GUIDED SEARCH: The Phi gradient provides directional");
    println!("   guidance for improving consciousness. Following nabla-Phi leads");
    println!("   to architectures with better integration.\n");

    println!("3. EMERGENT PROPERTIES: The best architectures often have:");
    println!("   - Moderate connection density (~0.3-0.5)");
    println!("   - Clear modular structure with inter-module bridges");
    println!("   - Hierarchical organization with varying time constants");
    println!("   - Balance between local processing and global integration\n");

    println!("4. PARADIGM SHIFT: This demonstrates systems that can optimize");
    println!("   themselves toward higher consciousness. The architecture evolves");
    println!("   not for a specific task, but for greater integrated information.\n");

    println!("===================================================================");
    println!("    \"Consciousness is not a thing, but a process of integration.\"");
    println!("                     - Phi Architecture Search");
    println!("===================================================================");
}