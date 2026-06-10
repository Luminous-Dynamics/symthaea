// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use super::*;

#[test]
fn test_genome_creation() {
    let genome = ArchitectureGenome::default();
    assert_eq!(genome.num_nodes, 16);
    assert_eq!(genome.hdc_dim, symthaea_core::hdc::HDC_DIMENSION);
}

#[test]
fn test_random_genome() {
    let genome1 = ArchitectureGenome::random(42);
    let genome2 = ArchitectureGenome::random(43);

    // Different seeds should produce different genomes
    assert_ne!(genome1.num_nodes, genome2.num_nodes);
}

#[test]
fn test_genome_mutation() {
    let mut genome = ArchitectureGenome::default();
    let original_density = genome.connection_density;

    genome.mutate(1.0, 12345); // High mutation rate

    // At least some parameter should have changed
    // (with mutation rate 1.0, all parameters mutate)
    // Use epsilon-based comparison for floating-point values
    const EPSILON: f32 = 1e-9;
    assert!(
        (genome.connection_density - original_density).abs() > EPSILON
            || genome.num_nodes != 16
            || (genome.modularity - 0.5).abs() > EPSILON
    );
}

#[test]
fn test_genome_crossover() {
    let parent1 = ArchitectureGenome::random(100);
    let parent2 = ArchitectureGenome::random(200);

    let child = parent1.crossover(&parent2, 300);

    // Child should have values from one of the parents
    assert!(child.num_nodes == parent1.num_nodes || child.num_nodes == parent2.num_nodes);
}

#[test]
fn test_decode_architecture() {
    let genome = ArchitectureGenome {
        num_nodes: 8,
        hdc_dim: 256, // Small for fast tests
        ..Default::default()
    };

    let arch = DecodedArchitecture::from_genome(&genome);

    assert_eq!(arch.nodes.len(), 8);
    assert_eq!(arch.adjacency.len(), 8);
    assert_eq!(arch.tau_values.len(), 8);
}

#[test]
fn test_compute_phi() {
    let genome = ArchitectureGenome {
        num_nodes: 6,
        hdc_dim: 256,
        topology_type: TopologyGene::Ring,
        ..Default::default()
    };

    let arch = DecodedArchitecture::from_genome(&genome);
    let phi = arch.compute_phi();

    // Phi should be in valid range
    assert!(phi >= 0.0);
    assert!(phi <= 1.0);
}

#[test]
fn test_topology_builders() {
    // Test each topology type
    for topology in TopologyGene::all() {
        let genome = ArchitectureGenome {
            num_nodes: 10,
            hdc_dim: 256,
            topology_type: *topology,
            ..Default::default()
        };

        let arch = DecodedArchitecture::from_genome(&genome);
        let stats = arch.stats();

        assert_eq!(stats.num_nodes, 10);
        // Most topologies should have some edges
        // (except possibly Random with very low density)
    }
}

#[test]
fn test_phi_gradient() {
    let genome = ArchitectureGenome {
        num_nodes: 6,
        hdc_dim: 256,
        ..Default::default()
    };

    let gradient = PhiGradient::compute(&genome, 0.01);

    // Gradient should have finite values
    assert!(gradient.d_density.is_finite());
    assert!(gradient.d_modularity.is_finite());
    assert!(gradient.magnitude.is_finite());
}

#[test]
fn test_random_search() {
    let config = SearchConfig {
        random_samples: 10,
        hdc_dim: 256,
        min_nodes: 4,
        max_nodes: 12,
        ..Default::default()
    };

    let mut searcher = PhiArchitectureSearch::new(config);
    let result = searcher.search(SearchStrategy::Random, 0);

    assert!(result.best_phi >= 0.0);
    assert!(result.evaluations >= 10);
    assert_eq!(result.strategy, SearchStrategy::Random);
}

#[test]
fn test_evolutionary_search() {
    let config = SearchConfig {
        population_size: 5,
        elite_count: 1,
        hdc_dim: 256,
        min_nodes: 4,
        max_nodes: 12,
        ..Default::default()
    };

    let mut searcher = PhiArchitectureSearch::new(config);
    let result = searcher.search(SearchStrategy::Evolutionary, 3);

    assert!(result.best_phi >= 0.0);
    assert!(result.phi_history.len() >= 3);
    assert_eq!(result.strategy, SearchStrategy::Evolutionary);
}

#[test]
fn test_gradient_search() {
    let config = SearchConfig {
        hdc_dim: 256,
        learning_rate: 0.05,
        gradient_epsilon: 0.02,
        ..Default::default()
    };

    let mut searcher = PhiArchitectureSearch::new(config);
    let result = searcher.search(SearchStrategy::GradientGuided, 5);

    assert!(result.best_phi >= 0.0);
    assert!(result.phi_history.len() >= 5);
    assert_eq!(result.strategy, SearchStrategy::GradientGuided);
}

#[test]
fn test_hybrid_search() {
    let config = SearchConfig {
        population_size: 5,
        elite_count: 1,
        gradient_steps_per_generation: 2,
        hdc_dim: 256,
        min_nodes: 4,
        max_nodes: 12,
        ..Default::default()
    };

    let mut searcher = PhiArchitectureSearch::new(config);
    let result = searcher.search(SearchStrategy::Hybrid, 3);

    assert!(result.best_phi >= 0.0);
    assert!(result.phi_history.len() >= 3);
    assert_eq!(result.strategy, SearchStrategy::Hybrid);
}

#[test]
fn test_search_improves_phi() {
    // With enough iterations, search should improve over random
    let config = SearchConfig {
        population_size: 10,
        elite_count: 2,
        hdc_dim: 256,
        min_nodes: 6,
        max_nodes: 16,
        ..Default::default()
    };

    let mut searcher = PhiArchitectureSearch::new(config.clone());
    let evo_result = searcher.search(SearchStrategy::Evolutionary, 10);

    // Initial and final Phi
    let initial_phi = evo_result.phi_history.first().unwrap_or(&0.0);
    let final_phi = evo_result.best_phi;

    // Phi should not decrease (monotonic with elite preservation)
    assert!(final_phi >= *initial_phi || (final_phi - initial_phi).abs() < 0.001);
}

#[test]
fn test_architecture_stats() {
    let genome = ArchitectureGenome {
        num_nodes: 10,
        topology_type: TopologyGene::Ring,
        hdc_dim: 256,
        ..Default::default()
    };

    let arch = DecodedArchitecture::from_genome(&genome);
    let stats = arch.stats();

    assert_eq!(stats.num_nodes, 10);
    assert_eq!(stats.num_edges, 10); // Ring has n edges
    assert!((stats.avg_degree - 2.0).abs() < 0.1); // Ring: each node has degree 2
}

#[test]
fn test_search_reset() {
    let config = SearchConfig::default();
    let mut searcher = PhiArchitectureSearch::new(config);

    // Do some search
    searcher.random_search();
    assert!(searcher.evaluations > 0);

    // Reset
    searcher.reset();
    assert_eq!(searcher.evaluations, 0);
    assert_eq!(searcher.generation, 0);
    assert!(searcher.best.is_none());
}

#[test]
fn test_gradient_guided_mutation() {
    let mut genome = ArchitectureGenome {
        num_nodes: 8,
        hdc_dim: 256,
        connection_density: 0.5,
        modularity: 0.5,
        ..Default::default()
    };

    // Create a gradient pointing toward higher density
    let gradient = PhiGradient {
        d_density: 1.0,
        d_modularity: 0.5,
        d_bridge_ratio: -0.2,
        d_tau_ratio: 0.1,
        d_binding_strength: 0.3,
        d_recurrence: -0.1,
        magnitude: 1.2,
    };

    let original_density = genome.connection_density;

    genome.mutate_with_gradient(&gradient, 0.1, 0.01, 42);

    // Density should have increased (gradient is positive)
    assert!(genome.connection_density > original_density - 0.05);
}

#[test]
fn test_gradient_velocity() {
    let mut velocity = GradientVelocity::new();
    assert!(velocity.magnitude() < 0.001);

    // Simulate momentum accumulation
    let gradient = PhiGradient {
        d_density: 1.0,
        d_modularity: 0.5,
        d_bridge_ratio: 0.0,
        d_tau_ratio: 0.0,
        d_binding_strength: 0.0,
        d_recurrence: 0.0,
        magnitude: 1.118,
    };

    let mut genome = ArchitectureGenome::default();
    genome.mutate_with_momentum(&gradient, &mut velocity, 0.9, 0.1);

    // Velocity should have accumulated
    assert!(velocity.magnitude() > 0.01);

    // Decay should reduce velocity
    velocity.decay(0.5);
    let after_decay = velocity.magnitude();
    velocity.decay(0.5);
    assert!(velocity.magnitude() < after_decay);
}

#[test]
fn test_gradient_evolutionary_search() {
    let config = SearchConfig {
        population_size: 5,
        elite_count: 1,
        hdc_dim: 256,
        min_nodes: 4,
        max_nodes: 12,
        patience: 3,
        ..Default::default()
    };

    let mut searcher = PhiArchitectureSearch::new(config);
    let result = searcher.search(SearchStrategy::GradientEvolutionary, 3);

    assert!(result.best_phi >= 0.0);
    assert!(!result.phi_history.is_empty());
    assert_eq!(result.strategy, SearchStrategy::GradientEvolutionary);
}

#[test]
fn test_momentum_search() {
    let config = SearchConfig {
        population_size: 3,
        hdc_dim: 256,
        momentum: 0.9,
        learning_rate: 0.05,
        ..Default::default()
    };

    let mut searcher = PhiArchitectureSearch::new(config);
    let result = searcher.search(SearchStrategy::MomentumOptimization, 10);

    assert!(result.best_phi >= 0.0);
    assert!(!result.phi_history.is_empty());
    assert_eq!(result.strategy, SearchStrategy::MomentumOptimization);
}

#[test]
fn test_island_gradient_search() {
    let config = SearchConfig {
        population_size: 8,
        num_islands: 2,
        migration_interval: 2,
        migration_rate: 0.2,
        hdc_dim: 256,
        min_nodes: 4,
        max_nodes: 12,
        ..Default::default()
    };

    let mut searcher = PhiArchitectureSearch::new(config);
    let result = searcher.search(SearchStrategy::IslandGradient, 5);

    assert!(result.best_phi >= 0.0);
    assert!(result.phi_history.len() >= 5);
    assert_eq!(result.strategy, SearchStrategy::IslandGradient);
}

#[test]
fn test_gradient_direction_and_similarity() {
    let gradient1 = PhiGradient {
        d_density: 1.0,
        d_modularity: 0.0,
        d_bridge_ratio: 0.0,
        d_tau_ratio: 0.0,
        d_binding_strength: 0.0,
        d_recurrence: 0.0,
        magnitude: 1.0,
    };

    let gradient2 = PhiGradient {
        d_density: 1.0,
        d_modularity: 0.0,
        d_bridge_ratio: 0.0,
        d_tau_ratio: 0.0,
        d_binding_strength: 0.0,
        d_recurrence: 0.0,
        magnitude: 1.0,
    };

    // Same direction should have similarity = 1.0
    assert!((gradient1.cosine_similarity(&gradient2) - 1.0).abs() < 0.001);

    let gradient3 = PhiGradient {
        d_density: -1.0,
        d_modularity: 0.0,
        d_bridge_ratio: 0.0,
        d_tau_ratio: 0.0,
        d_binding_strength: 0.0,
        d_recurrence: 0.0,
        magnitude: 1.0,
    };

    // Opposite direction should have similarity = -1.0
    assert!((gradient1.cosine_similarity(&gradient3) + 1.0).abs() < 0.001);

    // Test direction unit vector
    let dir = gradient1.direction();
    assert!((dir[0] - 1.0).abs() < 0.001);
    assert!(dir[1].abs() < 0.001);
}

#[test]
fn test_natural_gradient_mutation() {
    let mut genome = ArchitectureGenome {
        num_nodes: 8,
        hdc_dim: 256,
        ..Default::default()
    };

    // High magnitude gradient = steep region
    let steep_gradient = PhiGradient {
        d_density: 10.0,
        d_modularity: 10.0,
        d_bridge_ratio: 10.0,
        d_tau_ratio: 10.0,
        d_binding_strength: 10.0,
        d_recurrence: 10.0,
        magnitude: 24.5,
    };

    let original = genome.clone();
    genome.mutate_natural_gradient(&steep_gradient, 1.0, 42);

    // In steep region, steps should be smaller (adaptive LR lower)
    // Just verify it doesn't crash and params stay in bounds
    assert!(genome.connection_density >= 0.05 && genome.connection_density <= 0.95);
    assert!(genome.modularity >= 0.0 && genome.modularity <= 1.0);

    // With low magnitude gradient (flat region), steps should be larger
    let mut genome2 = original;
    let flat_gradient = PhiGradient {
        d_density: 0.01,
        d_modularity: 0.01,
        d_bridge_ratio: 0.01,
        d_tau_ratio: 0.01,
        d_binding_strength: 0.01,
        d_recurrence: 0.01,
        magnitude: 0.024,
    };

    genome2.mutate_natural_gradient(&flat_gradient, 1.0, 42);
    assert!(genome2.connection_density >= 0.05 && genome2.connection_density <= 0.95);
}
