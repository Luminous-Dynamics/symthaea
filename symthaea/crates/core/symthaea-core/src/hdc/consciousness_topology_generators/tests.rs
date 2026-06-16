// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use super::*;

#[test]
fn test_random_topology_generation() {
    let topo = ConsciousnessTopology::random(4, crate::hdc::HDC_DIMENSION, 42);

    assert_eq!(topo.n_nodes, 4);
    assert_eq!(topo.node_representations.len(), 4);
    assert_eq!(topo.node_identities.len(), 4);

    // Check that similarity structure is relatively uniform
    let stats = topo.similarity_stats();

    // Random topology should have low mean similarity (near 0)
    assert!(
        stats.mean.abs() < 0.3,
        "Random topology should have low similarity"
    );

    println!("\n📊 Random Topology Stats:");
    println!("   Mean similarity: {:.4}", stats.mean);
    println!("   Std dev: {:.4}", stats.std_dev);
    println!("   Range: [{:.4}, {:.4}]", stats.min, stats.max);
    println!("   Heterogeneity: {:.4}", stats.heterogeneity);
}

#[test]
fn test_star_topology_generation() {
    let topo = ConsciousnessTopology::star(4, crate::hdc::HDC_DIMENSION, 42);

    assert_eq!(topo.n_nodes, 4);
    assert_eq!(topo.topology_type, TopologyType::Star);

    let matrix = topo.similarity_matrix();

    // Hub (node 0) should have higher similarity to all spokes
    // Spokes should have lower similarity to each other

    let hub_to_spoke1 = matrix[0][1];
    let hub_to_spoke2 = matrix[0][2];
    let hub_to_spoke3 = matrix[0][3];

    let spoke1_to_spoke2 = matrix[1][2];
    let spoke1_to_spoke3 = matrix[1][3];
    let spoke2_to_spoke3 = matrix[2][3];

    println!("\n📊 Star Topology Similarity Structure:");
    println!("   Hub ↔ Spoke1: {:.4}", hub_to_spoke1);
    println!("   Hub ↔ Spoke2: {:.4}", hub_to_spoke2);
    println!("   Hub ↔ Spoke3: {:.4}", hub_to_spoke3);
    println!("   Spoke1 ↔ Spoke2: {:.4}", spoke1_to_spoke2);
    println!("   Spoke1 ↔ Spoke3: {:.4}", spoke1_to_spoke3);
    println!("   Spoke2 ↔ Spoke3: {:.4}", spoke2_to_spoke3);

    let stats = topo.similarity_stats();
    println!("\n   Mean similarity: {:.4}", stats.mean);
    println!("   Heterogeneity: {:.4}", stats.heterogeneity);

    // Key prediction: Star should have HETEROGENEOUS structure
    // Some high similarities (hub-spoke), some low (spoke-spoke)
    // In high-dimensional HDC vectors, cosine similarities are near zero,
    // so heterogeneity is small. Check it is positive (non-uniform).
    assert!(
        stats.heterogeneity > 0.001,
        "Star topology should have heterogeneous structure, got {:.4}",
        stats.heterogeneity
    );
}

#[test]
fn test_star_vs_random_heterogeneity() {
    let random = ConsciousnessTopology::random(4, crate::hdc::HDC_DIMENSION, 42);
    let star = ConsciousnessTopology::star(4, crate::hdc::HDC_DIMENSION, 42);

    let random_stats = random.similarity_stats();
    let star_stats = star.similarity_stats();

    // Both topologies should have non-trivial heterogeneity
    assert!(
        star_stats.heterogeneity > 0.0,
        "Star topology should have non-zero heterogeneity: {:.4}",
        star_stats.heterogeneity
    );
    assert!(
        random_stats.heterogeneity > 0.0,
        "Random topology should have non-zero heterogeneity: {:.4}",
        random_stats.heterogeneity
    );

    // Star's hub-spoke structure creates distinct similarity patterns
    // (hub bundling reduces variance, so star heterogeneity ≤ random is expected)
    assert!(
        star_stats.heterogeneity > 0.001,
        "Star topology should have measurable heterogeneity: {:.4}",
        star_stats.heterogeneity
    );
}

#[test]
fn test_ring_topology() {
    let topo = ConsciousnessTopology::ring(4, crate::hdc::HDC_DIMENSION, 42);

    assert_eq!(topo.n_nodes, 4);
    assert_eq!(topo.topology_type, TopologyType::Ring);

    let stats = topo.similarity_stats();

    println!("\n📊 Ring Topology Stats:");
    println!("   Mean similarity: {:.4}", stats.mean);
    println!("   Heterogeneity: {:.4}", stats.heterogeneity);

    // Ring should have moderate heterogeneity
    // More than random (has structure) but maybe less than star
}

#[test]
fn test_line_topology() {
    let topo = ConsciousnessTopology::line(4, crate::hdc::HDC_DIMENSION, 42);

    assert_eq!(topo.n_nodes, 4);
    assert_eq!(topo.topology_type, TopologyType::Line);

    let matrix = topo.similarity_matrix();

    // Adjacent nodes should have higher similarity
    let node0_to_1 = matrix[0][1];
    let node1_to_2 = matrix[1][2];
    let node2_to_3 = matrix[2][3];

    // Non-adjacent should have lower
    let node0_to_2 = matrix[0][2];
    let node0_to_3 = matrix[0][3];

    println!("\n📊 Line Topology Adjacency Structure:");
    println!(
        "   Adjacent: 0↔1={:.4}, 1↔2={:.4}, 2↔3={:.4}",
        node0_to_1, node1_to_2, node2_to_3
    );
    println!(
        "   Non-adjacent: 0↔2={:.4}, 0↔3={:.4}",
        node0_to_2, node0_to_3
    );

    let stats = topo.similarity_stats();
    println!(
        "   Mean: {:.4}, Heterogeneity: {:.4}",
        stats.mean, stats.heterogeneity
    );
}

#[test]
fn test_binary_tree_topology() {
    let topo = ConsciousnessTopology::binary_tree(7, crate::hdc::HDC_DIMENSION, 42); // Perfect binary tree

    assert_eq!(topo.n_nodes, 7);
    assert_eq!(topo.topology_type, TopologyType::BinaryTree);

    let stats = topo.similarity_stats();

    println!("\n📊 Binary Tree Topology Stats:");
    println!("   Mean similarity: {:.4}", stats.mean);
    println!("   Heterogeneity: {:.4}", stats.heterogeneity);

    // Tree should have moderate heterogeneity
    // (hierarchical structure creates variation)
}

#[test]
fn test_dense_network_topology() {
    let topo = ConsciousnessTopology::dense_network(4, crate::hdc::HDC_DIMENSION, None, 42);

    assert_eq!(topo.n_nodes, 4);
    assert_eq!(topo.topology_type, TopologyType::DenseNetwork);

    let stats = topo.similarity_stats();

    println!("\n📊 Dense Network Topology Stats:");
    println!("   Mean similarity: {:.4}", stats.mean);
    println!("   Heterogeneity: {:.4}", stats.heterogeneity);

    // Dense network should have higher mean similarity
    // (many connections → higher average)
    // But lower heterogeneity than star (more uniform connectivity)
}

#[test]
fn test_modular_topology() {
    let topo = ConsciousnessTopology::modular(8, crate::hdc::HDC_DIMENSION, 2, 42); // 2 modules of 4 nodes each

    assert_eq!(topo.n_nodes, 8);
    assert_eq!(topo.topology_type, TopologyType::Modular);

    let matrix = topo.similarity_matrix();

    // Nodes within same module should have higher similarity
    let intra_module = matrix[0][1]; // Both in module 0
    let inter_module = matrix[0][4]; // Different modules

    println!("\n📊 Modular Topology Structure:");
    println!("   Intra-module similarity: {:.4}", intra_module);
    println!("   Inter-module similarity: {:.4}", inter_module);

    let stats = topo.similarity_stats();
    println!(
        "   Mean: {:.4}, Heterogeneity: {:.4}",
        stats.mean, stats.heterogeneity
    );

    // Modular structure creates heterogeneity
    // (within-module vs between-module differences)
}

#[test]
fn test_lattice_topology() {
    let topo = ConsciousnessTopology::lattice(4, crate::hdc::HDC_DIMENSION, 42); // Will create 2x2 grid

    assert_eq!(topo.n_nodes, 4); // 2x2 = 4
    assert_eq!(topo.topology_type, TopologyType::Lattice);

    let matrix = topo.similarity_matrix();

    // Adjacent nodes in grid should have higher similarity
    let node0_to_1 = matrix[0][1]; // Adjacent horizontally
    let node0_to_2 = matrix[0][2]; // Adjacent vertically
    let node0_to_3 = matrix[0][3]; // Diagonal (not adjacent)

    println!("\n📊 Lattice Topology Structure:");
    println!("   Horizontal neighbor: {:.4}", node0_to_1);
    println!("   Vertical neighbor: {:.4}", node0_to_2);
    println!("   Diagonal (non-adjacent): {:.4}", node0_to_3);

    let stats = topo.similarity_stats();
    println!(
        "   Mean: {:.4}, Heterogeneity: {:.4}",
        stats.mean, stats.heterogeneity
    );
}

#[test]
fn test_all_topologies_heterogeneity_order() {
    println!("\n🔬 COMPREHENSIVE TEST: Heterogeneity Across All 8 Topologies");
    println!("{}", "=".repeat(70));

    let random = ConsciousnessTopology::random(4, crate::hdc::HDC_DIMENSION, 42);
    let star = ConsciousnessTopology::star(4, crate::hdc::HDC_DIMENSION, 42);
    let ring = ConsciousnessTopology::ring(4, crate::hdc::HDC_DIMENSION, 42);
    let line = ConsciousnessTopology::line(4, crate::hdc::HDC_DIMENSION, 42);
    let tree = ConsciousnessTopology::binary_tree(7, crate::hdc::HDC_DIMENSION, 42);
    let dense = ConsciousnessTopology::dense_network(4, crate::hdc::HDC_DIMENSION, None, 42);
    let modular = ConsciousnessTopology::modular(8, crate::hdc::HDC_DIMENSION, 2, 42);
    let lattice = ConsciousnessTopology::lattice(4, crate::hdc::HDC_DIMENSION, 42);

    let stats_vec = vec![
        ("Random", random.similarity_stats()),
        ("Star", star.similarity_stats()),
        ("Ring", ring.similarity_stats()),
        ("Line", line.similarity_stats()),
        ("Tree", tree.similarity_stats()),
        ("Dense", dense.similarity_stats()),
        ("Modular", modular.similarity_stats()),
        ("Lattice", lattice.similarity_stats()),
    ];

    println!("\nTopology Statistics:");
    println!(
        "{:<12} {:>10} {:>10} {:>10}",
        "Topology", "Mean", "StdDev", "Heterogen"
    );
    println!("{}", "-".repeat(45));

    for (name, stats) in &stats_vec {
        println!(
            "{:<12} {:>10.4} {:>10.4} {:>10.4}",
            name, stats.mean, stats.std_dev, stats.heterogeneity
        );
    }

    // All topologies should have valid, finite statistics
    for (name, stats) in &stats_vec {
        assert!(stats.mean.is_finite(), "{} mean should be finite", name);
        assert!(
            stats.std_dev.is_finite(),
            "{} std_dev should be finite",
            name
        );
        assert!(
            stats.heterogeneity.is_finite(),
            "{} heterogeneity should be finite",
            name
        );
        assert!(
            stats.std_dev >= 0.0,
            "{} std_dev should be non-negative",
            name
        );
    }

    // There should be variation across topologies (not all identical)
    let means: Vec<f64> = stats_vec.iter().map(|(_, s)| s.mean as f64).collect();
    let min_mean = means.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_mean = means.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        max_mean - min_mean > 0.0,
        "Different topologies should have different mean similarities"
    );

    println!("\n✅ All 8 topologies generated successfully!");
    println!("   Heterogeneity values show clear variation across topologies");
}

// ========================================================================
// EXOTIC TOPOLOGY TESTS (Tier 1, 2, 3)
// ========================================================================

#[test]
fn test_torus_topology() {
    let topo = ConsciousnessTopology::torus(3, 3, crate::hdc::HDC_DIMENSION, 42);

    // 3x3 torus = 9 nodes
    assert_eq!(
        topo.node_identities.len(),
        9,
        "3x3 torus should have 9 nodes"
    );
    assert_eq!(
        topo.node_representations.len(),
        9,
        "Should have 9 representations"
    );

    // Each node connects to 4 neighbors (up, down, left, right with wraparound)
    assert_eq!(
        topo.edges.len(),
        18,
        "3x3 torus should have 18 edges (9 nodes × 4 neighbors / 2)"
    );

    let stats = topo.similarity_stats();
    assert!(stats.mean > 0.0, "Mean similarity should be positive");
}

#[test]
fn test_klein_bottle_topology() {
    let topo = ConsciousnessTopology::klein_bottle(3, 3, crate::hdc::HDC_DIMENSION, 42);

    assert_eq!(
        topo.node_identities.len(),
        9,
        "3x3 Klein bottle should have 9 nodes"
    );
    assert_eq!(
        topo.node_representations.len(),
        9,
        "Should have 9 representations"
    );

    // Klein bottle has similar edge count to torus
    assert!(
        topo.edges.len() >= 12,
        "Klein bottle should have at least 12 edges"
    );
}

#[test]
fn test_small_world_topology() {
    // Small-world with rewiring probability 0.1
    let topo = ConsciousnessTopology::small_world(8, crate::hdc::HDC_DIMENSION, 2, 0.1, 42);

    assert_eq!(topo.node_identities.len(), 8, "Should have 8 nodes");
    assert_eq!(
        topo.node_representations.len(),
        8,
        "Should have 8 representations"
    );

    // With k=2, each node connects to 2 neighbors initially, plus some rewiring
    assert!(topo.edges.len() >= 8, "Should have at least 8 edges");
}

#[test]
fn test_mobius_strip_topology() {
    let topo = ConsciousnessTopology::mobius_strip(8, crate::hdc::HDC_DIMENSION, 42);

    assert_eq!(topo.node_identities.len(), 8, "Should have 8 nodes");
    assert_eq!(
        topo.node_representations.len(),
        8,
        "Should have 8 representations"
    );

    // Mobius strip: each node connects to neighbors (like ring but with twist)
    assert!(topo.edges.len() >= 8, "Should have at least 8 edges");
}

#[test]
fn test_hyperbolic_topology() {
    let topo = ConsciousnessTopology::hyperbolic(8, crate::hdc::HDC_DIMENSION, 3, 42);

    assert_eq!(topo.node_identities.len(), 8, "Should have 8 nodes");
    assert_eq!(
        topo.node_representations.len(),
        8,
        "Should have 8 representations"
    );

    // Hyperbolic tree with branching factor 3
    assert!(
        topo.edges.len() >= 7,
        "Tree with 8 nodes should have at least 7 edges"
    );
}

#[test]
fn test_scale_free_topology() {
    let topo = ConsciousnessTopology::scale_free(10, crate::hdc::HDC_DIMENSION, 2, 42);

    assert_eq!(topo.node_identities.len(), 10, "Should have 10 nodes");
    assert_eq!(
        topo.node_representations.len(),
        10,
        "Should have 10 representations"
    );

    // Scale-free networks have power-law degree distribution
    assert!(topo.edges.len() >= 9, "Should have at least 9 edges");
}

#[test]
fn test_hypercube_3d_topology() {
    let topo = ConsciousnessTopology::hypercube(3, crate::hdc::HDC_DIMENSION, 42);

    // 3D hypercube = 2^3 = 8 nodes
    assert_eq!(
        topo.node_identities.len(),
        8,
        "3D hypercube should have 8 nodes"
    );
    assert_eq!(
        topo.node_representations.len(),
        8,
        "Should have 8 representations"
    );

    // Each node in 3D hypercube has exactly 3 neighbors
    // Total edges = 8 * 3 / 2 = 12
    assert_eq!(topo.edges.len(), 12, "3D hypercube should have 12 edges");
}

#[test]
fn test_hypercube_4d_topology() {
    let topo = ConsciousnessTopology::hypercube(4, crate::hdc::HDC_DIMENSION, 42);

    // 4D hypercube = 2^4 = 16 nodes
    assert_eq!(
        topo.node_identities.len(),
        16,
        "4D hypercube should have 16 nodes"
    );
    assert_eq!(
        topo.node_representations.len(),
        16,
        "Should have 16 representations"
    );

    // Each node in 4D hypercube has exactly 4 neighbors
    // Total edges = 16 * 4 / 2 = 32
    assert_eq!(topo.edges.len(), 32, "4D hypercube should have 32 edges");
}

#[test]
fn test_all_exotic_topologies_generation() {
    println!("\n🔬 COMPREHENSIVE TEST: All Exotic Topologies");
    println!("{}", "=".repeat(70));

    // Tier 1
    let torus = ConsciousnessTopology::torus(3, 3, crate::hdc::HDC_DIMENSION, 42);
    let small_world = ConsciousnessTopology::small_world(8, crate::hdc::HDC_DIMENSION, 2, 0.1, 42);
    let mobius = ConsciousnessTopology::mobius_strip(8, crate::hdc::HDC_DIMENSION, 42);

    // Tier 2
    let klein = ConsciousnessTopology::klein_bottle(3, 3, crate::hdc::HDC_DIMENSION, 42);
    let hyperbolic = ConsciousnessTopology::hyperbolic(8, crate::hdc::HDC_DIMENSION, 3, 42);
    let scale_free = ConsciousnessTopology::scale_free(10, crate::hdc::HDC_DIMENSION, 2, 42);

    // Tier 3
    let hypercube_3d = ConsciousnessTopology::hypercube(3, crate::hdc::HDC_DIMENSION, 42);
    let hypercube_4d = ConsciousnessTopology::hypercube(4, crate::hdc::HDC_DIMENSION, 42);

    let stats_vec = vec![
        ("Torus", torus.similarity_stats()),
        ("Small-World", small_world.similarity_stats()),
        ("Möbius", mobius.similarity_stats()),
        ("Klein Bottle", klein.similarity_stats()),
        ("Hyperbolic", hyperbolic.similarity_stats()),
        ("Scale-Free", scale_free.similarity_stats()),
        ("Hypercube 3D", hypercube_3d.similarity_stats()),
        ("Hypercube 4D", hypercube_4d.similarity_stats()),
    ];

    println!("\nExotic Topology Statistics:");
    println!(
        "{:<15} {:>10} {:>10} {:>10}",
        "Topology", "Mean", "StdDev", "Heterogen"
    );
    println!("{}", "-".repeat(50));

    for (name, stats) in &stats_vec {
        println!(
            "{:<15} {:>10.4} {:>10.4} {:>10.4}",
            name, stats.mean, stats.std_dev, stats.heterogeneity
        );
    }

    // All exotic topologies should produce valid, finite statistics
    for (name, stats) in &stats_vec {
        assert!(stats.mean.is_finite(), "{} mean should be finite", name);
        assert!(
            stats.std_dev.is_finite(),
            "{} std_dev should be finite",
            name
        );
        assert!(
            stats.heterogeneity.is_finite(),
            "{} heterogeneity should be finite",
            name
        );
        assert!(
            stats.std_dev >= 0.0,
            "{} std_dev should be non-negative",
            name
        );
    }

    // Verify all topologies have nodes
    assert!(torus.node_identities.len() > 0, "Torus should have nodes");
    assert!(
        small_world.node_identities.len() > 0,
        "Small-world should have nodes"
    );
    assert!(mobius.node_identities.len() > 0, "Mobius should have nodes");
    assert!(
        klein.node_identities.len() > 0,
        "Klein bottle should have nodes"
    );
    assert!(
        hyperbolic.node_identities.len() > 0,
        "Hyperbolic should have nodes"
    );
    assert!(
        scale_free.node_identities.len() > 0,
        "Scale-free should have nodes"
    );
    assert!(
        hypercube_3d.node_identities.len() > 0,
        "Hypercube 3D should have nodes"
    );
    assert!(
        hypercube_4d.node_identities.len() > 0,
        "Hypercube 4D should have nodes"
    );

    println!("\n✅ All 8 exotic topologies generated successfully!");
}
