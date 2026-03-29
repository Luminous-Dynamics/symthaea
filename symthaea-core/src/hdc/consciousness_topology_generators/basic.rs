// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Tier 1 topology generators: Original 8 basic topologies.
//!
//! Random, Star, Ring, Line, BinaryTree, DenseNetwork, Modular.

use super::super::unified_hv::ContinuousHV;
use super::types::{ConsciousnessTopology, TopologyType};

impl ConsciousnessTopology {
    /// Generate a random topology
    ///
    /// All nodes have random connections. This creates a relatively uniform
    /// similarity structure and should have low Φ (baseline).
    ///
    /// # Arguments
    /// * `n_nodes` - Number of nodes
    /// * `dim` - Hypervector dimension
    /// * `seed` - Random seed for reproducibility
    pub fn random(n_nodes: usize, dim: usize, seed: u64) -> Self {
        assert!(n_nodes >= 2, "Need at least 2 nodes");
        assert!(dim >= 256, "Dimension should be >= 256 for good separation");

        // Create unique basis vectors for each node
        let node_identities: Vec<ContinuousHV> =
            (0..n_nodes).map(|i| ContinuousHV::basis(i, dim)).collect();

        // For random topology, each node representation is just a random vector
        // This creates uniform similarity structure
        let node_representations: Vec<ContinuousHV> = (0..n_nodes)
            .map(|i| ContinuousHV::random(dim, seed + (i as u64 * 1000)))
            .collect();

        // Random edges: connect ~50% of possible edges randomly
        // Use proper LCG PRNG to avoid correlations from sequential seeds
        let mut edges = Vec::new();
        let mut rng_state = seed;
        for i in 0..n_nodes {
            for j in (i + 1)..n_nodes {
                // LCG: x_{n+1} = (a * x_n + c) mod m (using wrapping arithmetic for mod 2^64)
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                // Use high bits for better randomness
                if (rng_state >> 33) % 100 < 50 {
                    // ~50% probability
                    edges.push((i, j));
                }
            }
        }

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::Random,
            edges,
        }
    }

    /// Generate a star topology
    ///
    /// One central hub connected to all spokes. Spokes are not connected to each other.
    /// The hub should have high similarity to all spokes, but spokes should have low
    /// similarity to each other. This creates heterogeneous structure and high Φ.
    ///
    /// # Arguments
    /// * `n_nodes` - Number of nodes (must be >= 2, hub + at least 1 spoke)
    /// * `dim` - Hypervector dimension
    /// * `seed` - Random seed for reproducibility
    pub fn star(n_nodes: usize, dim: usize, seed: u64) -> Self {
        assert!(n_nodes >= 2, "Star needs at least 2 nodes (hub + 1 spoke)");
        assert!(dim >= 256, "Dimension should be >= 256 for good separation");

        // Create unique basis vectors for each node with seed-based variation
        // This ensures different seeds produce different Star topologies
        let node_identities: Vec<ContinuousHV> = (0..n_nodes)
            .map(|i| {
                let base = ContinuousHV::basis(i, dim);
                // Add 5% random noise based on seed to create variation
                let noise = ContinuousHV::random(dim, seed + (i as u64 * 1000)).scale(0.05_f32);
                base.add(&noise)
            })
            .collect();

        // Node 0 is the hub, nodes 1..n are spokes
        let hub_id = &node_identities[0];
        let spoke_ids = &node_identities[1..];

        let mut node_representations = Vec::with_capacity(n_nodes);

        // Hub representation = bundle of all spoke connections
        // hub ⊗ spoke1, hub ⊗ spoke2, ..., hub ⊗ spokeN
        let hub_connections: Vec<ContinuousHV> = spoke_ids
            .iter()
            .map(|spoke_id| hub_id.bind(spoke_id))
            .collect();

        // Add seed-based variation to hub to ensure different samples
        let hub_base = ContinuousHV::bundle_owned(&hub_connections);
        let hub_noise = ContinuousHV::random(dim, seed + 999999).scale(0.05_f32);
        let hub_repr = hub_base.add(&hub_noise);
        node_representations.push(hub_repr);

        // Each spoke representation = single connection to hub with seed variation
        // spoke ⊗ hub + small noise
        for (i, spoke_id) in spoke_ids.iter().enumerate() {
            let spoke_base = spoke_id.bind(hub_id);
            // Add 5% noise to each spoke (different for each spoke)
            let spoke_noise =
                ContinuousHV::random(dim, seed + ((i + 1) as u64 * 100000)).scale(0.05_f32);
            let spoke_repr = spoke_base.add(&spoke_noise);
            node_representations.push(spoke_repr);
        }

        // Star edges: hub (node 0) connected to all spokes (nodes 1..n)
        let edges: Vec<(usize, usize)> = (1..n_nodes).map(|i| (0, i)).collect();

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::Star,
            edges,
        }
    }

    /// Generate a ring topology
    ///
    /// Each node connected to its two neighbors in a circle.
    /// Creates moderate integration - more than line, less than dense.
    ///
    /// # Arguments
    /// * `n_nodes` - Number of nodes (must be >= 3 for meaningful ring)
    /// * `dim` - Hypervector dimension
    /// * `seed` - Random seed for reproducibility
    pub fn ring(n_nodes: usize, dim: usize, _seed: u64) -> Self {
        assert!(n_nodes >= 3, "Ring needs at least 3 nodes");
        assert!(dim >= 256, "Dimension should be >= 256 for good separation");

        let node_identities: Vec<ContinuousHV> =
            (0..n_nodes).map(|i| ContinuousHV::basis(i, dim)).collect();

        let mut node_representations = Vec::with_capacity(n_nodes);

        // Each node connects to prev and next in ring
        for i in 0..n_nodes {
            let prev = (i + n_nodes - 1) % n_nodes;
            let next = (i + 1) % n_nodes;

            let conn1 = node_identities[i].bind(&node_identities[prev]);
            let conn2 = node_identities[i].bind(&node_identities[next]);

            let repr = ContinuousHV::bundle_owned(&[conn1, conn2]);
            node_representations.push(repr);
        }

        // Ring edges: each node connects to next (wrapping)
        let edges: Vec<(usize, usize)> = (0..n_nodes).map(|i| (i, (i + 1) % n_nodes)).collect();

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::Ring,
            edges,
        }
    }

    /// Generate a line topology
    ///
    /// Linear chain: node1 - node2 - node3 - node4
    /// Lower integration than ring (no wraparound).
    ///
    /// # Arguments
    /// * `n_nodes` - Number of nodes (must be >= 2)
    /// * `dim` - Hypervector dimension
    /// * `seed` - Random seed for reproducibility
    pub fn line(n_nodes: usize, dim: usize, _seed: u64) -> Self {
        assert!(n_nodes >= 2, "Line needs at least 2 nodes");
        assert!(dim >= 256, "Dimension should be >= 256 for good separation");

        let node_identities: Vec<ContinuousHV> =
            (0..n_nodes).map(|i| ContinuousHV::basis(i, dim)).collect();

        let mut node_representations = Vec::with_capacity(n_nodes);

        for i in 0..n_nodes {
            let mut connections = Vec::new();

            // Connect to previous (if exists)
            if i > 0 {
                connections.push(node_identities[i].bind(&node_identities[i - 1]));
            }

            // Connect to next (if exists)
            if i < n_nodes - 1 {
                connections.push(node_identities[i].bind(&node_identities[i + 1]));
            }

            let repr = if connections.is_empty() {
                // Isolated node (shouldn't happen with n >= 2)
                node_identities[i].clone()
            } else {
                ContinuousHV::bundle_owned(&connections)
            };

            node_representations.push(repr);
        }

        // Line edges: sequential connections
        let edges: Vec<(usize, usize)> =
            (0..n_nodes.saturating_sub(1)).map(|i| (i, i + 1)).collect();

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::Line,
            edges,
        }
    }

    /// Generate a binary tree topology
    ///
    /// Hierarchical structure with parent-child relationships.
    /// Each node connects to its parent (if not root) and children (if not leaf).
    /// Creates moderate integration through hierarchical structure.
    ///
    /// # Arguments
    /// * `n_nodes` - Number of nodes (works best with 2^k - 1 nodes for complete tree)
    /// * `dim` - Hypervector dimension
    /// * `seed` - Random seed for reproducibility
    pub fn binary_tree(n_nodes: usize, dim: usize, _seed: u64) -> Self {
        assert!(n_nodes >= 1, "Tree needs at least 1 node");
        assert!(dim >= 256, "Dimension should be >= 256 for good separation");

        let node_identities: Vec<ContinuousHV> =
            (0..n_nodes).map(|i| ContinuousHV::basis(i, dim)).collect();

        let mut node_representations = Vec::with_capacity(n_nodes);

        for i in 0..n_nodes {
            let mut connections = Vec::new();

            // Connect to parent (if not root)
            if i > 0 {
                let parent = (i - 1) / 2;
                connections.push(node_identities[i].bind(&node_identities[parent]));
            }

            // Connect to left child (if exists)
            let left_child = 2 * i + 1;
            if left_child < n_nodes {
                connections.push(node_identities[i].bind(&node_identities[left_child]));
            }

            // Connect to right child (if exists)
            let right_child = 2 * i + 2;
            if right_child < n_nodes {
                connections.push(node_identities[i].bind(&node_identities[right_child]));
            }

            let repr = if connections.is_empty() {
                // Root node with no children (n=1 case)
                node_identities[i].clone()
            } else {
                ContinuousHV::bundle_owned(&connections)
            };

            node_representations.push(repr);
        }

        // Binary tree edges: parent-child connections
        let mut edges = Vec::new();
        for i in 0..n_nodes {
            let left_child = 2 * i + 1;
            if left_child < n_nodes {
                edges.push((i, left_child));
            }
            let right_child = 2 * i + 2;
            if right_child < n_nodes {
                edges.push((i, right_child));
            }
        }

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::BinaryTree,
            edges,
        }
    }

    /// Generate a dense network topology
    ///
    /// High connectivity: each node connects to many others.
    /// For efficiency, connect each node to k nearest neighbors in index space.
    /// Creates high integration through many connections.
    ///
    /// # Arguments
    /// * `n_nodes` - Number of nodes
    /// * `dim` - Hypervector dimension
    /// * `k` - Number of connections per node (default: n-1 for complete graph)
    /// * `seed` - Random seed for reproducibility
    pub fn dense_network(n_nodes: usize, dim: usize, k: Option<usize>, _seed: u64) -> Self {
        assert!(n_nodes >= 2, "Dense network needs at least 2 nodes");
        assert!(dim >= 256, "Dimension should be >= 256 for good separation");

        // k=None means complete graph (all-to-all connections), otherwise use specified k
        let k = k.map(|k| k.min(n_nodes - 1)).unwrap_or(n_nodes - 1);

        let node_identities: Vec<ContinuousHV> =
            (0..n_nodes).map(|i| ContinuousHV::basis(i, dim)).collect();

        let mut node_representations = Vec::with_capacity(n_nodes);

        for i in 0..n_nodes {
            let mut connections = Vec::new();

            // Connect to k nearest neighbors (in index space, wrapping around)
            for offset in 1..=k {
                let neighbor1 = (i + offset) % n_nodes;
                let neighbor2 = (i + n_nodes - offset) % n_nodes;

                if neighbor1 != i {
                    connections.push(node_identities[i].bind(&node_identities[neighbor1]));
                }
                if neighbor2 != i && neighbor2 != neighbor1 {
                    connections.push(node_identities[i].bind(&node_identities[neighbor2]));
                }
            }

            let repr = ContinuousHV::bundle_owned(&connections);
            node_representations.push(repr);
        }

        // Dense network edges: k-nearest neighbors for each node
        let mut edges = Vec::new();
        for i in 0..n_nodes {
            for offset in 1..=k {
                let neighbor = (i + offset) % n_nodes;
                if neighbor > i {
                    // Only add each edge once
                    edges.push((i, neighbor));
                }
            }
        }

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::DenseNetwork,
            edges,
        }
    }

    /// Generate a modular network topology
    ///
    /// Clustered communities with dense intra-module connections
    /// and sparse inter-module connections. Creates moderate integration
    /// through community structure.
    ///
    /// # Arguments
    /// * `n_nodes` - Number of nodes
    /// * `dim` - Hypervector dimension
    /// * `n_modules` - Number of modules/communities
    /// * `seed` - Random seed for reproducibility
    pub fn modular(n_nodes: usize, dim: usize, n_modules: usize, _seed: u64) -> Self {
        assert!(n_nodes >= n_modules, "Need at least one node per module");
        assert!(
            n_modules >= 2,
            "Need at least 2 modules for meaningful modularity"
        );
        assert!(dim >= 256, "Dimension should be >= 256 for good separation");

        let node_identities: Vec<ContinuousHV> =
            (0..n_nodes).map(|i| ContinuousHV::basis(i, dim)).collect();

        let nodes_per_module = n_nodes / n_modules;

        let mut node_representations = Vec::with_capacity(n_nodes);

        for i in 0..n_nodes {
            let my_module = i / nodes_per_module;
            let mut connections = Vec::new();

            // Connect to all nodes in same module
            let module_start = my_module * nodes_per_module;
            let module_end = ((my_module + 1) * nodes_per_module).min(n_nodes);

            for j in module_start..module_end {
                if j != i {
                    connections.push(node_identities[i].bind(&node_identities[j]));
                }
            }

            // Sparse inter-module connections (just to next module)
            if my_module < n_modules - 1 {
                let next_module_start = (my_module + 1) * nodes_per_module;
                if next_module_start < n_nodes {
                    connections.push(node_identities[i].bind(&node_identities[next_module_start]));
                }
            }

            let repr = if connections.is_empty() {
                node_identities[i].clone()
            } else {
                ContinuousHV::bundle_owned(&connections)
            };

            node_representations.push(repr);
        }

        // Modular edges: intra-module + inter-module connections
        let mut edges = Vec::new();
        for i in 0..n_nodes {
            let my_module = i / nodes_per_module;
            let module_start = my_module * nodes_per_module;
            let module_end = ((my_module + 1) * nodes_per_module).min(n_nodes);

            // Intra-module edges
            for j in module_start..module_end {
                if j > i {
                    edges.push((i, j));
                }
            }

            // Inter-module edge to next module
            if my_module < n_modules - 1 {
                let next_module_start = (my_module + 1) * nodes_per_module;
                if next_module_start < n_nodes && i == module_start {
                    edges.push((i, next_module_start));
                }
            }
        }

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::Modular,
            edges,
        }
    }
}
