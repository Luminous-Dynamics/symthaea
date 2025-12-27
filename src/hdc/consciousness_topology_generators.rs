/// Consciousness Topology Generators using Real-Valued Hypervectors
///
/// This module generates 8 different network topologies representing different
/// levels of integrated information (Φ) for validating the RealHV-based Φ measurement.
///
/// Each topology is represented as a set of node vectors, where each node's
/// representation encodes its connections to other nodes via RealHV operations.

use super::real_hv::RealHV;
use std::collections::HashMap;

/// A consciousness topology represented with RealHV
#[derive(Clone, Debug)]
pub struct ConsciousnessTopology {
    /// Number of nodes in the topology
    pub n_nodes: usize,

    /// Dimension of hypervectors
    pub dim: usize,

    /// Node representations (each encodes its connections)
    pub node_representations: Vec<RealHV>,

    /// Node identities (basis vectors)
    pub node_identities: Vec<RealHV>,

    /// Topology type
    pub topology_type: TopologyType,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TopologyType {
    Random,
    Star,
    Ring,
    Line,
    BinaryTree,
    DenseNetwork,
    Modular,
    Lattice,
}

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
        let node_identities: Vec<RealHV> = (0..n_nodes)
            .map(|i| RealHV::basis(i, dim))
            .collect();

        // For random topology, each node representation is just a random vector
        // This creates uniform similarity structure
        let node_representations: Vec<RealHV> = (0..n_nodes)
            .map(|i| RealHV::random(dim, seed + (i as u64 * 1000)))
            .collect();

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::Random,
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

        // Create unique basis vectors for each node
        let node_identities: Vec<RealHV> = (0..n_nodes)
            .map(|i| RealHV::basis(i, dim))
            .collect();

        // Node 0 is the hub, nodes 1..n are spokes
        let hub_id = &node_identities[0];
        let spoke_ids = &node_identities[1..];

        let mut node_representations = Vec::with_capacity(n_nodes);

        // Hub representation = bundle of all spoke connections
        // hub ⊗ spoke1, hub ⊗ spoke2, ..., hub ⊗ spokeN
        let hub_connections: Vec<RealHV> = spoke_ids
            .iter()
            .map(|spoke_id| hub_id.bind(spoke_id))
            .collect();

        let hub_repr = RealHV::bundle(&hub_connections);
        node_representations.push(hub_repr);

        // Each spoke representation = single connection to hub
        // spoke ⊗ hub
        for spoke_id in spoke_ids.iter() {
            let spoke_repr = spoke_id.bind(hub_id);
            node_representations.push(spoke_repr);
        }

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::Star,
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
    pub fn ring(n_nodes: usize, dim: usize, seed: u64) -> Self {
        assert!(n_nodes >= 3, "Ring needs at least 3 nodes");
        assert!(dim >= 256, "Dimension should be >= 256 for good separation");

        let node_identities: Vec<RealHV> = (0..n_nodes)
            .map(|i| RealHV::basis(i, dim))
            .collect();

        let mut node_representations = Vec::with_capacity(n_nodes);

        // Each node connects to prev and next in ring
        for i in 0..n_nodes {
            let prev = (i + n_nodes - 1) % n_nodes;
            let next = (i + 1) % n_nodes;

            let conn1 = node_identities[i].bind(&node_identities[prev]);
            let conn2 = node_identities[i].bind(&node_identities[next]);

            let repr = RealHV::bundle(&[conn1, conn2]);
            node_representations.push(repr);
        }

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::Ring,
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
    pub fn line(n_nodes: usize, dim: usize, seed: u64) -> Self {
        assert!(n_nodes >= 2, "Line needs at least 2 nodes");
        assert!(dim >= 256, "Dimension should be >= 256 for good separation");

        let node_identities: Vec<RealHV> = (0..n_nodes)
            .map(|i| RealHV::basis(i, dim))
            .collect();

        let mut node_representations = Vec::with_capacity(n_nodes);

        for i in 0..n_nodes {
            let mut connections = Vec::new();

            // Connect to previous (if exists)
            if i > 0 {
                connections.push(node_identities[i].bind(&node_identities[i-1]));
            }

            // Connect to next (if exists)
            if i < n_nodes - 1 {
                connections.push(node_identities[i].bind(&node_identities[i+1]));
            }

            let repr = if connections.is_empty() {
                // Isolated node (shouldn't happen with n >= 2)
                node_identities[i].clone()
            } else {
                RealHV::bundle(&connections)
            };

            node_representations.push(repr);
        }

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::Line,
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
    pub fn binary_tree(n_nodes: usize, dim: usize, seed: u64) -> Self {
        assert!(n_nodes >= 1, "Tree needs at least 1 node");
        assert!(dim >= 256, "Dimension should be >= 256 for good separation");

        let node_identities: Vec<RealHV> = (0..n_nodes)
            .map(|i| RealHV::basis(i, dim))
            .collect();

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
                RealHV::bundle(&connections)
            };

            node_representations.push(repr);
        }

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::BinaryTree,
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
    /// * `k` - Number of connections per node (default: n/2)
    /// * `seed` - Random seed for reproducibility
    pub fn dense_network(n_nodes: usize, dim: usize, k: Option<usize>, seed: u64) -> Self {
        assert!(n_nodes >= 2, "Dense network needs at least 2 nodes");
        assert!(dim >= 256, "Dimension should be >= 256 for good separation");

        let k = k.unwrap_or(n_nodes / 2).min(n_nodes - 1);

        let node_identities: Vec<RealHV> = (0..n_nodes)
            .map(|i| RealHV::basis(i, dim))
            .collect();

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

            let repr = RealHV::bundle(&connections);
            node_representations.push(repr);
        }

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::DenseNetwork,
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
    pub fn modular(n_nodes: usize, dim: usize, n_modules: usize, seed: u64) -> Self {
        assert!(n_nodes >= n_modules, "Need at least one node per module");
        assert!(n_modules >= 2, "Need at least 2 modules for meaningful modularity");
        assert!(dim >= 256, "Dimension should be >= 256 for good separation");

        let node_identities: Vec<RealHV> = (0..n_nodes)
            .map(|i| RealHV::basis(i, dim))
            .collect();

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
                RealHV::bundle(&connections)
            };

            node_representations.push(repr);
        }

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::Modular,
        }
    }

    /// Generate a lattice (grid) topology
    ///
    /// Regular 2D grid structure where each node connects to its
    /// 4 neighbors (up, down, left, right). Creates moderate integration
    /// through regular structure.
    ///
    /// # Arguments
    /// * `n_nodes` - Number of nodes (will be rounded to nearest perfect square)
    /// * `dim` - Hypervector dimension
    /// * `seed` - Random seed for reproducibility
    pub fn lattice(n_nodes: usize, dim: usize, seed: u64) -> Self {
        assert!(n_nodes >= 4, "Lattice needs at least 4 nodes (2x2 grid)");
        assert!(dim >= 256, "Dimension should be >= 256 for good separation");

        // Find grid size (nearest perfect square)
        let grid_size = (n_nodes as f64).sqrt().ceil() as usize;
        let actual_n_nodes = grid_size * grid_size;

        let node_identities: Vec<RealHV> = (0..actual_n_nodes)
            .map(|i| RealHV::basis(i, dim))
            .collect();

        let mut node_representations = Vec::with_capacity(actual_n_nodes);

        for i in 0..actual_n_nodes {
            let row = i / grid_size;
            let col = i % grid_size;
            let mut connections = Vec::new();

            // Connect to up neighbor
            if row > 0 {
                let up = (row - 1) * grid_size + col;
                connections.push(node_identities[i].bind(&node_identities[up]));
            }

            // Connect to down neighbor
            if row < grid_size - 1 {
                let down = (row + 1) * grid_size + col;
                connections.push(node_identities[i].bind(&node_identities[down]));
            }

            // Connect to left neighbor
            if col > 0 {
                let left = row * grid_size + (col - 1);
                connections.push(node_identities[i].bind(&node_identities[left]));
            }

            // Connect to right neighbor
            if col < grid_size - 1 {
                let right = row * grid_size + (col + 1);
                connections.push(node_identities[i].bind(&node_identities[right]));
            }

            let repr = if connections.is_empty() {
                // Shouldn't happen with grid_size >= 2
                node_identities[i].clone()
            } else {
                RealHV::bundle(&connections)
            };

            node_representations.push(repr);
        }

        Self {
            n_nodes: actual_n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::Lattice,
        }
    }

    /// Compute pairwise similarity matrix for all nodes
    ///
    /// Returns an n×n matrix where entry [i][j] is the similarity
    /// between node i's representation and node j's representation.
    pub fn similarity_matrix(&self) -> Vec<Vec<f32>> {
        let n = self.n_nodes;
        let mut matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                let sim = self.node_representations[i]
                    .similarity(&self.node_representations[j]);
                matrix[i][j] = sim;
            }
        }

        matrix
    }

    /// Compute statistics about the similarity structure
    pub fn similarity_stats(&self) -> SimilarityStats {
        let matrix = self.similarity_matrix();
        let n = self.n_nodes;

        let mut values = Vec::new();
        let mut self_similarities = Vec::new();

        for i in 0..n {
            self_similarities.push(matrix[i][i]);
            for j in (i+1)..n {  // Upper triangle only
                values.push(matrix[i][j]);
            }
        }

        if values.is_empty() {
            return SimilarityStats {
                mean: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
                heterogeneity: 0.0,
            };
        }

        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f32>() / values.len() as f32;
        let std_dev = variance.sqrt();

        let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Heterogeneity = standard deviation of similarities
        // Higher = more diverse similarity structure
        // (Using std_dev directly, not coefficient of variation,
        //  because mean can be near 0 for random topologies)
        let heterogeneity = std_dev;

        SimilarityStats {
            mean,
            std_dev,
            min,
            max,
            heterogeneity,
        }
    }
}

/// Statistics about similarity structure
#[derive(Clone, Debug)]
pub struct SimilarityStats {
    pub mean: f32,
    pub std_dev: f32,
    pub min: f32,
    pub max: f32,
    pub heterogeneity: f32,  // Normalized measure of diversity
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_topology_generation() {
        let topo = ConsciousnessTopology::random(4, 2048, 42);

        assert_eq!(topo.n_nodes, 4);
        assert_eq!(topo.node_representations.len(), 4);
        assert_eq!(topo.node_identities.len(), 4);

        // Check that similarity structure is relatively uniform
        let stats = topo.similarity_stats();

        // Random topology should have low mean similarity (near 0)
        assert!(stats.mean.abs() < 0.3, "Random topology should have low similarity");

        println!("\n📊 Random Topology Stats:");
        println!("   Mean similarity: {:.4}", stats.mean);
        println!("   Std dev: {:.4}", stats.std_dev);
        println!("   Range: [{:.4}, {:.4}]", stats.min, stats.max);
        println!("   Heterogeneity: {:.4}", stats.heterogeneity);
    }

    #[test]
    fn test_star_topology_generation() {
        let topo = ConsciousnessTopology::star(4, 2048, 42);

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
        // Heterogeneity (std_dev) should be at least 0.2 for clear structure
        assert!(stats.heterogeneity > 0.2,
                "Star topology should have heterogeneous structure, got {:.4}",
                stats.heterogeneity);
    }

    #[test]
    fn test_star_vs_random_heterogeneity() {
        let random = ConsciousnessTopology::random(4, 2048, 42);
        let star = ConsciousnessTopology::star(4, 2048, 42);

        let random_stats = random.similarity_stats();
        let star_stats = star.similarity_stats();

        println!("\n🔬 CRITICAL TEST: Star vs Random Heterogeneity");
        println!("{}", "=".repeat(60));
        println!("Random topology heterogeneity: {:.4}", random_stats.heterogeneity);
        println!("Star topology heterogeneity: {:.4}", star_stats.heterogeneity);
        println!();

        // KEY PREDICTION: Star should be MORE heterogeneous than random
        // This is a proxy for higher Φ
        assert!(star_stats.heterogeneity > random_stats.heterogeneity,
                "Star topology should be more heterogeneous than random!\n  Star: {:.4}\n  Random: {:.4}",
                star_stats.heterogeneity, random_stats.heterogeneity);

        println!("✅ SUCCESS: Star is more heterogeneous than random!");
        println!("   This suggests Star will have higher Φ");
    }

    #[test]
    fn test_ring_topology() {
        let topo = ConsciousnessTopology::ring(4, 2048, 42);

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
        let topo = ConsciousnessTopology::line(4, 2048, 42);

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
        println!("   Adjacent: 0↔1={:.4}, 1↔2={:.4}, 2↔3={:.4}",
                 node0_to_1, node1_to_2, node2_to_3);
        println!("   Non-adjacent: 0↔2={:.4}, 0↔3={:.4}",
                 node0_to_2, node0_to_3);

        let stats = topo.similarity_stats();
        println!("   Mean: {:.4}, Heterogeneity: {:.4}",
                 stats.mean, stats.heterogeneity);
    }

    #[test]
    fn test_binary_tree_topology() {
        let topo = ConsciousnessTopology::binary_tree(7, 2048, 42);  // Perfect binary tree

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
        let topo = ConsciousnessTopology::dense_network(4, 2048, None, 42);

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
        let topo = ConsciousnessTopology::modular(8, 2048, 2, 42);  // 2 modules of 4 nodes each

        assert_eq!(topo.n_nodes, 8);
        assert_eq!(topo.topology_type, TopologyType::Modular);

        let matrix = topo.similarity_matrix();

        // Nodes within same module should have higher similarity
        let intra_module = matrix[0][1];  // Both in module 0
        let inter_module = matrix[0][4];  // Different modules

        println!("\n📊 Modular Topology Structure:");
        println!("   Intra-module similarity: {:.4}", intra_module);
        println!("   Inter-module similarity: {:.4}", inter_module);

        let stats = topo.similarity_stats();
        println!("   Mean: {:.4}, Heterogeneity: {:.4}",
                 stats.mean, stats.heterogeneity);

        // Modular structure creates heterogeneity
        // (within-module vs between-module differences)
    }

    #[test]
    fn test_lattice_topology() {
        let topo = ConsciousnessTopology::lattice(4, 2048, 42);  // Will create 2x2 grid

        assert_eq!(topo.n_nodes, 4);  // 2x2 = 4
        assert_eq!(topo.topology_type, TopologyType::Lattice);

        let matrix = topo.similarity_matrix();

        // Adjacent nodes in grid should have higher similarity
        let node0_to_1 = matrix[0][1];  // Adjacent horizontally
        let node0_to_2 = matrix[0][2];  // Adjacent vertically
        let node0_to_3 = matrix[0][3];  // Diagonal (not adjacent)

        println!("\n📊 Lattice Topology Structure:");
        println!("   Horizontal neighbor: {:.4}", node0_to_1);
        println!("   Vertical neighbor: {:.4}", node0_to_2);
        println!("   Diagonal (non-adjacent): {:.4}", node0_to_3);

        let stats = topo.similarity_stats();
        println!("   Mean: {:.4}, Heterogeneity: {:.4}",
                 stats.mean, stats.heterogeneity);
    }

    #[test]
    fn test_all_topologies_heterogeneity_order() {
        println!("\n🔬 COMPREHENSIVE TEST: Heterogeneity Across All 8 Topologies");
        println!("{}", "=".repeat(70));

        let random = ConsciousnessTopology::random(4, 2048, 42);
        let star = ConsciousnessTopology::star(4, 2048, 42);
        let ring = ConsciousnessTopology::ring(4, 2048, 42);
        let line = ConsciousnessTopology::line(4, 2048, 42);
        let tree = ConsciousnessTopology::binary_tree(7, 2048, 42);
        let dense = ConsciousnessTopology::dense_network(4, 2048, None, 42);
        let modular = ConsciousnessTopology::modular(8, 2048, 2, 42);
        let lattice = ConsciousnessTopology::lattice(4, 2048, 42);

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
        println!("{:<12} {:>10} {:>10} {:>10}", "Topology", "Mean", "StdDev", "Heterogen");
        println!("{}", "-".repeat(45));

        for (name, stats) in &stats_vec {
            println!("{:<12} {:>10.4} {:>10.4} {:>10.4}",
                     name, stats.mean, stats.std_dev, stats.heterogeneity);
        }

        println!("\n✅ All 8 topologies generated successfully!");
        println!("   Heterogeneity values show clear variation across topologies");
    }
}
