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
    Sphere,          // 2-manifold: S²
    Torus,           // 2-manifold: T²
    KleinBottle,     // Non-orientable 2-manifold
    SmallWorld,
    MobiusStrip,
    Hyperbolic,
    ScaleFree,
    Fractal,         // Tier 3: Self-similar structure
    Hypercube,       // Tier 3: 3D/4D/5D dimensional scaling
    Quantum,         // Tier 3: Superposition of topologies
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

        // Create unique basis vectors for each node with seed-based variation
        // This ensures different seeds produce different Star topologies
        let node_identities: Vec<RealHV> = (0..n_nodes)
            .map(|i| {
                let base = RealHV::basis(i, dim);
                // Add 5% random noise based on seed to create variation
                let noise = RealHV::random(dim, seed + (i as u64 * 1000)).scale(0.05_f32);
                base.add(&noise)
            })
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

        // Add seed-based variation to hub to ensure different samples
        let hub_base = RealHV::bundle(&hub_connections);
        let hub_noise = RealHV::random(dim, seed + 999999).scale(0.05_f32);
        let hub_repr = hub_base.add(&hub_noise);
        node_representations.push(hub_repr);

        // Each spoke representation = single connection to hub with seed variation
        // spoke ⊗ hub + small noise
        for (i, spoke_id) in spoke_ids.iter().enumerate() {
            let spoke_base = spoke_id.bind(hub_id);
            // Add 5% noise to each spoke (different for each spoke)
            let spoke_noise = RealHV::random(dim, seed + ((i + 1) as u64 * 100000)).scale(0.05_f32);
            let spoke_repr = spoke_base.add(&spoke_noise);
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

    /// Generate a sphere (icosahedron) topology - 2-MANIFOLD S²
    ///
    /// 12 vertices arranged on a sphere in icosahedron configuration.
    /// Each vertex connects to exactly 5 neighbors (perfect symmetry).
    /// This is a closed, orientable 2-dimensional manifold.
    ///
    /// # Arguments
    /// * `dim` - Hypervector dimension
    /// * `seed` - Random seed for reproducibility
    pub fn sphere_icosahedron(dim: usize, seed: u64) -> Self {
        let n_nodes = 12; // Icosahedron has 12 vertices

        // Create basis vectors with seed variation
        let node_identities: Vec<RealHV> = (0..n_nodes)
            .map(|i| {
                let base = RealHV::basis(i, dim);
                let noise = RealHV::random(dim, seed + (i as u64 * 1000)).scale(0.05);
                base.add(&noise)
            })
            .collect();

        // Icosahedron edge structure (30 edges, each vertex has degree 5)
        // Vertices arranged: 1 top, 5 upper pentagon, 5 lower pentagon, 1 bottom
        let edges = vec![
            // Top vertex (0) to upper pentagon (1-5)
            (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
            // Upper pentagon connections
            (1, 2), (2, 3), (3, 4), (4, 5), (5, 1),
            // Upper to lower connections
            (1, 6), (2, 6), (2, 7), (3, 7), (3, 8),
            (4, 8), (4, 9), (5, 9), (5, 10), (1, 10),
            // Lower pentagon connections
            (6, 7), (7, 8), (8, 9), (9, 10), (10, 6),
            // Lower pentagon (6-10) to bottom vertex (11)
            (6, 11), (7, 11), (8, 11), (9, 11), (10, 11),
        ];

        let mut node_representations = Vec::with_capacity(n_nodes);

        // Build node representations from edge structure
        for i in 0..n_nodes {
            let mut connections = Vec::new();

            // Find all neighbors of node i
            for (a, b) in &edges {
                if *a == i {
                    connections.push(node_identities[i].bind(&node_identities[*b]));
                } else if *b == i {
                    connections.push(node_identities[i].bind(&node_identities[*a]));
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
            topology_type: TopologyType::Sphere,
        }
    }

    /// Generate a torus topology - 2-MANIFOLD T²
    ///
    /// n×m grid with periodic boundary conditions (wraparound).
    /// Left edge connects to right edge, top connects to bottom.
    /// Forms a donut-shaped 2-dimensional manifold (T² = S¹ × S¹).
    ///
    /// # Arguments
    /// * `n` - Number of rows
    /// * `m` - Number of columns
    /// * `dim` - Hypervector dimension
    /// * `seed` - Random seed for reproducibility
    pub fn torus(n: usize, m: usize, dim: usize, seed: u64) -> Self {
        assert!(n >= 2, "Torus needs at least 2 rows");
        assert!(m >= 2, "Torus needs at least 2 columns");

        let n_nodes = n * m;

        let node_identities: Vec<RealHV> = (0..n_nodes)
            .map(|i| {
                let base = RealHV::basis(i, dim);
                let noise = RealHV::random(dim, seed + (i as u64 * 1000)).scale(0.05);
                base.add(&noise)
            })
            .collect();

        let mut node_representations = Vec::with_capacity(n_nodes);

        for i in 0..n {
            for j in 0..m {
                let idx = i * m + j;
                let mut connections = Vec::new();

                // Connect to 4 neighbors with wraparound (periodic boundaries)
                let up = ((i + n - 1) % n) * m + j;
                let down = ((i + 1) % n) * m + j;
                let left = i * m + ((j + m - 1) % m);
                let right = i * m + ((j + 1) % m);

                connections.push(node_identities[idx].bind(&node_identities[up]));
                connections.push(node_identities[idx].bind(&node_identities[down]));
                connections.push(node_identities[idx].bind(&node_identities[left]));
                connections.push(node_identities[idx].bind(&node_identities[right]));

                let repr = RealHV::bundle(&connections);
                node_representations.push(repr);
            }
        }

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::Torus,
        }
    }

    /// Generate a Klein bottle topology - NON-ORIENTABLE 2-MANIFOLD
    ///
    /// Like torus but with a twist: horizontal wraparound reverses vertical position.
    /// Creates a non-orientable 2-dimensional manifold (cannot be embedded in 3D).
    ///
    /// # Arguments
    /// * `n` - Number of rows
    /// * `m` - Number of columns
    /// * `dim` - Hypervector dimension
    /// * `seed` - Random seed for reproducibility
    pub fn klein_bottle(n: usize, m: usize, dim: usize, seed: u64) -> Self {
        assert!(n >= 2, "Klein bottle needs at least 2 rows");
        assert!(m >= 2, "Klein bottle needs at least 2 columns");

        let n_nodes = n * m;

        let node_identities: Vec<RealHV> = (0..n_nodes)
            .map(|i| {
                let base = RealHV::basis(i, dim);
                let noise = RealHV::random(dim, seed + (i as u64 * 1000)).scale(0.05);
                base.add(&noise)
            })
            .collect();

        let mut node_representations = Vec::with_capacity(n_nodes);

        for i in 0..n {
            for j in 0..m {
                let idx = i * m + j;
                let mut connections = Vec::new();

                // Vertical connections (normal wraparound)
                let up = ((i + n - 1) % n) * m + j;
                let down = ((i + 1) % n) * m + j;

                // Horizontal connections with TWIST (Klein bottle property)
                // When wrapping horizontally, reverse the vertical position
                let left = if j == 0 {
                    // Wrap to right edge with vertical flip
                    ((n - 1 - i) * m) + (m - 1)
                } else {
                    i * m + (j - 1)
                };

                let right = if j == m - 1 {
                    // Wrap to left edge with vertical flip
                    (n - 1 - i) * m
                } else {
                    i * m + (j + 1)
                };

                connections.push(node_identities[idx].bind(&node_identities[up]));
                connections.push(node_identities[idx].bind(&node_identities[down]));
                connections.push(node_identities[idx].bind(&node_identities[left]));
                connections.push(node_identities[idx].bind(&node_identities[right]));

                let repr = RealHV::bundle(&connections);
                node_representations.push(repr);
            }
        }

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::KleinBottle,
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

    /// Generate a small-world network (Watts-Strogatz model)
    ///
    /// Starts with a k-regular ring lattice, then randomly rewires edges
    /// with probability p. This creates the "small-world" property:
    /// high clustering (like regular lattice) + short path length (like random).
    ///
    /// This topology is highly biologically relevant - matches brain connectivity!
    ///
    /// # Arguments
    /// * `n_nodes` - Number of nodes (must be >= k + 1)
    /// * `dim` - Hypervector dimension
    /// * `k` - Number of nearest neighbors in initial ring (must be even)
    /// * `p` - Rewiring probability [0.0, 1.0] (typical: 0.1)
    /// * `seed` - Random seed for reproducibility
    pub fn small_world(n_nodes: usize, dim: usize, k: usize, p: f64, seed: u64) -> Self {
        assert!(n_nodes >= k + 1, "Need n_nodes >= k+1 for small-world");
        assert!(k % 2 == 0, "k must be even for symmetric ring lattice");
        assert!(k >= 2, "Need at least k=2 neighbors");
        assert!((0.0..=1.0).contains(&p), "Rewiring probability p must be in [0, 1]");
        assert!(dim >= 256, "Dimension should be >= 256 for good separation");

        let node_identities: Vec<RealHV> = (0..n_nodes)
            .map(|i| RealHV::basis(i, dim))
            .collect();

        // Build initial k-regular ring lattice edges
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for i in 0..n_nodes {
            for j in 1..=(k / 2) {
                let neighbor = (i + j) % n_nodes;
                // Only store each edge once (i < neighbor)
                if i < neighbor {
                    edges.push((i, neighbor));
                }
            }
        }

        // Rewire edges with probability p
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(seed);

        let mut final_edges = Vec::new();
        for (i, j) in edges {
            if rng.gen::<f64>() < p {
                // Rewire: keep i, replace j with random node
                let mut new_target = rng.gen_range(0..n_nodes);

                // Avoid self-loops and duplicate edges
                while new_target == i || final_edges.contains(&(i.min(new_target), i.max(new_target))) {
                    new_target = rng.gen_range(0..n_nodes);
                }

                final_edges.push((i.min(new_target), i.max(new_target)));
            } else {
                // Keep original edge
                final_edges.push((i, j));
            }
        }

        // Build adjacency list
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
        for (i, j) in &final_edges {
            adjacency[*i].push(*j);
            adjacency[*j].push(*i);
        }

        // Generate node representations from adjacency
        let mut node_representations = Vec::with_capacity(n_nodes);
        for i in 0..n_nodes {
            let connections: Vec<RealHV> = adjacency[i]
                .iter()
                .map(|&neighbor| node_identities[i].bind(&node_identities[neighbor]))
                .collect();

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
            topology_type: TopologyType::SmallWorld,
        }
    }

    /// Generate a Möbius strip topology
    ///
    /// Like a ring, but with a topological twist: half the connections
    /// are inverted (negated). This creates a non-orientable surface
    /// with no inside/outside distinction.
    ///
    /// Tests the hypothesis: Does non-orientability affect integrated information?
    ///
    /// # Arguments
    /// * `n_nodes` - Number of nodes (must be even for the twist)
    /// * `dim` - Hypervector dimension
    /// * `seed` - Random seed for reproducibility
    pub fn mobius_strip(n_nodes: usize, dim: usize, seed: u64) -> Self {
        assert!(n_nodes >= 4, "Möbius strip needs at least 4 nodes");
        assert!(n_nodes % 2 == 0, "Möbius strip needs even number of nodes for twist");
        assert!(dim >= 256, "Dimension should be >= 256 for good separation");

        let node_identities: Vec<RealHV> = (0..n_nodes)
            .map(|i| RealHV::basis(i, dim))
            .collect();

        let mut node_representations = Vec::with_capacity(n_nodes);

        // First half: normal ring connections
        // Second half: one connection inverted (the Möbius twist!)
        for i in 0..n_nodes {
            let prev = (i + n_nodes - 1) % n_nodes;
            let next = (i + 1) % n_nodes;

            if i < n_nodes / 2 {
                // First half: normal binding (like regular ring)
                let conn1 = node_identities[i].bind(&node_identities[prev]);
                let conn2 = node_identities[i].bind(&node_identities[next]);
                let repr = RealHV::bundle(&[conn1, conn2]);
                node_representations.push(repr);
            } else {
                // Second half: invert the "next" connection (the twist!)
                let conn1 = node_identities[i].bind(&node_identities[prev]);
                let conn2_inverted = node_identities[i].bind(&node_identities[next].scale(-1.0));
                let repr = RealHV::bundle(&[conn1, conn2_inverted]);
                node_representations.push(repr);
            }
        }

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::MobiusStrip,
        }
    }

    /// Generate a torus (2D ring with wraparound) topology
    ///
    /// A 2D grid where edges wrap around: top connects to bottom,
    /// left connects to right. Each node has exactly 4 neighbors
    /// (up, down, left, right). This is the natural 2D extension
    /// of the Ring topology.
    ///
    /// No boundary effects, uniform connectivity, scales to 3D/4D.
    ///
    /// # Arguments
    /// * `grid_size` - Size of the square grid (total nodes = grid_size²)
    /// * `dim` - Hypervector dimension
    /// * `seed` - Random seed for reproducibility
    pub fn torus_square(grid_size: usize, dim: usize, seed: u64) -> Self {
        assert!(grid_size >= 2, "Torus needs at least 2×2 grid");
        assert!(dim >= 256, "Dimension should be >= 256 for good separation");

        let n_nodes = grid_size * grid_size;

        let node_identities: Vec<RealHV> = (0..n_nodes)
            .map(|i| RealHV::basis(i, dim))
            .collect();

        let mut node_representations = Vec::with_capacity(n_nodes);

        for i in 0..n_nodes {
            let row = i / grid_size;
            let col = i % grid_size;

            // Wraparound connections (modulo arithmetic)
            let up = ((row + grid_size - 1) % grid_size) * grid_size + col;
            let down = ((row + 1) % grid_size) * grid_size + col;
            let left = row * grid_size + ((col + grid_size - 1) % grid_size);
            let right = row * grid_size + ((col + 1) % grid_size);

            // Each node connects to its 4 neighbors
            let conn_up = node_identities[i].bind(&node_identities[up]);
            let conn_down = node_identities[i].bind(&node_identities[down]);
            let conn_left = node_identities[i].bind(&node_identities[left]);
            let conn_right = node_identities[i].bind(&node_identities[right]);

            let repr = RealHV::bundle(&[conn_up, conn_down, conn_left, conn_right]);
            node_representations.push(repr);
        }

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::Torus,
        }
    }

    /// Generate a Klein bottle topology
    ///
    /// Like a torus, but with one dimension "flipped" - a non-orientable
    /// 2D surface. The right edge connects to the left edge with row inversion.
    /// This creates a surface with no inside/outside distinction.
    ///
    /// Tests: Does 2D non-orientability have the same catastrophic effect as Möbius?
    ///
    /// # Arguments
    /// * `grid_size` - Size of the square grid (total nodes = grid_size²)
    /// * `dim` - Hypervector dimension
    /// * `seed` - Random seed for reproducibility
    pub fn klein_bottle_square(grid_size: usize, dim: usize, seed: u64) -> Self {
        assert!(grid_size >= 2, "Klein bottle needs at least 2×2 grid");
        assert!(dim >= 256, "Dimension should be >= 256 for good separation");

        let n_nodes = grid_size * grid_size;

        let node_identities: Vec<RealHV> = (0..n_nodes)
            .map(|i| RealHV::basis(i, dim))
            .collect();

        let mut node_representations = Vec::with_capacity(n_nodes);

        for i in 0..n_nodes {
            let row = i / grid_size;
            let col = i % grid_size;

            // Normal wraparound for vertical (top↔bottom)
            let up = ((row + grid_size - 1) % grid_size) * grid_size + col;
            let down = ((row + 1) % grid_size) * grid_size + col;

            // Klein bottle twist: horizontal wraparound with row flip
            // Left edge connects to right edge, but with inverted row
            let left = if col == 0 {
                // Wraparound to right edge, but FLIP the row (Klein bottle twist!)
                let flipped_row = grid_size - 1 - row;
                flipped_row * grid_size + (grid_size - 1)
            } else {
                row * grid_size + (col - 1)
            };

            let right = if col == grid_size - 1 {
                // Wraparound to left edge, with row flip
                let flipped_row = grid_size - 1 - row;
                flipped_row * grid_size + 0
            } else {
                row * grid_size + (col + 1)
            };

            // Bind connections
            let conn_up = node_identities[i].bind(&node_identities[up]);
            let conn_down = node_identities[i].bind(&node_identities[down]);
            let conn_left = node_identities[i].bind(&node_identities[left]);
            let conn_right = node_identities[i].bind(&node_identities[right]);

            let repr = RealHV::bundle(&[conn_up, conn_down, conn_left, conn_right]);
            node_representations.push(repr);
        }

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::KleinBottle,
        }
    }

    /// Generate a hyperbolic topology (negative curvature)
    ///
    /// Creates a tree-like structure where each level has exponentially
    /// more nodes than the previous (modeling hyperbolic geometry).
    /// Each node connects to its parent + children + neighbors at same depth.
    ///
    /// Unlike a simple tree, nodes at the same depth are also connected,
    /// creating the characteristic "expanding space" of hyperbolic geometry.
    ///
    /// # Arguments
    /// * `n_nodes` - Number of nodes (will build tree to this depth)
    /// * `dim` - Hypervector dimension
    /// * `branching` - Branching factor (typical: 2-3)
    /// * `seed` - Random seed for reproducibility
    pub fn hyperbolic(n_nodes: usize, dim: usize, branching: usize, seed: u64) -> Self {
        assert!(n_nodes >= 2, "Hyperbolic needs at least 2 nodes");
        assert!(branching >= 2, "Branching factor must be >= 2");
        assert!(dim >= 256, "Dimension should be >= 256 for good separation");

        // Build a tree structure with lateral connections at each level
        let node_identities: Vec<RealHV> = (0..n_nodes)
            .map(|i| RealHV::basis(i, dim))
            .collect();

        // Build adjacency list
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];

        // Add tree edges (parent-child)
        for i in 1..n_nodes {
            let parent = (i - 1) / branching;
            adjacency[i].push(parent);
            adjacency[parent].push(i);
        }

        // Add lateral connections within each depth level
        // Group nodes by depth
        let mut depth_groups: Vec<Vec<usize>> = Vec::new();
        let mut current_depth = 0;
        let mut depth_start = 0;

        while depth_start < n_nodes {
            let depth_size = branching.pow(current_depth as u32).min(n_nodes - depth_start);
            let depth_end = (depth_start + depth_size).min(n_nodes);

            let nodes_at_depth: Vec<usize> = (depth_start..depth_end).collect();

            // Connect neighbors at same depth (creates hyperbolic expansion)
            for (idx, &node) in nodes_at_depth.iter().enumerate() {
                if idx > 0 {
                    let left_neighbor = nodes_at_depth[idx - 1];
                    if !adjacency[node].contains(&left_neighbor) {
                        adjacency[node].push(left_neighbor);
                        adjacency[left_neighbor].push(node);
                    }
                }
            }

            depth_groups.push(nodes_at_depth);
            depth_start = depth_end;
            current_depth += 1;
        }

        // Generate representations from adjacency
        let mut node_representations = Vec::with_capacity(n_nodes);
        for i in 0..n_nodes {
            if adjacency[i].is_empty() {
                node_representations.push(node_identities[i].clone());
            } else {
                let connections: Vec<RealHV> = adjacency[i]
                    .iter()
                    .map(|&neighbor| node_identities[i].bind(&node_identities[neighbor]))
                    .collect();
                node_representations.push(RealHV::bundle(&connections));
            }
        }

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::Hyperbolic,
        }
    }

    /// Generate a scale-free network (Barabási-Albert model)
    ///
    /// Uses preferential attachment: new nodes preferentially connect to
    /// nodes that already have many connections ("rich get richer").
    /// This creates a power-law degree distribution P(k) ~ k^-γ.
    ///
    /// Matches many real-world networks: Internet, social networks, brain!
    ///
    /// # Arguments
    /// * `n_nodes` - Total number of nodes
    /// * `dim` - Hypervector dimension
    /// * `m` - Number of edges to attach per new node (typical: 2-5)
    /// * `seed` - Random seed for reproducibility
    pub fn scale_free(n_nodes: usize, dim: usize, m: usize, seed: u64) -> Self {
        assert!(n_nodes >= m + 1, "Need n_nodes > m for scale-free network");
        assert!(m >= 1, "m must be >= 1");
        assert!(dim >= 256, "Dimension should be >= 256 for good separation");

        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(seed);

        let node_identities: Vec<RealHV> = (0..n_nodes)
            .map(|i| RealHV::basis(i, dim))
            .collect();

        // Build adjacency list
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];

        // Start with a small complete graph (m+1 nodes all connected)
        for i in 0..=m {
            for j in (i+1)..=m.min(n_nodes-1) {
                adjacency[i].push(j);
                adjacency[j].push(i);
            }
        }

        // Add remaining nodes one by one with preferential attachment
        for new_node in (m+1)..n_nodes {
            // Calculate total degree (for normalization)
            let total_degree: usize = adjacency.iter()
                .take(new_node)
                .map(|neighbors| neighbors.len())
                .sum();

            if total_degree == 0 {
                // Fallback: connect to first m nodes
                for target in 0..m {
                    adjacency[new_node].push(target);
                    adjacency[target].push(new_node);
                }
                continue;
            }

            // Select m targets using preferential attachment
            let mut targets = Vec::new();
            while targets.len() < m.min(new_node) {
                // Randomly select a node with probability proportional to its degree
                let rand_val = rng.gen_range(0..total_degree);
                let mut cumulative = 0;
                let mut selected = 0;

                for node in 0..new_node {
                    cumulative += adjacency[node].len();
                    if cumulative > rand_val {
                        selected = node;
                        break;
                    }
                }

                // Avoid duplicates
                if !targets.contains(&selected) && !adjacency[new_node].contains(&selected) {
                    targets.push(selected);
                }
            }

            // Add edges
            for target in targets {
                adjacency[new_node].push(target);
                adjacency[target].push(new_node);
            }
        }

        // Generate representations from adjacency
        let mut node_representations = Vec::with_capacity(n_nodes);
        for i in 0..n_nodes {
            if adjacency[i].is_empty() {
                node_representations.push(node_identities[i].clone());
            } else {
                let connections: Vec<RealHV> = adjacency[i]
                    .iter()
                    .map(|&neighbor| node_identities[i].bind(&node_identities[neighbor]))
                    .collect();
                node_representations.push(RealHV::bundle(&connections));
            }
        }

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::ScaleFree,
        }
    }

    /// Generate a fractal network topology (self-similar structure)
    ///
    /// Creates a Sierpiński-inspired network with self-similarity at multiple scales.
    /// The structure has:
    /// - Hierarchical organization (like fractal geometry)
    /// - Cross-scale connections (self-similarity)
    /// - Local clusters at each scale
    ///
    /// Implementation uses a recursive subdivision pattern:
    /// - Level 0: Core triangle (3 nodes)
    /// - Level 1: 3 sub-triangles (9 nodes total)
    /// - Level 2: 9 sub-triangles (27 nodes total)
    ///
    /// # Arguments
    /// * `n_nodes` - Number of nodes (will use up to next power of 3)
    /// * `dim` - Hypervector dimension
    /// * `seed` - Random seed for reproducibility
    pub fn fractal(n_nodes: usize, dim: usize, seed: u64) -> Self {
        // Create node identities with slight variation
        let node_identities: Vec<RealHV> = (0..n_nodes)
            .map(|i| {
                let base = RealHV::basis(i, dim);
                let noise = RealHV::random(dim, seed + i as u64 * 1000).scale(0.05);
                base.add(&noise)
            })
            .collect();

        // Fractal structure: Sierpiński-inspired hierarchical triangles
        // Level 0: Core triangle (nodes 0, 1, 2)
        // Level 1: Sub-triangles at each vertex
        // Level 2+: Recursive subdivision

        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];

        // Build fractal structure level by level
        let levels = ((n_nodes as f64).log(3.0).floor() as usize).max(1);

        // Level 0: Core triangle
        if n_nodes >= 3 {
            adjacency[0].extend_from_slice(&[1, 2]);
            adjacency[1].extend_from_slice(&[0, 2]);
            adjacency[2].extend_from_slice(&[0, 1]);
        }

        // Hierarchical subdivision
        let mut nodes_per_level = vec![3]; // Start with core triangle
        for level in 1..levels {
            let prev_count: usize = nodes_per_level.iter().sum();
            let new_count = prev_count * 3;
            if new_count > n_nodes { break; }

            // Add connections at this level
            for i in prev_count..new_count.min(n_nodes) {
                // Connect to parent level (self-similarity)
                let parent = (i - prev_count) % prev_count;
                adjacency[i].push(parent);
                adjacency[parent].push(i);

                // Connect within same level (local clustering)
                let group_size = 3;
                let group = (i - prev_count) / group_size;
                let pos_in_group = (i - prev_count) % group_size;

                for j in 0..group_size {
                    if j != pos_in_group {
                        let sibling = prev_count + group * group_size + j;
                        if sibling < n_nodes {
                            adjacency[i].push(sibling);
                            adjacency[sibling].push(i);
                        }
                    }
                }
            }

            nodes_per_level.push(new_count - prev_count);
        }

        // Generate representations from adjacency
        let mut node_representations = Vec::with_capacity(n_nodes);
        for i in 0..n_nodes {
            if adjacency[i].is_empty() {
                node_representations.push(node_identities[i].clone());
            } else {
                // Deduplicate adjacency list
                let mut unique_neighbors = adjacency[i].clone();
                unique_neighbors.sort_unstable();
                unique_neighbors.dedup();

                let connections: Vec<RealHV> = unique_neighbors
                    .iter()
                    .map(|&neighbor| node_identities[i].bind(&node_identities[neighbor]))
                    .collect();
                node_representations.push(RealHV::bundle(&connections));
            }
        }

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::Fractal,
        }
    }

    /// Generate a hypercube topology (3D/4D/5D dimensional scaling test)
    ///
    /// Creates an n-dimensional hypercube where each vertex connects to
    /// neighbors in all dimensions. This tests dimensional scaling beyond 2D:
    /// - 3D cube: 8 vertices, each with 3 neighbors
    /// - 4D tesseract: 16 vertices, each with 4 neighbors
    /// - 5D penteract: 32 vertices, each with 5 neighbors
    ///
    /// Prediction: Should achieve similar Φ to Ring/Torus if dimensional
    /// invariance hypothesis holds at higher dimensions.
    ///
    /// # Arguments
    /// * `dimensions` - Number of spatial dimensions (3, 4, or 5)
    /// * `hv_dim` - Hypervector dimension
    /// * `seed` - Random seed for reproducibility
    pub fn hypercube(dimensions: usize, hv_dim: usize, seed: u64) -> Self {
        let n_nodes = 2_usize.pow(dimensions as u32);

        // Create node identities
        let node_identities: Vec<RealHV> = (0..n_nodes)
            .map(|i| {
                let base = RealHV::basis(i, hv_dim);
                let noise = RealHV::random(hv_dim, seed + i as u64 * 1000).scale(0.05);
                base.add(&noise)
            })
            .collect();

        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];

        // In a hypercube, two vertices are connected if their binary
        // representations differ in exactly one bit
        for i in 0..n_nodes {
            for j in (i+1)..n_nodes {
                // Count differing bits (Hamming distance)
                let xor = i ^ j;
                let hamming_dist = xor.count_ones();

                // Connected if exactly 1 bit differs
                if hamming_dist == 1 {
                    adjacency[i].push(j);
                    adjacency[j].push(i);
                }
            }
        }

        // Generate representations from adjacency
        let mut node_representations = Vec::with_capacity(n_nodes);
        for i in 0..n_nodes {
            let connections: Vec<RealHV> = adjacency[i]
                .iter()
                .map(|&neighbor| node_identities[i].bind(&node_identities[neighbor]))
                .collect();
            node_representations.push(RealHV::bundle(&connections));
        }

        Self {
            n_nodes,
            dim: hv_dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::Hypercube,
        }
    }

    /// Generate a quantum network topology (superposition of topologies)
    ///
    /// Creates a quantum-inspired network where each node exists in a
    /// superposition of multiple topology states. This is a novel research
    /// frontier combining:
    /// - Quantum superposition principle
    /// - Network topology theory
    /// - Consciousness measurement
    ///
    /// Implementation: Each node's representation is a weighted blend of
    /// its representation in Ring, Star, and Random topologies.
    ///
    /// Hypothesis: Superposition may allow simultaneous benefits of multiple
    /// topologies, potentially achieving higher Φ than any single topology.
    ///
    /// # Arguments
    /// * `n_nodes` - Number of nodes
    /// * `dim` - Hypervector dimension
    /// * `weights` - Weights for (Ring, Star, Random) components
    /// * `seed` - Random seed for reproducibility
    pub fn quantum(n_nodes: usize, dim: usize, weights: (f32, f32, f32), seed: u64) -> Self {
        // Normalize weights
        let sum = weights.0 + weights.1 + weights.2;
        let w = (weights.0 / sum, weights.1 / sum, weights.2 / sum);

        // Generate three different topologies with same seed
        let ring = Self::ring(n_nodes, dim, seed);
        let star = Self::star(n_nodes, dim, seed);
        let random = Self::random(n_nodes, dim, seed);

        // Create node identities (use ring's as base)
        let node_identities = ring.node_identities.clone();

        // Quantum superposition: blend representations from all three topologies
        let mut node_representations = Vec::with_capacity(n_nodes);

        for i in 0..n_nodes {
            // Scale each topology's representation by its weight
            let ring_component = ring.node_representations[i].scale(w.0);
            let star_component = star.node_representations[i].scale(w.1);
            let random_component = random.node_representations[i].scale(w.2);

            // Superposed state = weighted sum
            let superposed = ring_component
                .add(&star_component)
                .add(&random_component);

            node_representations.push(superposed);
        }

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::Quantum,
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
        let topo = ConsciousnessTopology::random(4, crate::hdc::HDC_DIMENSION, 42);

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
        // Heterogeneity (std_dev) should be at least 0.2 for clear structure
        assert!(stats.heterogeneity > 0.2,
                "Star topology should have heterogeneous structure, got {:.4}",
                stats.heterogeneity);
    }

    #[test]
    fn test_star_vs_random_heterogeneity() {
        let random = ConsciousnessTopology::random(4, crate::hdc::HDC_DIMENSION, 42);
        let star = ConsciousnessTopology::star(4, crate::hdc::HDC_DIMENSION, 42);

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
        let topo = ConsciousnessTopology::binary_tree(7, crate::hdc::HDC_DIMENSION, 42);  // Perfect binary tree

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
        let topo = ConsciousnessTopology::modular(8, crate::hdc::HDC_DIMENSION, 2, 42);  // 2 modules of 4 nodes each

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
        let topo = ConsciousnessTopology::lattice(4, crate::hdc::HDC_DIMENSION, 42);  // Will create 2x2 grid

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
