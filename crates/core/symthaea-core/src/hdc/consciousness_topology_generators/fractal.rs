// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Tier 3 topology generators: Fractal and complex topologies + analysis.
//!
//! ScaleFree, Fractal, Hypercube, Sierpinski, FractalTree, Koch, Quantum.
//! Also includes similarity analysis methods.

use super::super::unified_hv::ContinuousHV;
use super::types::{ConsciousnessTopology, SimilarityStats, TopologyType};

impl ConsciousnessTopology {
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
        assert!(n_nodes > m, "Need n_nodes > m for scale-free network");
        assert!(m >= 1, "m must be >= 1");
        assert!(dim >= 256, "Dimension should be >= 256 for good separation");

        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(seed);

        let node_identities: Vec<ContinuousHV> =
            (0..n_nodes).map(|i| ContinuousHV::basis(i, dim)).collect();

        // Build adjacency list
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];

        // Start with a small complete graph (m+1 nodes all connected)
        for i in 0..=m {
            for j in (i + 1)..=m.min(n_nodes - 1) {
                adjacency[i].push(j);
                adjacency[j].push(i);
            }
        }

        // Add remaining nodes one by one with preferential attachment
        for new_node in (m + 1)..n_nodes {
            // Calculate total degree (for normalization)
            let total_degree: usize = adjacency
                .iter()
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
                let connections: Vec<ContinuousHV> = adjacency[i]
                    .iter()
                    .map(|&neighbor| node_identities[i].bind(&node_identities[neighbor]))
                    .collect();
                node_representations.push(ContinuousHV::bundle_owned(&connections));
            }
        }

        // Extract edges from adjacency (only i < j to avoid duplicates)
        let mut edges = Vec::new();
        for i in 0..n_nodes {
            for &j in &adjacency[i] {
                if i < j {
                    edges.push((i, j));
                }
            }
        }

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::ScaleFree,
            edges,
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
        let node_identities: Vec<ContinuousHV> = (0..n_nodes)
            .map(|i| {
                let base = ContinuousHV::basis(i, dim);
                let noise = ContinuousHV::random(dim, seed + i as u64 * 1000).scale(0.05);
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
        for _level in 1..levels {
            let prev_count: usize = nodes_per_level.iter().sum();
            let new_count = prev_count * 3;
            if new_count > n_nodes {
                break;
            }

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

                let connections: Vec<ContinuousHV> = unique_neighbors
                    .iter()
                    .map(|&neighbor| node_identities[i].bind(&node_identities[neighbor]))
                    .collect();
                node_representations.push(ContinuousHV::bundle_owned(&connections));
            }
        }

        // Extract edges from adjacency (deduplicated, only i < j)
        let mut edges = Vec::new();
        for i in 0..n_nodes {
            let mut unique_neighbors = adjacency[i].clone();
            unique_neighbors.sort_unstable();
            unique_neighbors.dedup();
            for j in unique_neighbors {
                if i < j {
                    edges.push((i, j));
                }
            }
        }

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::Fractal,
            edges,
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
        let node_identities: Vec<ContinuousHV> = (0..n_nodes)
            .map(|i| {
                let base = ContinuousHV::basis(i, hv_dim);
                let noise = ContinuousHV::random(hv_dim, seed + i as u64 * 1000).scale(0.05);
                base.add(&noise)
            })
            .collect();

        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];

        // In a hypercube, two vertices are connected if their binary
        // representations differ in exactly one bit
        for i in 0..n_nodes {
            for j in (i + 1)..n_nodes {
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
            let connections: Vec<ContinuousHV> = adjacency[i]
                .iter()
                .map(|&neighbor| node_identities[i].bind(&node_identities[neighbor]))
                .collect();
            node_representations.push(ContinuousHV::bundle_owned(&connections));
        }

        // Extract edges from adjacency (only i < j to avoid duplicates)
        let mut edges = Vec::new();
        for i in 0..n_nodes {
            for &j in &adjacency[i] {
                if i < j {
                    edges.push((i, j));
                }
            }
        }

        Self {
            n_nodes,
            dim: hv_dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::Hypercube,
            edges,
        }
    }

    /// Generate a Sierpinski Gasket topology (fractal triangle, d≈1.585)
    ///
    /// Creates a hierarchical triangular fractal network with self-similar
    /// structure at multiple scales. The fractal dimension is approximately
    /// 1.585 (log(3)/log(2)), placing it between 1D (Ring) and 2D (Torus).
    ///
    /// Structure:
    /// - Depth 0: 3 vertices (single triangle)
    /// - Depth 1: 6 vertices (4 sub-triangles, center removed)
    /// - Depth 2: 12 vertices (16 sub-triangles, centers removed)
    /// - Depth 3: 42 vertices (self-similar pattern continues)
    ///
    /// Each node connects to:
    /// - Its neighbors within the same triangle
    /// - The parent/child node in the hierarchical structure
    ///
    /// Hypothesis: Fractal dimension d≈1.585 should produce Φ between
    /// Ring (1D, Φ≈0.4954) and Torus (2D, Φ≈0.4954), if Φ is a continuous
    /// function of dimension. However, self-similarity across scales may
    /// provide additional integration benefits.
    ///
    /// # Arguments
    /// * `depth` - Iteration depth (0-5 recommended, produces 3·3^depth vertices)
    /// * `dim` - Hypervector dimension
    /// * `seed` - Random seed for reproducibility
    pub fn sierpinski_gasket(depth: usize, dim: usize, seed: u64) -> Self {
        assert!(depth <= 6, "Depth > 6 creates too many nodes (>2000)");
        assert!(dim >= 256, "Dimension should be >= 256 for good separation");

        // Calculate number of nodes
        // Sierpinski gasket: 3 * (3^depth - 1) / 2 + 3 vertices
        let n_nodes = if depth == 0 {
            3
        } else {
            3 * (3_usize.pow(depth as u32) - 1) / 2 + 3
        };

        // Create node identities with variation
        let node_identities: Vec<ContinuousHV> = (0..n_nodes)
            .map(|i| {
                let base = ContinuousHV::basis(i, dim);
                let noise = ContinuousHV::random(dim, seed + i as u64 * 1000).scale(0.05);
                base.add(&noise)
            })
            .collect();

        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];

        // Build hierarchical structure
        // Level 0: Core triangle (nodes 0, 1, 2)
        if n_nodes >= 3 {
            adjacency[0].extend_from_slice(&[1, 2]);
            adjacency[1].extend_from_slice(&[0, 2]);
            adjacency[2].extend_from_slice(&[0, 1]);
        }

        if depth > 0 {
            // Recursive subdivision
            let mut current_node = 3;

            for d in 1..=depth {
                let triangles_at_level = 3_usize.pow(d as u32);
                let nodes_per_triangle = 3;

                for tri in 0..triangles_at_level {
                    if current_node + nodes_per_triangle > n_nodes {
                        break;
                    }

                    // Create sub-triangle
                    let a = current_node;
                    let b = current_node + 1;
                    let c = current_node + 2;

                    // Connect triangle vertices to each other
                    adjacency[a].extend_from_slice(&[b, c]);
                    adjacency[b].extend_from_slice(&[a, c]);
                    adjacency[c].extend_from_slice(&[a, b]);

                    // Connect to parent triangle (hierarchical connection)
                    let parent = tri / 3;
                    if parent < current_node {
                        adjacency[a].push(parent);
                        adjacency[parent].push(a);
                    }

                    // Connect to sibling triangles (self-similarity)
                    let sibling_offset = tri % 3;
                    if sibling_offset > 0 && current_node >= nodes_per_triangle {
                        let sibling = current_node - nodes_per_triangle;
                        adjacency[a].push(sibling);
                        adjacency[sibling].push(a);
                    }

                    current_node += nodes_per_triangle;
                }
            }
        }

        // Generate representations from adjacency
        let mut node_representations = Vec::with_capacity(n_nodes);
        for i in 0..n_nodes {
            if adjacency[i].is_empty() {
                node_representations.push(node_identities[i].clone());
            } else {
                // Remove duplicates
                let mut unique_neighbors = adjacency[i].clone();
                unique_neighbors.sort_unstable();
                unique_neighbors.dedup();

                let connections: Vec<ContinuousHV> = unique_neighbors
                    .iter()
                    .map(|&neighbor| node_identities[i].bind(&node_identities[neighbor]))
                    .collect();
                node_representations.push(ContinuousHV::bundle_owned(&connections));
            }
        }

        // Extract edges from adjacency (deduplicated, only i < j)
        let mut edges = Vec::new();
        for i in 0..n_nodes {
            let mut unique_neighbors = adjacency[i].clone();
            unique_neighbors.sort_unstable();
            unique_neighbors.dedup();
            for j in unique_neighbors {
                if i < j {
                    edges.push((i, j));
                }
            }
        }

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::SierpinskiGasket,
            edges,
        }
    }

    /// Generate a Fractal Tree topology (self-similar hierarchical branching)
    ///
    /// Creates a self-similar tree structure where each branch subdivides
    /// into multiple sub-branches at all scales. Unlike a regular binary tree,
    /// the fractal tree maintains the same branching pattern at every level.
    ///
    /// Structure:
    /// - Root node (1 node)
    /// - Level 1: `branching` children
    /// - Level 2: `branching²` children
    /// - Level d: `branching^d` children
    ///
    /// Total nodes = (branching^(depth+1) - 1) / (branching - 1)
    ///
    /// Hypothesis: Self-similar branching may achieve higher Φ than regular
    /// binary tree due to uniform structure at all scales. If branching=2,
    /// should closely match binary tree baseline (Φ≈0.4668). Higher branching
    /// factors may reduce integration due to increased hub centrality.
    ///
    /// Biological relevance: Neural dendritic trees, vascular networks, and
    /// bronchial trees all exhibit fractal branching patterns.
    ///
    /// # Arguments
    /// * `depth` - Tree depth (0-4 recommended for branching=2)
    /// * `branching` - Number of children per node (2-4 typical)
    /// * `dim` - Hypervector dimension
    /// * `seed` - Random seed for reproducibility
    pub fn fractal_tree(depth: usize, branching: usize, dim: usize, seed: u64) -> Self {
        assert!(branching >= 2, "Branching factor must be >= 2");
        assert!(depth <= 8, "Depth too large, would create too many nodes");
        assert!(dim >= 256, "Dimension should be >= 256 for good separation");

        // Calculate number of nodes
        // Geometric series: sum_{i=0}^{depth} branching^i = (branching^(depth+1) - 1) / (branching - 1)
        let n_nodes = branching
            .checked_pow((depth + 1) as u32)
            .and_then(|p| p.checked_sub(1))
            .map(|v| v / (branching - 1))
            .expect("Branching^(depth+1) overflows usize — reduce depth or branching");

        assert!(
            n_nodes <= 10000,
            "Too many nodes ({n_nodes}). Reduce depth or branching factor."
        );

        // Create node identities with variation
        let node_identities: Vec<ContinuousHV> = (0..n_nodes)
            .map(|i| {
                let base = ContinuousHV::basis(i, dim);
                let noise = ContinuousHV::random(dim, seed + i as u64 * 1000).scale(0.05);
                base.add(&noise)
            })
            .collect();

        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];

        // Build tree structure
        // Node 0 is the root
        // Nodes 1..branching are level 1
        // Nodes branching..(branching + branching²) are level 2, etc.

        for node in 0..n_nodes {
            // Calculate children for this node
            let first_child = node * branching + 1;

            for child_offset in 0..branching {
                let child = first_child + child_offset;

                if child < n_nodes {
                    // Parent → child connection
                    adjacency[node].push(child);
                    adjacency[child].push(node);

                    // Optional: Connect siblings at each level for local clustering
                    // This makes it more brain-like (dendritic trees have crosslinks)
                    for sibling_offset in 0..branching {
                        if sibling_offset != child_offset {
                            let sibling = first_child + sibling_offset;
                            if sibling < n_nodes {
                                adjacency[child].push(sibling);
                                adjacency[sibling].push(child);
                            }
                        }
                    }
                }
            }
        }

        // Generate representations from adjacency
        let mut node_representations = Vec::with_capacity(n_nodes);
        for i in 0..n_nodes {
            if adjacency[i].is_empty() {
                // Leaf node with no connections
                node_representations.push(node_identities[i].clone());
            } else {
                // Remove duplicates
                let mut unique_neighbors = adjacency[i].clone();
                unique_neighbors.sort_unstable();
                unique_neighbors.dedup();

                let connections: Vec<ContinuousHV> = unique_neighbors
                    .iter()
                    .map(|&neighbor| node_identities[i].bind(&node_identities[neighbor]))
                    .collect();
                node_representations.push(ContinuousHV::bundle_owned(&connections));
            }
        }

        // Extract edges from adjacency (deduplicated, only i < j)
        let mut edges = Vec::new();
        for i in 0..n_nodes {
            let mut unique_neighbors = adjacency[i].clone();
            unique_neighbors.sort_unstable();
            unique_neighbors.dedup();
            for j in unique_neighbors {
                if i < j {
                    edges.push((i, j));
                }
            }
        }

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::FractalTree,
            edges,
        }
    }

    /// Generate a Koch Snowflake topology (fractal curve, d ≈ 1.262)
    ///
    /// The Koch Snowflake is a famous fractal curve where each line segment
    /// is replaced by 4 segments forming a triangular "bump". This creates
    /// a self-similar structure with fractal dimension log(4)/log(3) ≈ 1.262.
    ///
    /// Structure:
    /// - Depth 0: Triangle with 3 nodes (boundary path)
    /// - Depth 1: 12 segments, 12 nodes
    /// - Depth 2: 48 segments, 48 nodes
    /// - Depth n: 3 * 4^n segments
    ///
    /// Nodes are connected along the curve (path-like connectivity), but
    /// the self-similar structure creates hierarchical relationships at
    /// multiple scales.
    ///
    /// Hypothesis: The infinite perimeter in finite area creates unique
    /// integration properties - local connectivity but fractal self-similarity.
    ///
    /// # Arguments
    /// * `depth` - Recursion depth (0-5 recommended, 5 gives 3072 nodes)
    /// * `dim` - Hypervector dimension
    /// * `seed` - Random seed for reproducibility
    #[allow(dead_code)]
    pub fn koch_snowflake(depth: usize, dim: usize, seed: u64) -> Self {
        assert!(depth <= 5, "Depth > 5 creates too many nodes (>3000)");
        assert!(dim >= 256, "Dimension should be >= 256 for good separation");

        // Calculate number of nodes (one per segment vertex)
        // Koch curve: 3 * 4^depth edges, same number of nodes for closed curve
        let n_nodes = 3 * 4_usize.pow(depth as u32);

        // Create node identities with seed-based variation
        let node_identities: Vec<ContinuousHV> = (0..n_nodes)
            .map(|i| {
                let base = ContinuousHV::basis(i % dim, dim);
                let noise = ContinuousHV::random(dim, seed + i as u64 * 1000).scale(0.05);
                base.add(&noise)
            })
            .collect();

        // Build adjacency list
        // Koch snowflake is a closed curve - connect consecutive nodes
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];

        // Connect along the curve (circular path)
        for i in 0..n_nodes {
            let next = (i + 1) % n_nodes;
            let prev = (i + n_nodes - 1) % n_nodes;
            adjacency[i].push(next);
            adjacency[i].push(prev);
        }

        // Add hierarchical connections based on fractal self-similarity
        // At each level, nodes that were peaks in the previous iteration
        // maintain special relationships
        if depth > 0 {
            // Connect nodes that correspond to the same position at different scales
            // This captures the self-similar structure of the Koch curve

            // For each scale level
            for level in 1..=depth {
                let segment_size = 4_usize.pow(level as u32);
                let num_segments = n_nodes / segment_size;

                // Connect midpoints of each segment (the "peak" nodes)
                for seg in 0..num_segments {
                    let segment_start = seg * segment_size;
                    let peak_node = segment_start + segment_size / 2;

                    // Connect peak to segment endpoints (hierarchical structure)
                    if peak_node < n_nodes {
                        let start_node = segment_start;
                        let end_node = (segment_start + segment_size) % n_nodes;

                        if !adjacency[peak_node].contains(&start_node) {
                            adjacency[peak_node].push(start_node);
                            adjacency[start_node].push(peak_node);
                        }
                        if !adjacency[peak_node].contains(&end_node) {
                            adjacency[peak_node].push(end_node);
                            adjacency[end_node].push(peak_node);
                        }
                    }
                }
            }
        }

        // Generate representations from adjacency
        let mut node_representations = Vec::with_capacity(n_nodes);
        for i in 0..n_nodes {
            if adjacency[i].is_empty() {
                node_representations.push(node_identities[i].clone());
            } else {
                // Remove duplicates
                let mut unique_neighbors = adjacency[i].clone();
                unique_neighbors.sort_unstable();
                unique_neighbors.dedup();

                let connections: Vec<ContinuousHV> = unique_neighbors
                    .iter()
                    .map(|&neighbor| node_identities[i].bind(&node_identities[neighbor]))
                    .collect();
                node_representations.push(ContinuousHV::bundle_owned(&connections));
            }
        }

        // Build edge list for compatibility
        let mut edges = Vec::new();
        for i in 0..n_nodes {
            for &neighbor in &adjacency[i] {
                if i < neighbor {
                    edges.push((i, neighbor));
                }
            }
        }

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::KochSnowflake,
            edges,
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
            let superposed = ring_component.add(&star_component).add(&random_component);

            node_representations.push(superposed);
        }

        // Combine edges from all three topologies (deduplicated)
        let mut edges: Vec<(usize, usize)> = ring
            .edges
            .iter()
            .chain(star.edges.iter())
            .chain(random.edges.iter())
            .cloned()
            .collect();
        edges.sort_unstable();
        edges.dedup();

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::Quantum,
            edges,
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
                let sim = self.node_representations[i].similarity(&self.node_representations[j]);
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
            for j in (i + 1)..n {
                // Upper triangle only
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
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
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
