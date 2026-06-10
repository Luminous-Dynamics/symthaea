// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Tier 2 topology generators: Geometric and manifold topologies.
//!
//! Sphere, Torus, KleinBottle, Lattice, SmallWorld, Möbius, Hyperbolic.

use super::super::unified_hv::ContinuousHV;
use super::types::{ConsciousnessTopology, TopologyType};

impl ConsciousnessTopology {
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
        let node_identities: Vec<ContinuousHV> = (0..n_nodes)
            .map(|i| {
                let base = ContinuousHV::basis(i, dim);
                let noise = ContinuousHV::random(dim, seed + (i as u64 * 1000)).scale(0.05);
                base.add(&noise)
            })
            .collect();

        // Icosahedron edge structure (30 edges, each vertex has degree 5)
        // Vertices arranged: 1 top, 5 upper pentagon, 5 lower pentagon, 1 bottom
        let edges = vec![
            // Top vertex (0) to upper pentagon (1-5)
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            // Upper pentagon connections
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 1),
            // Upper to lower connections
            (1, 6),
            (2, 6),
            (2, 7),
            (3, 7),
            (3, 8),
            (4, 8),
            (4, 9),
            (5, 9),
            (5, 10),
            (1, 10),
            // Lower pentagon connections
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (10, 6),
            // Lower pentagon (6-10) to bottom vertex (11)
            (6, 11),
            (7, 11),
            (8, 11),
            (9, 11),
            (10, 11),
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
                ContinuousHV::bundle_owned(&connections)
            };

            node_representations.push(repr);
        }

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::Sphere,
            edges,
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

        let node_identities: Vec<ContinuousHV> = (0..n_nodes)
            .map(|i| {
                let base = ContinuousHV::basis(i, dim);
                let noise = ContinuousHV::random(dim, seed + (i as u64 * 1000)).scale(0.05);
                base.add(&noise)
            })
            .collect();

        let mut node_representations = Vec::with_capacity(n_nodes);
        let mut edges = Vec::new();

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

                // Add all edges as (min, max) pairs to handle wraparound
                edges.push((idx.min(down), idx.max(down)));
                edges.push((idx.min(right), idx.max(right)));

                let repr = ContinuousHV::bundle_owned(&connections);
                node_representations.push(repr);
            }
        }

        // Deduplicate edges
        edges.sort_unstable();
        edges.dedup();

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::Torus,
            edges,
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

        let node_identities: Vec<ContinuousHV> = (0..n_nodes)
            .map(|i| {
                let base = ContinuousHV::basis(i, dim);
                let noise = ContinuousHV::random(dim, seed + (i as u64 * 1000)).scale(0.05);
                base.add(&noise)
            })
            .collect();

        let mut node_representations = Vec::with_capacity(n_nodes);
        let mut edges = Vec::new();

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

                // Add all edges as (min, max) pairs to handle wraparound
                edges.push((idx.min(down), idx.max(down)));
                edges.push((idx.min(right), idx.max(right)));

                let repr = ContinuousHV::bundle_owned(&connections);
                node_representations.push(repr);
            }
        }

        // Deduplicate edges
        edges.sort_unstable();
        edges.dedup();

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::KleinBottle,
            edges,
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
    pub fn lattice(n_nodes: usize, dim: usize, _seed: u64) -> Self {
        assert!(n_nodes >= 4, "Lattice needs at least 4 nodes (2x2 grid)");
        assert!(dim >= 256, "Dimension should be >= 256 for good separation");

        // Find grid size (nearest perfect square)
        let grid_size = (n_nodes as f64).sqrt().ceil() as usize;
        let actual_n_nodes = grid_size * grid_size;

        let node_identities: Vec<ContinuousHV> = (0..actual_n_nodes)
            .map(|i| ContinuousHV::basis(i, dim))
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
                ContinuousHV::bundle_owned(&connections)
            };

            node_representations.push(repr);
        }

        // Lattice edges: grid connections (right and down to avoid duplicates)
        let mut edges = Vec::new();
        for i in 0..actual_n_nodes {
            let row = i / grid_size;
            let col = i % grid_size;

            // Connect to right neighbor
            if col < grid_size - 1 {
                let right = row * grid_size + (col + 1);
                edges.push((i, right));
            }

            // Connect to down neighbor
            if row < grid_size - 1 {
                let down = (row + 1) * grid_size + col;
                edges.push((i, down));
            }
        }

        Self {
            n_nodes: actual_n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::Lattice,
            edges,
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
        assert!(n_nodes > k, "Need n_nodes >= k+1 for small-world");
        assert!(
            k.is_multiple_of(2),
            "k must be even for symmetric ring lattice"
        );
        assert!(k >= 2, "Need at least k=2 neighbors");
        assert!(
            (0.0..=1.0).contains(&p),
            "Rewiring probability p must be in [0, 1]"
        );
        assert!(dim >= 256, "Dimension should be >= 256 for good separation");

        let node_identities: Vec<ContinuousHV> =
            (0..n_nodes).map(|i| ContinuousHV::basis(i, dim)).collect();

        // Build initial k-regular ring lattice edges
        // Use min/max to handle wraparound edges correctly, then deduplicate
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for i in 0..n_nodes {
            for j in 1..=(k / 2) {
                let neighbor = (i + j) % n_nodes;
                edges.push((i.min(neighbor), i.max(neighbor)));
            }
        }
        edges.sort_unstable();
        edges.dedup();

        // Rewire edges with probability p
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(seed);

        let mut final_edges = Vec::new();
        for (i, j) in edges {
            if rng.r#gen::<f64>() < p {
                // Rewire: keep i, replace j with random node
                let mut new_target = rng.gen_range(0..n_nodes);

                // Avoid self-loops and duplicate edges
                while new_target == i
                    || final_edges.contains(&(i.min(new_target), i.max(new_target)))
                {
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
            let connections: Vec<ContinuousHV> = adjacency[i]
                .iter()
                .map(|&neighbor| node_identities[i].bind(&node_identities[neighbor]))
                .collect();

            let repr = if connections.is_empty() {
                node_identities[i].clone()
            } else {
                ContinuousHV::bundle_owned(&connections)
            };

            node_representations.push(repr);
        }

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::SmallWorld,
            edges: final_edges,
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
    pub fn mobius_strip(n_nodes: usize, dim: usize, _seed: u64) -> Self {
        assert!(n_nodes >= 4, "Möbius strip needs at least 4 nodes");
        assert!(
            n_nodes.is_multiple_of(2),
            "Möbius strip needs even number of nodes for twist"
        );
        assert!(dim >= 256, "Dimension should be >= 256 for good separation");

        let node_identities: Vec<ContinuousHV> =
            (0..n_nodes).map(|i| ContinuousHV::basis(i, dim)).collect();

        let mut node_representations = Vec::with_capacity(n_nodes);

        // Build edge list (ring topology) - use min/max for wraparound edge
        let mut edges: Vec<(usize, usize)> = (0..n_nodes)
            .map(|i| {
                let next = (i + 1) % n_nodes;
                (i.min(next), i.max(next))
            })
            .collect();
        edges.sort_unstable();
        edges.dedup();

        // First half: normal ring connections
        // Second half: one connection inverted (the Möbius twist!)
        for i in 0..n_nodes {
            let prev = (i + n_nodes - 1) % n_nodes;
            let next = (i + 1) % n_nodes;

            if i < n_nodes / 2 {
                // First half: normal binding (like regular ring)
                let conn1 = node_identities[i].bind(&node_identities[prev]);
                let conn2 = node_identities[i].bind(&node_identities[next]);
                let repr = ContinuousHV::bundle_owned(&[conn1, conn2]);
                node_representations.push(repr);
            } else {
                // Second half: invert the "next" connection (the twist!)
                let conn1 = node_identities[i].bind(&node_identities[prev]);
                let conn2_inverted = node_identities[i].bind(&node_identities[next].scale(-1.0));
                let repr = ContinuousHV::bundle_owned(&[conn1, conn2_inverted]);
                node_representations.push(repr);
            }
        }

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::MobiusStrip,
            edges,
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
    pub fn torus_square(grid_size: usize, dim: usize, _seed: u64) -> Self {
        assert!(grid_size >= 2, "Torus needs at least 2×2 grid");
        assert!(dim >= 256, "Dimension should be >= 256 for good separation");

        let n_nodes = grid_size * grid_size;

        let node_identities: Vec<ContinuousHV> =
            (0..n_nodes).map(|i| ContinuousHV::basis(i, dim)).collect();

        let mut node_representations = Vec::with_capacity(n_nodes);
        let mut edges = Vec::new();

        for i in 0..n_nodes {
            let row = i / grid_size;
            let col = i % grid_size;

            // Wraparound connections (modulo arithmetic)
            let up = ((row + grid_size - 1) % grid_size) * grid_size + col;
            let down = ((row + 1) % grid_size) * grid_size + col;
            let left = row * grid_size + ((col + grid_size - 1) % grid_size);
            let right = row * grid_size + ((col + 1) % grid_size);

            // Add all edges as (min, max) pairs to handle wraparound
            edges.push((i.min(down), i.max(down)));
            edges.push((i.min(right), i.max(right)));

            // Each node connects to its 4 neighbors
            let conn_up = node_identities[i].bind(&node_identities[up]);
            let conn_down = node_identities[i].bind(&node_identities[down]);
            let conn_left = node_identities[i].bind(&node_identities[left]);
            let conn_right = node_identities[i].bind(&node_identities[right]);

            let repr = ContinuousHV::bundle_owned(&[conn_up, conn_down, conn_left, conn_right]);
            node_representations.push(repr);
        }

        // Deduplicate edges
        edges.sort_unstable();
        edges.dedup();

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::Torus,
            edges,
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
    pub fn klein_bottle_square(grid_size: usize, dim: usize, _seed: u64) -> Self {
        assert!(grid_size >= 2, "Klein bottle needs at least 2×2 grid");
        assert!(dim >= 256, "Dimension should be >= 256 for good separation");

        let n_nodes = grid_size * grid_size;

        let node_identities: Vec<ContinuousHV> =
            (0..n_nodes).map(|i| ContinuousHV::basis(i, dim)).collect();

        let mut node_representations = Vec::with_capacity(n_nodes);
        let mut edges = Vec::new();

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
                flipped_row * grid_size
            } else {
                row * grid_size + (col + 1)
            };

            // Add edges (only add if i < neighbor to avoid duplicates)
            if i < down {
                edges.push((i, down));
            }
            if i < right {
                edges.push((i, right));
            }

            // Bind connections
            let conn_up = node_identities[i].bind(&node_identities[up]);
            let conn_down = node_identities[i].bind(&node_identities[down]);
            let conn_left = node_identities[i].bind(&node_identities[left]);
            let conn_right = node_identities[i].bind(&node_identities[right]);

            let repr = ContinuousHV::bundle_owned(&[conn_up, conn_down, conn_left, conn_right]);
            node_representations.push(repr);
        }

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::KleinBottle,
            edges,
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
    pub fn hyperbolic(n_nodes: usize, dim: usize, branching: usize, _seed: u64) -> Self {
        assert!(n_nodes >= 2, "Hyperbolic needs at least 2 nodes");
        assert!(branching >= 2, "Branching factor must be >= 2");
        assert!(dim >= 256, "Dimension should be >= 256 for good separation");

        // Build a tree structure with lateral connections at each level
        let node_identities: Vec<ContinuousHV> =
            (0..n_nodes).map(|i| ContinuousHV::basis(i, dim)).collect();

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
            let depth_size = branching
                .pow(current_depth as u32)
                .min(n_nodes - depth_start);
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
            topology_type: TopologyType::Hyperbolic,
            edges,
        }
    }
}
