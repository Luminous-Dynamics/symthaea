// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Decoded architecture (phenotype) from genome representation.

use symthaea_core::hdc::ContinuousHV;
use symthaea_core::phi_engine::PhiEngine;

use super::genome::{ArchitectureGenome, TopologyGene};

// ═══════════════════════════════════════════════════════════════════════════════
// ARCHITECTURE PHENOTYPE (DECODED NETWORK)
// ═══════════════════════════════════════════════════════════════════════════════

/// A decoded architecture ready for Phi evaluation
#[derive(Debug, Clone)]
pub struct DecodedArchitecture {
    /// Node hypervector representations
    pub nodes: Vec<ContinuousHV>,

    /// Adjacency list (node index -> list of (neighbor, weight))
    pub adjacency: Vec<Vec<(usize, f32)>>,

    /// Time constants per node
    pub tau_values: Vec<f32>,

    /// Module assignment per node
    pub module_assignment: Vec<usize>,

    /// Hierarchy level per node
    pub level_assignment: Vec<usize>,

    /// The genome that generated this architecture
    pub genome: ArchitectureGenome,
}

impl DecodedArchitecture {
    /// Decode a genome into a functional architecture
    pub fn from_genome(genome: &ArchitectureGenome) -> Self {
        let n = genome.num_nodes;
        let dim = genome.hdc_dim;
        let mut state = genome.seed;

        // Helper for pseudo-random generation
        let next_f32 = |s: &mut u64| -> f32 {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            (*s as f32) / (u64::MAX as f32)
        };

        // Generate node hypervectors
        let nodes: Vec<ContinuousHV> = (0..n)
            .map(|i| ContinuousHV::random(dim, genome.seed + i as u64 * 1000))
            .collect();

        // Assign modules
        let module_assignment: Vec<usize> = (0..n).map(|i| i % genome.num_modules.max(1)).collect();

        // Assign hierarchy levels
        let level_assignment: Vec<usize> =
            (0..n).map(|i| i % genome.hierarchy_depth.max(1)).collect();

        // Compute tau values based on level
        let tau_values: Vec<f32> = level_assignment
            .iter()
            .map(|&level| genome.base_tau * genome.tau_ratio.powi(level as i32))
            .collect();

        // Build adjacency list based on topology type
        let adjacency = match genome.topology_type {
            TopologyGene::Random => {
                Self::build_random_topology(n, genome.connection_density, &mut state, next_f32)
            }
            TopologyGene::Ring => Self::build_ring_topology(n),
            TopologyGene::Star => Self::build_star_topology(n),
            TopologyGene::HierarchicalTree => {
                Self::build_hierarchical_tree(n, genome.hierarchy_depth)
            }
            TopologyGene::Modular => Self::build_modular_topology(
                n,
                genome.num_modules,
                genome.connection_density,
                genome.bridge_ratio,
                &mut state,
                next_f32,
            ),
            TopologyGene::ScaleFree => Self::build_scale_free_topology(n, &mut state, next_f32),
            TopologyGene::SmallWorld => {
                Self::build_small_world_topology(n, genome.connection_density, &mut state, next_f32)
            }
            TopologyGene::Lattice => Self::build_lattice_topology(n),
            TopologyGene::CorePeriphery => Self::build_core_periphery_topology(
                n,
                genome.connection_density,
                &mut state,
                next_f32,
            ),
            TopologyGene::Attention => {
                Self::build_attention_topology(n, genome.connection_density, &mut state, next_f32)
            }
        };

        Self {
            nodes,
            adjacency,
            tau_values,
            module_assignment,
            level_assignment,
            genome: genome.clone(),
        }
    }

    // Topology builders

    fn build_random_topology(
        n: usize,
        density: f32,
        state: &mut u64,
        next_f32: impl Fn(&mut u64) -> f32,
    ) -> Vec<Vec<(usize, f32)>> {
        let mut adj = vec![Vec::new(); n];
        for i in 0..n {
            for j in (i + 1)..n {
                if next_f32(state) < density {
                    let weight = 0.5 + next_f32(state) * 0.5;
                    adj[i].push((j, weight));
                    adj[j].push((i, weight));
                }
            }
        }
        adj
    }

    fn build_ring_topology(n: usize) -> Vec<Vec<(usize, f32)>> {
        let mut adj = vec![Vec::new(); n];
        for i in 0..n {
            let next = (i + 1) % n;
            adj[i].push((next, 1.0));
            adj[next].push((i, 1.0));
        }
        adj
    }

    fn build_star_topology(n: usize) -> Vec<Vec<(usize, f32)>> {
        let mut adj = vec![Vec::new(); n];
        // Node 0 is the hub
        for i in 1..n {
            adj[0].push((i, 1.0));
            adj[i].push((0, 1.0));
        }
        adj
    }

    fn build_hierarchical_tree(n: usize, _depth: usize) -> Vec<Vec<(usize, f32)>> {
        let mut adj = vec![Vec::new(); n];
        // Binary tree structure
        for i in 0..n {
            let left = 2 * i + 1;
            let right = 2 * i + 2;
            if left < n {
                adj[i].push((left, 1.0));
                adj[left].push((i, 0.8)); // Asymmetric for hierarchy
            }
            if right < n {
                adj[i].push((right, 1.0));
                adj[right].push((i, 0.8));
            }
        }
        adj
    }

    fn build_modular_topology(
        n: usize,
        num_modules: usize,
        density: f32,
        bridge_ratio: f32,
        state: &mut u64,
        next_f32: impl Fn(&mut u64) -> f32,
    ) -> Vec<Vec<(usize, f32)>> {
        let mut adj = vec![Vec::new(); n];
        let module_size = n / num_modules.max(1);

        for i in 0..n {
            let module_i = i / module_size.max(1);
            for j in (i + 1)..n {
                let module_j = j / module_size.max(1);
                let threshold = if module_i == module_j {
                    density // Intra-module: use full density
                } else {
                    density * bridge_ratio // Inter-module: reduced density
                };

                if next_f32(state) < threshold {
                    let weight = 0.5 + next_f32(state) * 0.5;
                    adj[i].push((j, weight));
                    adj[j].push((i, weight));
                }
            }
        }
        adj
    }

    fn build_scale_free_topology(
        n: usize,
        state: &mut u64,
        next_f32: impl Fn(&mut u64) -> f32,
    ) -> Vec<Vec<(usize, f32)>> {
        let mut adj = vec![Vec::new(); n];
        let mut degree = vec![1usize; n];

        // Start with a small connected core
        if n >= 2 {
            adj[0].push((1, 1.0));
            adj[1].push((0, 1.0));
            degree[0] = 1;
            degree[1] = 1;
        }

        // Preferential attachment
        for i in 2..n {
            let total_degree: usize = degree.iter().sum();
            let mut edges_added = 0;
            let target_edges = 2.min(i);

            for j in 0..i {
                if edges_added >= target_edges {
                    break;
                }
                // Preferential attachment probability
                let prob = (degree[j] as f32) / (total_degree as f32 + 0.001);
                if next_f32(state) < prob * 2.0 {
                    adj[i].push((j, 1.0));
                    adj[j].push((i, 1.0));
                    degree[i] += 1;
                    degree[j] += 1;
                    edges_added += 1;
                }
            }
        }
        adj
    }

    fn build_small_world_topology(
        n: usize,
        density: f32,
        state: &mut u64,
        next_f32: impl Fn(&mut u64) -> f32,
    ) -> Vec<Vec<(usize, f32)>> {
        // Start with ring
        let mut adj = Self::build_ring_topology(n);

        // Add shortcuts
        let shortcut_prob = density * 0.3;
        for i in 0..n {
            if next_f32(state) < shortcut_prob {
                let target = (next_f32(state) * n as f32) as usize % n;
                if target != i && target != (i + 1) % n && target != (i + n - 1) % n {
                    // Check if edge already exists
                    if !adj[i].iter().any(|(j, _)| *j == target) {
                        adj[i].push((target, 0.7));
                        adj[target].push((i, 0.7));
                    }
                }
            }
        }
        adj
    }

    fn build_lattice_topology(n: usize) -> Vec<Vec<(usize, f32)>> {
        let mut adj = vec![Vec::new(); n];
        let side = (n as f32).sqrt() as usize;

        for i in 0..n {
            let row = i / side;
            let col = i % side;

            // Right neighbor
            if col + 1 < side {
                let right = row * side + col + 1;
                if right < n {
                    adj[i].push((right, 1.0));
                    adj[right].push((i, 1.0));
                }
            }

            // Down neighbor
            let down = (row + 1) * side + col;
            if down < n {
                adj[i].push((down, 1.0));
                adj[down].push((i, 1.0));
            }
        }
        adj
    }

    fn build_core_periphery_topology(
        n: usize,
        density: f32,
        state: &mut u64,
        next_f32: impl Fn(&mut u64) -> f32,
    ) -> Vec<Vec<(usize, f32)>> {
        let mut adj = vec![Vec::new(); n];
        let core_size = (n as f32 * 0.3).max(2.0) as usize;

        for i in 0..n {
            for j in (i + 1)..n {
                let both_core = i < core_size && j < core_size;
                let one_core = i < core_size || j < core_size;

                let threshold = if both_core {
                    density * 2.0 // Dense core
                } else if one_core {
                    density * 0.5 // Core-periphery connections
                } else {
                    density * 0.1 // Sparse periphery
                };

                if next_f32(state) < threshold.min(0.95) {
                    let weight = if both_core {
                        1.0
                    } else {
                        0.5 + next_f32(state) * 0.3
                    };
                    adj[i].push((j, weight));
                    adj[j].push((i, weight));
                }
            }
        }
        adj
    }

    fn build_attention_topology(
        n: usize,
        density: f32,
        state: &mut u64,
        next_f32: impl Fn(&mut u64) -> f32,
    ) -> Vec<Vec<(usize, f32)>> {
        let mut adj = vec![Vec::new(); n];
        let third = n / 3;

        // Q nodes: 0..third
        // K nodes: third..2*third
        // V nodes: 2*third..n

        // Q fully connects to K (attention weights)
        for q in 0..third {
            for k in third..(2 * third).min(n) {
                if next_f32(state) < density {
                    let weight = 0.5 + next_f32(state) * 0.5;
                    adj[q].push((k, weight));
                }
            }
        }

        // K connects to V (value retrieval)
        for k in third..(2 * third).min(n) {
            for v in (2 * third)..n {
                if next_f32(state) < density {
                    let weight = 0.5 + next_f32(state) * 0.5;
                    adj[k].push((v, weight));
                }
            }
        }

        adj
    }

    /// Convert to node representations for Phi calculation
    pub fn to_node_representations(&self) -> Vec<ContinuousHV> {
        self.nodes
            .iter()
            .enumerate()
            .map(|(i, node)| {
                let neighbors = &self.adjacency[i];
                if neighbors.is_empty() {
                    node.clone()
                } else {
                    // Bind node with weighted bundle of neighbors
                    let neighbor_hvs: Vec<ContinuousHV> = neighbors
                        .iter()
                        .map(|(j, w)| self.nodes[*j].scale(*w))
                        .collect();

                    let bundle = ContinuousHV::bundle_owned(&neighbor_hvs);
                    node.bind(&bundle)
                }
            })
            .collect()
    }

    /// Compute Phi for this architecture
    pub fn compute_phi(&self) -> f64 {
        let representations = self.to_node_representations();
        let engine = PhiEngine::auto();

        // Convert ContinuousHV to ContinuousHV for PhiEngine
        let continuous_hvs: Vec<symthaea_core::hdc::unified_hv::ContinuousHV> = representations
            .iter()
            .map(|hv| symthaea_core::hdc::unified_hv::ContinuousHV::from_vec(hv.values.clone()))
            .collect();

        let result = engine.compute(&continuous_hvs);
        result.phi
    }

    /// Get architecture statistics
    pub fn stats(&self) -> ArchitectureStats {
        let n = self.nodes.len();
        let total_edges: usize = self.adjacency.iter().map(|a| a.len()).sum();
        let max_possible = n * (n - 1);
        let density = if max_possible > 0 {
            total_edges as f32 / max_possible as f32
        } else {
            0.0
        };

        // Degree distribution
        let degrees: Vec<usize> = self.adjacency.iter().map(|a| a.len()).collect();
        let avg_degree = degrees.iter().sum::<usize>() as f32 / n.max(1) as f32;
        let max_degree = *degrees.iter().max().unwrap_or(&0);
        let min_degree = *degrees.iter().min().unwrap_or(&0);

        ArchitectureStats {
            num_nodes: n,
            num_edges: total_edges / 2, // Undirected
            density,
            avg_degree,
            max_degree,
            min_degree,
            num_modules: self.genome.num_modules,
            hierarchy_depth: self.genome.hierarchy_depth,
            topology: Some(self.topology_metrics()),
        }
    }

    /// Compute topological metrics using consciousness topology analysis.
    ///
    /// Converts the architecture's adjacency list into a simplicial complex
    /// and computes Betti numbers (connected components, loops, voids).
    pub fn topology_metrics(&self) -> TopologyMetrics {
        use symthaea_consciousness_topology::{BettiNumbers, Simplex, SimplicialComplex};

        let mut complex = SimplicialComplex::new();
        let n = self.nodes.len();

        // Add 0-simplices (vertices)
        for i in 0..n {
            complex.add_simplex(Simplex::new(vec![i]), 0.0);
        }

        // Add 1-simplices (edges) from adjacency
        for (i, neighbors) in self.adjacency.iter().enumerate() {
            for &(j, weight) in neighbors {
                if j > i {
                    // Undirected: add each edge once
                    complex.add_simplex(Simplex::new(vec![i, j]), weight as f64);
                }
            }
        }

        // Add 2-simplices (triangles) where all edges exist
        let edge_set: std::collections::HashSet<(usize, usize)> = self
            .adjacency
            .iter()
            .enumerate()
            .flat_map(|(i, neighbors)| {
                neighbors
                    .iter()
                    .map(move |&(j, _)| if i < j { (i, j) } else { (j, i) })
            })
            .collect();

        for i in 0..n {
            for &(j, _) in &self.adjacency[i] {
                if j <= i {
                    continue;
                }
                for &(k, _) in &self.adjacency[j] {
                    if k <= j {
                        continue;
                    }
                    // Check if triangle i-j-k exists (all 3 edges)
                    if edge_set.contains(&(i, k)) {
                        complex.add_simplex(Simplex::new(vec![i, j, k]), 0.0);
                    }
                }
            }
        }

        let betti = BettiNumbers::from_complex(&complex);
        let interp = betti.interpretation();

        TopologyMetrics {
            betti_0: betti.beta_0,
            betti_1: betti.beta_1,
            betti_2: betti.beta_2,
            euler_characteristic: betti.euler_characteristic,
            unity: interp.unity,
            complexity: interp.complexity,
        }
    }
}

/// Statistics about a decoded architecture
#[derive(Debug, Clone)]
pub struct ArchitectureStats {
    pub num_nodes: usize,
    pub num_edges: usize,
    pub density: f32,
    pub avg_degree: f32,
    pub max_degree: usize,
    pub min_degree: usize,
    pub num_modules: usize,
    pub hierarchy_depth: usize,
    /// Topological metrics from consciousness topology analysis
    pub topology: Option<TopologyMetrics>,
}

/// Topological metrics for an architecture (Betti numbers + interpretation)
#[derive(Debug, Clone)]
pub struct TopologyMetrics {
    /// Connected components (1 = unified)
    pub betti_0: usize,
    /// 1-dimensional holes (loops/cycles)
    pub betti_1: usize,
    /// 2-dimensional voids
    pub betti_2: usize,
    /// Euler characteristic: chi = beta_0 - beta_1 + beta_2
    pub euler_characteristic: i64,
    /// Unity score (1.0 = fully connected, <1.0 = fragmented)
    pub unity: f64,
    /// Topological complexity (loops + voids)
    pub complexity: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Helper: build a small genome suitable for fast tests.
    fn small_genome(topology: TopologyGene) -> ArchitectureGenome {
        ArchitectureGenome {
            num_nodes: 8,
            hdc_dim: 256,
            hierarchy_depth: 2,
            num_modules: 2,
            connection_density: 0.4,
            bridge_ratio: 0.3,
            topology_type: topology,
            seed: 42,
            ..Default::default()
        }
    }

    #[test]
    fn test_from_genome_node_count_matches() {
        for &n in &[4usize, 8, 16] {
            let genome = ArchitectureGenome {
                num_nodes: n,
                hdc_dim: 256,
                ..Default::default()
            };
            let arch = DecodedArchitecture::from_genome(&genome);
            assert_eq!(arch.nodes.len(), n);
            assert_eq!(arch.adjacency.len(), n);
            assert_eq!(arch.tau_values.len(), n);
            assert_eq!(arch.module_assignment.len(), n);
            assert_eq!(arch.level_assignment.len(), n);
        }
    }

    #[test]
    fn test_from_genome_preserves_genome() {
        let genome = small_genome(TopologyGene::Ring);
        let arch = DecodedArchitecture::from_genome(&genome);
        assert_eq!(arch.genome.num_nodes, genome.num_nodes);
        assert_eq!(arch.genome.topology_type, genome.topology_type);
        assert_eq!(arch.genome.hdc_dim, genome.hdc_dim);
    }

    #[test]
    fn test_tau_values_follow_hierarchy() {
        let genome = ArchitectureGenome {
            num_nodes: 6,
            hierarchy_depth: 3,
            base_tau: 1000.0,
            tau_ratio: 0.5,
            hdc_dim: 256,
            topology_type: TopologyGene::Ring,
            ..Default::default()
        };
        let arch = DecodedArchitecture::from_genome(&genome);

        // level_assignment = [0, 1, 2, 0, 1, 2]
        // tau[level] = base_tau * tau_ratio^level
        let expected_tau_0 = 1000.0 * 0.5f32.powi(0); // 1000
        let expected_tau_1 = 1000.0 * 0.5f32.powi(1); // 500
        let expected_tau_2 = 1000.0 * 0.5f32.powi(2); // 250

        assert!((arch.tau_values[0] - expected_tau_0).abs() < 1e-3);
        assert!((arch.tau_values[1] - expected_tau_1).abs() < 1e-3);
        assert!((arch.tau_values[2] - expected_tau_2).abs() < 1e-3);
        // Node 3 wraps back to level 0
        assert!((arch.tau_values[3] - expected_tau_0).abs() < 1e-3);
    }

    #[test]
    fn test_module_assignment_wraps() {
        let genome = ArchitectureGenome {
            num_nodes: 10,
            num_modules: 3,
            hdc_dim: 256,
            topology_type: TopologyGene::Ring,
            ..Default::default()
        };
        let arch = DecodedArchitecture::from_genome(&genome);
        // i % 3 => [0,1,2,0,1,2,0,1,2,0]
        for (i, &m) in arch.module_assignment.iter().enumerate() {
            assert_eq!(m, i % 3);
        }
    }

    #[test]
    fn test_ring_topology_edges() {
        let genome = small_genome(TopologyGene::Ring);
        let arch = DecodedArchitecture::from_genome(&genome);
        let n = genome.num_nodes;

        // Ring: each node connects to its neighbor; n edges total (undirected)
        let stats = arch.stats();
        assert_eq!(stats.num_edges, n, "Ring should have n edges");
        assert!(
            (stats.avg_degree - 2.0).abs() < 0.1,
            "Ring avg degree should be 2"
        );
    }

    #[test]
    fn test_star_topology_hub_degree() {
        let genome = small_genome(TopologyGene::Star);
        let arch = DecodedArchitecture::from_genome(&genome);
        let n = genome.num_nodes;

        // Hub (node 0) should connect to all other nodes
        assert_eq!(arch.adjacency[0].len(), n - 1);
        // Leaf nodes should connect to hub only
        for i in 1..n {
            assert_eq!(arch.adjacency[i].len(), 1);
            assert_eq!(arch.adjacency[i][0].0, 0);
        }
    }

    #[test]
    fn test_hierarchical_tree_binary_structure() {
        let genome = ArchitectureGenome {
            num_nodes: 7, // Perfect binary tree: 3 levels
            hdc_dim: 256,
            topology_type: TopologyGene::HierarchicalTree,
            ..Default::default()
        };
        let arch = DecodedArchitecture::from_genome(&genome);

        // Root (node 0) should have children 1, 2
        let root_neighbors: Vec<usize> = arch.adjacency[0].iter().map(|(j, _)| *j).collect();
        assert!(root_neighbors.contains(&1));
        assert!(root_neighbors.contains(&2));

        // Node 1 should have children 3, 4 and parent 0
        let n1_neighbors: Vec<usize> = arch.adjacency[1].iter().map(|(j, _)| *j).collect();
        assert!(n1_neighbors.contains(&0)); // parent
        assert!(n1_neighbors.contains(&3)); // left child
        assert!(n1_neighbors.contains(&4)); // right child
    }

    #[test]
    fn test_all_topologies_produce_valid_adjacency() {
        for topology in TopologyGene::all() {
            let genome = small_genome(*topology);
            let arch = DecodedArchitecture::from_genome(&genome);
            let n = genome.num_nodes;

            // Adjacency should have exactly n entries
            assert_eq!(
                arch.adjacency.len(),
                n,
                "Topology {:?}: wrong adjacency len",
                topology
            );

            // All neighbor indices should be in range [0, n)
            for (i, neighbors) in arch.adjacency.iter().enumerate() {
                for (j, w) in neighbors {
                    assert!(
                        *j < n,
                        "Topology {:?}: node {} neighbor {} out of range",
                        topology,
                        i,
                        j
                    );
                    assert!(
                        w.is_finite(),
                        "Topology {:?}: node {} weight to {} is not finite",
                        topology,
                        i,
                        j
                    );
                }
            }
        }
    }

    #[test]
    fn test_to_node_representations_dimension() {
        let genome = small_genome(TopologyGene::Ring);
        let arch = DecodedArchitecture::from_genome(&genome);
        let reps = arch.to_node_representations();

        assert_eq!(reps.len(), genome.num_nodes);
        for hv in &reps {
            assert_eq!(hv.values.len(), genome.hdc_dim);
        }
    }

    #[test]
    fn test_compute_phi_finite_and_nonnegative() {
        for topology in TopologyGene::all() {
            let genome = ArchitectureGenome {
                num_nodes: 6,
                hdc_dim: 256,
                topology_type: *topology,
                ..Default::default()
            };
            let arch = DecodedArchitecture::from_genome(&genome);
            let phi = arch.compute_phi();
            assert!(phi.is_finite(), "Phi not finite for {:?}", topology);
            assert!(phi >= 0.0, "Phi negative for {:?}: {}", topology, phi);
        }
    }

    #[test]
    fn test_stats_density_calculation() {
        let genome = small_genome(TopologyGene::Star);
        let arch = DecodedArchitecture::from_genome(&genome);
        let stats = arch.stats();
        let n = genome.num_nodes;

        // Star: n-1 edges (undirected), 2*(n-1) directed entries
        assert_eq!(stats.num_edges, n - 1);
        // Density = 2*(n-1) / (n*(n-1)) = 2/n
        let expected_density = 2.0 / n as f32;
        assert!(
            (stats.density - expected_density).abs() < 0.01,
            "expected density ~{}, got {}",
            expected_density,
            stats.density
        );
    }

    #[test]
    fn test_topology_metrics_ring_has_loop() {
        let genome = small_genome(TopologyGene::Ring);
        let arch = DecodedArchitecture::from_genome(&genome);
        let topo = arch.topology_metrics();

        // Ring should be one connected component
        assert_eq!(topo.betti_0, 1, "Ring should have 1 connected component");
        // Ring should have exactly 1 loop (1-cycle)
        assert_eq!(topo.betti_1, 1, "Ring should have 1 loop");
    }

    #[test]
    fn test_topology_metrics_star_connected() {
        let genome = small_genome(TopologyGene::Star);
        let arch = DecodedArchitecture::from_genome(&genome);
        let topo = arch.topology_metrics();

        // Star is connected: betti_0 = 1
        assert_eq!(topo.betti_0, 1, "Star should be connected");
        // Star (tree) has no loops: betti_1 = 0
        assert_eq!(topo.betti_1, 0, "Star should have no loops");
        // Unity should be 1.0 for connected graph
        assert!((topo.unity - 1.0).abs() < 1e-9, "Star unity should be 1.0");
    }
}
