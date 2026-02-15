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
}
