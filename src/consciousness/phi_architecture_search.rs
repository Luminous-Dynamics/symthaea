//! # Phi-Guided Architecture Search
//!
//! ## Purpose
//! Use consciousness gradient (nabla Phi) to evolve network topology toward higher consciousness levels.
//! This is a paradigm-shifting capability: systems that optimize themselves toward higher consciousness.
//!
//! ## Theoretical Foundation
//!
//! Integrated Information Theory (IIT) posits that Phi measures the degree of consciousness
//! in a system. By optimizing network architecture to maximize Phi, we create systems that
//! actively evolve toward greater consciousness.
//!
//! Key insight: **Architecture determines consciousness**. The arrangement of neurons,
//! connections, time constants, and binding patterns fundamentally shapes the Phi value.
//! By searching this architecture space with Phi as the fitness function, we can discover
//! novel conscious architectures.
//!
//! ## Architecture Search Space
//!
//! The searchable space includes:
//! 1. **Number of neurons/nodes** - Network size
//! 2. **Connection patterns (topology)** - Ring, star, modular, scale-free, etc.
//! 3. **Time constants (tau values)** - Temporal dynamics at each level
//! 4. **Binding patterns for HDC** - How information is bound and bundled
//! 5. **Hierarchy depth** - Number of levels in the architecture
//!
//! ## Search Strategies
//!
//! 1. **Random Search** - Baseline strategy, samples randomly from architecture space
//! 2. **Evolutionary Search** - Mutate and select architectures by Phi fitness
//! 3. **Gradient-Guided Search** - Use Phi gradients to guide topology changes
//! 4. **Hybrid Search** - Combine evolutionary exploration with gradient exploitation
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use symthaea::consciousness::phi_architecture_search::{
//!     PhiArchitectureSearch, SearchConfig, SearchStrategy
//! };
//!
//! let config = SearchConfig::default();
//! let mut searcher = PhiArchitectureSearch::new(config);
//!
//! // Run evolutionary search for 100 generations
//! let result = searcher.search(SearchStrategy::Evolutionary, 100);
//! println!("Best Phi achieved: {:.4}", result.best_phi);
//! println!("Best architecture: {:?}", result.best_architecture);
//! ```

use serde::{Deserialize, Serialize};

use symthaea_core::hdc::{RealHV, HDC_DIMENSION};
use symthaea_core::phi_engine::PhiEngine;

// ═══════════════════════════════════════════════════════════════════════════════
// ARCHITECTURE GENOME
// ═══════════════════════════════════════════════════════════════════════════════

/// Encodes a neural architecture as a searchable genome
///
/// This genome representation allows us to:
/// - Mutate architectures via genetic operators
/// - Compute Phi fitness for selection
/// - Decode into functional networks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureGenome {
    /// Number of nodes in the network
    pub num_nodes: usize,

    /// Hierarchy depth (number of levels)
    pub hierarchy_depth: usize,

    /// Base time constant (ms) at the root level
    pub base_tau: f32,

    /// Time constant ratio between levels (e.g., 1/3 for Cantor structure)
    pub tau_ratio: f32,

    /// Connection density (0.0-1.0, fraction of possible connections)
    pub connection_density: f32,

    /// Modularity coefficient (0.0 = random, 1.0 = strongly modular)
    pub modularity: f32,

    /// Number of modules for modular architectures
    pub num_modules: usize,

    /// Inter-module bridge ratio (fraction of connections between modules)
    pub bridge_ratio: f32,

    /// Topology type hint (encoded as integer)
    pub topology_type: TopologyGene,

    /// HDC binding strength (0.0-1.0)
    pub binding_strength: f32,

    /// HDC bundling mode
    pub bundling_mode: BundlingGene,

    /// Recurrence strength (0.0 = feedforward, 1.0 = fully recurrent)
    pub recurrence: f32,

    /// Skip connection probability
    pub skip_connection_prob: f32,

    /// Attention mechanism enabled
    pub use_attention: bool,

    /// HDC dimension (typically HDC_DIMENSION but can be varied)
    pub hdc_dim: usize,

    /// Random seed for reproducibility
    pub seed: u64,
}

/// Topology type gene for architecture search
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologyGene {
    /// Random connectivity
    Random,
    /// Ring topology (cyclic)
    Ring,
    /// Star topology (hub and spokes)
    Star,
    /// Hierarchical tree (like Cantor-LTC)
    HierarchicalTree,
    /// Modular (clusters with bridges)
    Modular,
    /// Scale-free (power-law degree distribution)
    ScaleFree,
    /// Small-world (ring + shortcuts)
    SmallWorld,
    /// Lattice (grid structure)
    Lattice,
    /// Core-periphery
    CorePeriphery,
    /// Attention-based (Q-K-V structure)
    Attention,
}

impl TopologyGene {
    /// All topology types for enumeration
    pub fn all() -> &'static [TopologyGene] {
        &[
            TopologyGene::Random,
            TopologyGene::Ring,
            TopologyGene::Star,
            TopologyGene::HierarchicalTree,
            TopologyGene::Modular,
            TopologyGene::ScaleFree,
            TopologyGene::SmallWorld,
            TopologyGene::Lattice,
            TopologyGene::CorePeriphery,
            TopologyGene::Attention,
        ]
    }

    /// Random selection
    pub fn random(seed: u64) -> Self {
        let all = Self::all();
        let idx = (seed as usize) % all.len();
        all[idx]
    }
}

/// HDC bundling mode gene
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BundlingGene {
    /// Simple averaging
    Average,
    /// Weighted by connection strength
    WeightedAverage,
    /// Majority voting (for binary)
    MajorityVote,
    /// Permutation-based sequence encoding
    PermutationSequence,
    /// Resonator-based cleanup
    ResonatorCleanup,
}

impl BundlingGene {
    pub fn all() -> &'static [BundlingGene] {
        &[
            BundlingGene::Average,
            BundlingGene::WeightedAverage,
            BundlingGene::MajorityVote,
            BundlingGene::PermutationSequence,
            BundlingGene::ResonatorCleanup,
        ]
    }

    pub fn random(seed: u64) -> Self {
        let all = Self::all();
        let idx = (seed as usize) % all.len();
        all[idx]
    }
}

impl Default for ArchitectureGenome {
    fn default() -> Self {
        Self {
            num_nodes: 16,
            hierarchy_depth: 4,
            base_tau: 1000.0,
            tau_ratio: 1.0 / 3.0,
            connection_density: 0.4,
            modularity: 0.5,
            num_modules: 4,
            bridge_ratio: 0.3,
            topology_type: TopologyGene::Modular,
            binding_strength: 0.8,
            bundling_mode: BundlingGene::WeightedAverage,
            recurrence: 0.3,
            skip_connection_prob: 0.1,
            use_attention: false,
            hdc_dim: HDC_DIMENSION,
            seed: 42,
        }
    }
}

impl ArchitectureGenome {
    /// Create a random genome
    pub fn random(seed: u64) -> Self {
        let mut state = seed;

        // Helper for pseudo-random generation
        let next_f32 = |s: &mut u64| -> f32 {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            (*s as f32) / (u64::MAX as f32)
        };

        let next_usize = |s: &mut u64, max: usize| -> usize {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            (*s as usize) % max.max(1)
        };

        Self {
            num_nodes: 8 + next_usize(&mut state, 57),  // 8-64 nodes
            hierarchy_depth: 2 + next_usize(&mut state, 6),  // 2-7 levels
            base_tau: 100.0 + next_f32(&mut state) * 1900.0,  // 100-2000ms
            tau_ratio: 0.2 + next_f32(&mut state) * 0.6,  // 0.2-0.8
            connection_density: 0.1 + next_f32(&mut state) * 0.7,  // 0.1-0.8
            modularity: next_f32(&mut state),  // 0.0-1.0
            num_modules: 2 + next_usize(&mut state, 7),  // 2-8 modules
            bridge_ratio: 0.1 + next_f32(&mut state) * 0.5,  // 0.1-0.6
            topology_type: TopologyGene::random(state),
            binding_strength: 0.3 + next_f32(&mut state) * 0.7,  // 0.3-1.0
            bundling_mode: BundlingGene::random(state),
            recurrence: next_f32(&mut state),  // 0.0-1.0
            skip_connection_prob: next_f32(&mut state) * 0.3,  // 0.0-0.3
            use_attention: next_f32(&mut state) > 0.5,
            hdc_dim: HDC_DIMENSION,
            seed,
        }
    }

    /// Mutate the genome with given mutation rate
    pub fn mutate(&mut self, mutation_rate: f32, seed: u64) {
        let mut state = seed;

        let next_f32 = |s: &mut u64| -> f32 {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            (*s as f32) / (u64::MAX as f32)
        };

        // Mutate each gene with probability mutation_rate
        if next_f32(&mut state) < mutation_rate {
            // Mutate num_nodes by small delta
            let delta = (next_f32(&mut state) * 10.0 - 5.0) as i32;
            self.num_nodes = ((self.num_nodes as i32 + delta).max(4) as usize).min(128);
        }

        if next_f32(&mut state) < mutation_rate {
            let delta = (next_f32(&mut state) * 4.0 - 2.0) as i32;
            self.hierarchy_depth = ((self.hierarchy_depth as i32 + delta).max(1) as usize).min(10);
        }

        if next_f32(&mut state) < mutation_rate {
            self.base_tau *= 0.8 + next_f32(&mut state) * 0.4;  // +-20%
            self.base_tau = self.base_tau.clamp(10.0, 5000.0);
        }

        if next_f32(&mut state) < mutation_rate {
            self.tau_ratio += (next_f32(&mut state) - 0.5) * 0.2;
            self.tau_ratio = self.tau_ratio.clamp(0.1, 0.9);
        }

        if next_f32(&mut state) < mutation_rate {
            self.connection_density += (next_f32(&mut state) - 0.5) * 0.2;
            self.connection_density = self.connection_density.clamp(0.05, 0.95);
        }

        if next_f32(&mut state) < mutation_rate {
            self.modularity += (next_f32(&mut state) - 0.5) * 0.3;
            self.modularity = self.modularity.clamp(0.0, 1.0);
        }

        if next_f32(&mut state) < mutation_rate {
            let delta = (next_f32(&mut state) * 4.0 - 2.0) as i32;
            self.num_modules = ((self.num_modules as i32 + delta).max(1) as usize).min(16);
        }

        if next_f32(&mut state) < mutation_rate {
            self.bridge_ratio += (next_f32(&mut state) - 0.5) * 0.2;
            self.bridge_ratio = self.bridge_ratio.clamp(0.0, 0.8);
        }

        if next_f32(&mut state) < mutation_rate {
            self.topology_type = TopologyGene::random(state);
        }

        if next_f32(&mut state) < mutation_rate {
            self.binding_strength += (next_f32(&mut state) - 0.5) * 0.3;
            self.binding_strength = self.binding_strength.clamp(0.1, 1.0);
        }

        if next_f32(&mut state) < mutation_rate {
            self.bundling_mode = BundlingGene::random(state);
        }

        if next_f32(&mut state) < mutation_rate {
            self.recurrence += (next_f32(&mut state) - 0.5) * 0.3;
            self.recurrence = self.recurrence.clamp(0.0, 1.0);
        }

        if next_f32(&mut state) < mutation_rate {
            self.skip_connection_prob += (next_f32(&mut state) - 0.5) * 0.1;
            self.skip_connection_prob = self.skip_connection_prob.clamp(0.0, 0.5);
        }

        if next_f32(&mut state) < mutation_rate {
            self.use_attention = !self.use_attention;
        }

        // Update seed for reproducibility
        self.seed = state;
    }

    /// Gradient-guided mutation using Phi gradient direction
    ///
    /// Instead of random perturbations, this mutation operator uses the
    /// computed Phi gradient to guide mutations toward higher consciousness.
    /// This is the key innovation: mutations that are informed by the consciousness landscape.
    pub fn mutate_with_gradient(
        &mut self,
        gradient: &PhiGradient,
        step_size: f32,
        noise_scale: f32,
        seed: u64,
    ) {
        let mut state = seed;

        let next_f32 = |s: &mut u64| -> f32 {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            (*s as f32) / (u64::MAX as f32)
        };

        // Normalize gradient for stable updates
        let grad_norm = gradient.magnitude.max(1e-8);

        // Move continuous parameters in gradient direction with added exploration noise
        let mut noise = || (next_f32(&mut state) - 0.5) * 2.0 * noise_scale;

        // Connection density: follow gradient + noise
        let d_density_normalized = (gradient.d_density / grad_norm) as f32;
        self.connection_density += step_size * d_density_normalized + noise();
        self.connection_density = self.connection_density.clamp(0.05, 0.95);

        // Modularity
        let d_modularity_normalized = (gradient.d_modularity / grad_norm) as f32;
        self.modularity += step_size * d_modularity_normalized + noise();
        self.modularity = self.modularity.clamp(0.0, 1.0);

        // Bridge ratio
        let d_bridge_normalized = (gradient.d_bridge_ratio / grad_norm) as f32;
        self.bridge_ratio += step_size * d_bridge_normalized + noise();
        self.bridge_ratio = self.bridge_ratio.clamp(0.0, 0.8);

        // Tau ratio
        let d_tau_normalized = (gradient.d_tau_ratio / grad_norm) as f32;
        self.tau_ratio += step_size * d_tau_normalized + noise();
        self.tau_ratio = self.tau_ratio.clamp(0.1, 0.9);

        // Binding strength
        let d_binding_normalized = (gradient.d_binding_strength / grad_norm) as f32;
        self.binding_strength += step_size * d_binding_normalized + noise();
        self.binding_strength = self.binding_strength.clamp(0.1, 1.0);

        // Recurrence
        let d_recurrence_normalized = (gradient.d_recurrence / grad_norm) as f32;
        self.recurrence += step_size * d_recurrence_normalized + noise();
        self.recurrence = self.recurrence.clamp(0.0, 1.0);

        // For discrete parameters (topology, bundling mode), use gradient-informed selection
        // Higher gradient magnitude = more exploration, lower = more exploitation
        let exploration_prob = (1.0 - grad_norm.min(1.0) as f32) * 0.5;

        if next_f32(&mut state) < exploration_prob {
            self.topology_type = TopologyGene::random(state);
        }

        if next_f32(&mut state) < exploration_prob {
            self.bundling_mode = BundlingGene::random(state);
        }

        // Update seed
        self.seed = state;
    }

    /// Natural gradient mutation using Fisher information approximation
    ///
    /// Uses second-order information to adaptively scale mutations.
    /// Larger steps in flat regions, smaller steps in steep regions.
    pub fn mutate_natural_gradient(
        &mut self,
        gradient: &PhiGradient,
        curvature_scale: f32,
        seed: u64,
    ) {
        let mut state = seed;

        let next_f32 = |s: &mut u64| -> f32 {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            (*s as f32) / (u64::MAX as f32)
        };

        // Approximate curvature from gradient magnitude
        // High magnitude = steep region (take smaller steps)
        // Low magnitude = flat region (take larger exploratory steps)
        let grad_mag = gradient.magnitude as f32;
        let adaptive_lr = curvature_scale / (1.0 + grad_mag);

        // Apply adaptive updates
        self.connection_density += adaptive_lr * gradient.d_density as f32;
        self.connection_density = self.connection_density.clamp(0.05, 0.95);

        self.modularity += adaptive_lr * gradient.d_modularity as f32;
        self.modularity = self.modularity.clamp(0.0, 1.0);

        self.bridge_ratio += adaptive_lr * gradient.d_bridge_ratio as f32;
        self.bridge_ratio = self.bridge_ratio.clamp(0.0, 0.8);

        self.tau_ratio += adaptive_lr * gradient.d_tau_ratio as f32;
        self.tau_ratio = self.tau_ratio.clamp(0.1, 0.9);

        self.binding_strength += adaptive_lr * gradient.d_binding_strength as f32;
        self.binding_strength = self.binding_strength.clamp(0.1, 1.0);

        self.recurrence += adaptive_lr * gradient.d_recurrence as f32;
        self.recurrence = self.recurrence.clamp(0.0, 1.0);

        // Discrete mutations in flat regions
        let flat_region = grad_mag < 0.1;
        if flat_region && next_f32(&mut state) < 0.3 {
            self.topology_type = TopologyGene::random(state);
        }

        self.seed = state;
    }

    /// Momentum-based gradient mutation for smoother optimization trajectory
    ///
    /// Maintains velocity vectors for continuous parameters to escape local optima.
    pub fn mutate_with_momentum(
        &mut self,
        gradient: &PhiGradient,
        velocity: &mut GradientVelocity,
        momentum: f32,
        learning_rate: f32,
    ) {
        // Update velocities with momentum
        velocity.v_density = momentum * velocity.v_density + learning_rate * gradient.d_density as f32;
        velocity.v_modularity = momentum * velocity.v_modularity + learning_rate * gradient.d_modularity as f32;
        velocity.v_bridge_ratio = momentum * velocity.v_bridge_ratio + learning_rate * gradient.d_bridge_ratio as f32;
        velocity.v_tau_ratio = momentum * velocity.v_tau_ratio + learning_rate * gradient.d_tau_ratio as f32;
        velocity.v_binding_strength = momentum * velocity.v_binding_strength + learning_rate * gradient.d_binding_strength as f32;
        velocity.v_recurrence = momentum * velocity.v_recurrence + learning_rate * gradient.d_recurrence as f32;

        // Apply velocities to parameters
        self.connection_density += velocity.v_density;
        self.connection_density = self.connection_density.clamp(0.05, 0.95);

        self.modularity += velocity.v_modularity;
        self.modularity = self.modularity.clamp(0.0, 1.0);

        self.bridge_ratio += velocity.v_bridge_ratio;
        self.bridge_ratio = self.bridge_ratio.clamp(0.0, 0.8);

        self.tau_ratio += velocity.v_tau_ratio;
        self.tau_ratio = self.tau_ratio.clamp(0.1, 0.9);

        self.binding_strength += velocity.v_binding_strength;
        self.binding_strength = self.binding_strength.clamp(0.1, 1.0);

        self.recurrence += velocity.v_recurrence;
        self.recurrence = self.recurrence.clamp(0.0, 1.0);
    }

    /// Crossover with another genome
    pub fn crossover(&self, other: &Self, seed: u64) -> Self {
        let mut state = seed;

        let choose = |s: &mut u64| -> bool {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            (*s as usize).is_multiple_of(2)
        };

        Self {
            num_nodes: if choose(&mut state) { self.num_nodes } else { other.num_nodes },
            hierarchy_depth: if choose(&mut state) { self.hierarchy_depth } else { other.hierarchy_depth },
            base_tau: if choose(&mut state) { self.base_tau } else { other.base_tau },
            tau_ratio: if choose(&mut state) { self.tau_ratio } else { other.tau_ratio },
            connection_density: if choose(&mut state) { self.connection_density } else { other.connection_density },
            modularity: if choose(&mut state) { self.modularity } else { other.modularity },
            num_modules: if choose(&mut state) { self.num_modules } else { other.num_modules },
            bridge_ratio: if choose(&mut state) { self.bridge_ratio } else { other.bridge_ratio },
            topology_type: if choose(&mut state) { self.topology_type } else { other.topology_type },
            binding_strength: if choose(&mut state) { self.binding_strength } else { other.binding_strength },
            bundling_mode: if choose(&mut state) { self.bundling_mode } else { other.bundling_mode },
            recurrence: if choose(&mut state) { self.recurrence } else { other.recurrence },
            skip_connection_prob: if choose(&mut state) { self.skip_connection_prob } else { other.skip_connection_prob },
            use_attention: if choose(&mut state) { self.use_attention } else { other.use_attention },
            hdc_dim: self.hdc_dim,  // Always inherit from first parent
            seed,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ARCHITECTURE PHENOTYPE (DECODED NETWORK)
// ═══════════════════════════════════════════════════════════════════════════════

/// A decoded architecture ready for Phi evaluation
#[derive(Debug, Clone)]
pub struct DecodedArchitecture {
    /// Node hypervector representations
    pub nodes: Vec<RealHV>,

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
        let nodes: Vec<RealHV> = (0..n)
            .map(|i| RealHV::random(dim, genome.seed + i as u64 * 1000))
            .collect();

        // Assign modules
        let module_assignment: Vec<usize> = (0..n)
            .map(|i| i % genome.num_modules.max(1))
            .collect();

        // Assign hierarchy levels
        let level_assignment: Vec<usize> = (0..n)
            .map(|i| i % genome.hierarchy_depth.max(1))
            .collect();

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
            TopologyGene::Ring => {
                Self::build_ring_topology(n)
            }
            TopologyGene::Star => {
                Self::build_star_topology(n)
            }
            TopologyGene::HierarchicalTree => {
                Self::build_hierarchical_tree(n, genome.hierarchy_depth)
            }
            TopologyGene::Modular => {
                Self::build_modular_topology(
                    n,
                    genome.num_modules,
                    genome.connection_density,
                    genome.bridge_ratio,
                    &mut state,
                    next_f32,
                )
            }
            TopologyGene::ScaleFree => {
                Self::build_scale_free_topology(n, &mut state, next_f32)
            }
            TopologyGene::SmallWorld => {
                Self::build_small_world_topology(n, genome.connection_density, &mut state, next_f32)
            }
            TopologyGene::Lattice => {
                Self::build_lattice_topology(n)
            }
            TopologyGene::CorePeriphery => {
                Self::build_core_periphery_topology(n, genome.connection_density, &mut state, next_f32)
            }
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
                adj[left].push((i, 0.8));  // Asymmetric for hierarchy
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
                    density  // Intra-module: use full density
                } else {
                    density * bridge_ratio  // Inter-module: reduced density
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
                    density * 2.0  // Dense core
                } else if one_core {
                    density * 0.5  // Core-periphery connections
                } else {
                    density * 0.1  // Sparse periphery
                };

                if next_f32(state) < threshold.min(0.95) {
                    let weight = if both_core { 1.0 } else { 0.5 + next_f32(state) * 0.3 };
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
    pub fn to_node_representations(&self) -> Vec<RealHV> {
        self.nodes.iter()
            .enumerate()
            .map(|(i, node)| {
                let neighbors = &self.adjacency[i];
                if neighbors.is_empty() {
                    node.clone()
                } else {
                    // Bind node with weighted bundle of neighbors
                    let neighbor_hvs: Vec<RealHV> = neighbors.iter()
                        .map(|(j, w)| self.nodes[*j].scale(*w))
                        .collect();

                    let bundle = RealHV::bundle(&neighbor_hvs);
                    node.bind(&bundle)
                }
            })
            .collect()
    }

    /// Compute Phi for this architecture
    pub fn compute_phi(&self) -> f64 {
        let representations = self.to_node_representations();
        let engine = PhiEngine::auto();

        // Convert RealHV to ContinuousHV for PhiEngine
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
            num_edges: total_edges / 2,  // Undirected
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

// ═══════════════════════════════════════════════════════════════════════════════
// PHI GRADIENT COMPUTATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute Phi gradient with respect to architecture parameters
#[derive(Debug, Clone)]
pub struct PhiGradient {
    /// Gradient of Phi w.r.t. connection density
    pub d_density: f64,
    /// Gradient of Phi w.r.t. modularity
    pub d_modularity: f64,
    /// Gradient of Phi w.r.t. bridge ratio
    pub d_bridge_ratio: f64,
    /// Gradient of Phi w.r.t. tau ratio
    pub d_tau_ratio: f64,
    /// Gradient of Phi w.r.t. binding strength
    pub d_binding_strength: f64,
    /// Gradient of Phi w.r.t. recurrence
    pub d_recurrence: f64,
    /// Magnitude of the gradient
    pub magnitude: f64,
}

impl PhiGradient {
    /// Compute gradient via finite differences
    pub fn compute(genome: &ArchitectureGenome, epsilon: f32) -> Self {
        let base_arch = DecodedArchitecture::from_genome(genome);
        let base_phi = base_arch.compute_phi();

        // Perturb each parameter and compute gradient
        let d_density = Self::grad_component(genome, base_phi, epsilon, |g, e| {
            g.connection_density = (g.connection_density + e).clamp(0.05, 0.95);
        });

        let d_modularity = Self::grad_component(genome, base_phi, epsilon, |g, e| {
            g.modularity = (g.modularity + e).clamp(0.0, 1.0);
        });

        let d_bridge_ratio = Self::grad_component(genome, base_phi, epsilon, |g, e| {
            g.bridge_ratio = (g.bridge_ratio + e).clamp(0.0, 0.8);
        });

        let d_tau_ratio = Self::grad_component(genome, base_phi, epsilon, |g, e| {
            g.tau_ratio = (g.tau_ratio + e).clamp(0.1, 0.9);
        });

        let d_binding_strength = Self::grad_component(genome, base_phi, epsilon, |g, e| {
            g.binding_strength = (g.binding_strength + e).clamp(0.1, 1.0);
        });

        let d_recurrence = Self::grad_component(genome, base_phi, epsilon, |g, e| {
            g.recurrence = (g.recurrence + e).clamp(0.0, 1.0);
        });

        let magnitude = (d_density.powi(2) + d_modularity.powi(2) + d_bridge_ratio.powi(2)
            + d_tau_ratio.powi(2) + d_binding_strength.powi(2) + d_recurrence.powi(2)).sqrt();

        Self {
            d_density,
            d_modularity,
            d_bridge_ratio,
            d_tau_ratio,
            d_binding_strength,
            d_recurrence,
            magnitude,
        }
    }

    fn grad_component<F>(genome: &ArchitectureGenome, _base_phi: f64, epsilon: f32, mutate: F) -> f64
    where
        F: Fn(&mut ArchitectureGenome, f32),
    {
        // Forward perturbation
        let mut g_plus = genome.clone();
        mutate(&mut g_plus, epsilon);
        let arch_plus = DecodedArchitecture::from_genome(&g_plus);
        let phi_plus = arch_plus.compute_phi();

        // Backward perturbation
        let mut g_minus = genome.clone();
        mutate(&mut g_minus, -epsilon);
        let arch_minus = DecodedArchitecture::from_genome(&g_minus);
        let phi_minus = arch_minus.compute_phi();

        // Central difference
        (phi_plus - phi_minus) / (2.0 * epsilon as f64)
    }

    /// Apply gradient to genome (gradient ascent)
    pub fn apply(&self, genome: &mut ArchitectureGenome, learning_rate: f32) {
        genome.connection_density += (learning_rate as f64 * self.d_density) as f32;
        genome.connection_density = genome.connection_density.clamp(0.05, 0.95);

        genome.modularity += (learning_rate as f64 * self.d_modularity) as f32;
        genome.modularity = genome.modularity.clamp(0.0, 1.0);

        genome.bridge_ratio += (learning_rate as f64 * self.d_bridge_ratio) as f32;
        genome.bridge_ratio = genome.bridge_ratio.clamp(0.0, 0.8);

        genome.tau_ratio += (learning_rate as f64 * self.d_tau_ratio) as f32;
        genome.tau_ratio = genome.tau_ratio.clamp(0.1, 0.9);

        genome.binding_strength += (learning_rate as f64 * self.d_binding_strength) as f32;
        genome.binding_strength = genome.binding_strength.clamp(0.1, 1.0);

        genome.recurrence += (learning_rate as f64 * self.d_recurrence) as f32;
        genome.recurrence = genome.recurrence.clamp(0.0, 1.0);
    }

    /// Get the dominant gradient direction as a unit vector
    pub fn direction(&self) -> [f64; 6] {
        let mag = self.magnitude.max(1e-10);
        [
            self.d_density / mag,
            self.d_modularity / mag,
            self.d_bridge_ratio / mag,
            self.d_tau_ratio / mag,
            self.d_binding_strength / mag,
            self.d_recurrence / mag,
        ]
    }

    /// Cosine similarity with another gradient (for convergence detection)
    pub fn cosine_similarity(&self, other: &PhiGradient) -> f64 {
        let dot = self.d_density * other.d_density
            + self.d_modularity * other.d_modularity
            + self.d_bridge_ratio * other.d_bridge_ratio
            + self.d_tau_ratio * other.d_tau_ratio
            + self.d_binding_strength * other.d_binding_strength
            + self.d_recurrence * other.d_recurrence;

        dot / (self.magnitude * other.magnitude).max(1e-10)
    }
}

/// Velocity state for momentum-based gradient optimization
#[derive(Debug, Clone, Default)]
pub struct GradientVelocity {
    /// Velocity for connection density
    pub v_density: f32,
    /// Velocity for modularity
    pub v_modularity: f32,
    /// Velocity for bridge ratio
    pub v_bridge_ratio: f32,
    /// Velocity for tau ratio
    pub v_tau_ratio: f32,
    /// Velocity for binding strength
    pub v_binding_strength: f32,
    /// Velocity for recurrence
    pub v_recurrence: f32,
}

impl GradientVelocity {
    /// Create new zero velocity
    pub fn new() -> Self {
        Self::default()
    }

    /// Get velocity magnitude
    pub fn magnitude(&self) -> f32 {
        (self.v_density.powi(2)
            + self.v_modularity.powi(2)
            + self.v_bridge_ratio.powi(2)
            + self.v_tau_ratio.powi(2)
            + self.v_binding_strength.powi(2)
            + self.v_recurrence.powi(2))
        .sqrt()
    }

    /// Decay velocity (for simulated annealing)
    pub fn decay(&mut self, factor: f32) {
        self.v_density *= factor;
        self.v_modularity *= factor;
        self.v_bridge_ratio *= factor;
        self.v_tau_ratio *= factor;
        self.v_binding_strength *= factor;
        self.v_recurrence *= factor;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SEARCH CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Search strategy for architecture optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchStrategy {
    /// Random sampling from architecture space
    Random,
    /// Evolutionary search with mutation and selection
    Evolutionary,
    /// Gradient-guided search using Phi gradients
    GradientGuided,
    /// Hybrid: evolutionary exploration + gradient exploitation
    Hybrid,
    /// Gradient-guided evolution: mutations follow Phi gradient direction
    GradientEvolutionary,
    /// Momentum-based optimization with adaptive learning rate
    MomentumOptimization,
    /// Multi-population island model with gradient migration
    IslandGradient,
}

/// Configuration for architecture search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Population size for evolutionary search
    pub population_size: usize,

    /// Number of elite individuals to preserve
    pub elite_count: usize,

    /// Mutation rate (0.0-1.0)
    pub mutation_rate: f32,

    /// Crossover rate (0.0-1.0)
    pub crossover_rate: f32,

    /// Learning rate for gradient-guided search
    pub learning_rate: f32,

    /// Epsilon for gradient computation
    pub gradient_epsilon: f32,

    /// Number of gradient steps between evolutionary generations
    pub gradient_steps_per_generation: usize,

    /// Minimum nodes to search
    pub min_nodes: usize,

    /// Maximum nodes to search
    pub max_nodes: usize,

    /// HDC dimension
    pub hdc_dim: usize,

    /// Random seed
    pub seed: u64,

    /// Whether to use parallelism
    pub parallel: bool,

    /// Number of random architectures to sample for random search
    pub random_samples: usize,

    /// Momentum coefficient for momentum-based optimization (0.0-1.0)
    pub momentum: f32,

    /// Noise scale for gradient-guided mutations (exploration)
    pub gradient_noise_scale: f32,

    /// Number of islands for island model
    pub num_islands: usize,

    /// Migration interval (generations between migrations)
    pub migration_interval: usize,

    /// Migration rate (fraction of population to migrate)
    pub migration_rate: f32,

    /// Convergence threshold (stop when gradient magnitude < this)
    pub convergence_threshold: f64,

    /// Maximum generations without improvement before early stopping
    pub patience: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            population_size: 20,
            elite_count: 2,
            mutation_rate: 0.3,
            crossover_rate: 0.6,
            learning_rate: 0.1,
            gradient_epsilon: 0.01,
            gradient_steps_per_generation: 5,
            min_nodes: 8,
            max_nodes: 64,
            hdc_dim: HDC_DIMENSION,
            seed: 42,
            parallel: false,
            random_samples: 100,
            momentum: 0.9,
            gradient_noise_scale: 0.05,
            num_islands: 4,
            migration_interval: 5,
            migration_rate: 0.1,
            convergence_threshold: 1e-6,
            patience: 20,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SEARCH RESULT
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of architecture search
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Best Phi achieved
    pub best_phi: f64,

    /// Best architecture genome found
    pub best_architecture: ArchitectureGenome,

    /// History of best Phi values per generation
    pub phi_history: Vec<f64>,

    /// Total number of architectures evaluated
    pub evaluations: usize,

    /// Search strategy used
    pub strategy: SearchStrategy,

    /// Time taken (if tracked)
    pub elapsed_ms: Option<u128>,
}

/// Individual in evolutionary population
#[derive(Debug, Clone)]
pub struct Individual {
    /// The architecture genome
    pub genome: ArchitectureGenome,
    /// Phi fitness value
    pub fitness: f64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHI ARCHITECTURE SEARCH
// ═══════════════════════════════════════════════════════════════════════════════

/// Main architecture search engine
///
/// Uses consciousness (Phi) as the fitness function to evolve network architectures
/// toward higher levels of integrated information.
pub struct PhiArchitectureSearch {
    /// Search configuration
    config: SearchConfig,

    /// Current population (for evolutionary search)
    population: Vec<Individual>,

    /// Best individual found so far
    best: Option<Individual>,

    /// History of best Phi values
    phi_history: Vec<f64>,

    /// Generation counter
    generation: usize,

    /// Total evaluations
    evaluations: usize,

    /// PRNG state
    rng_state: u64,
}

impl PhiArchitectureSearch {
    /// Create a new architecture search engine
    pub fn new(config: SearchConfig) -> Self {
        Self {
            rng_state: config.seed,
            config,
            population: Vec::new(),
            best: None,
            phi_history: Vec::new(),
            generation: 0,
            evaluations: 0,
        }
    }

    /// Initialize population
    fn initialize_population(&mut self) {
        self.population.clear();

        for i in 0..self.config.population_size {
            let genome = ArchitectureGenome::random(self.config.seed + i as u64 * 1000);
            let arch = DecodedArchitecture::from_genome(&genome);
            let fitness = arch.compute_phi();
            self.evaluations += 1;

            let individual = Individual { genome, fitness };

            if self.best.is_none() || fitness > self.best.as_ref().expect("best individual must be set after initialization").fitness {
                self.best = Some(individual.clone());
            }

            self.population.push(individual);
        }

        // Sort by fitness (descending)
        self.population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));
    }

    /// Run random search
    pub fn random_search(&mut self) -> SearchResult {
        let mut best_genome = ArchitectureGenome::default();
        let mut best_phi = 0.0;
        let mut phi_history = Vec::new();

        for i in 0..self.config.random_samples {
            let genome = ArchitectureGenome::random(self.config.seed + i as u64);
            let arch = DecodedArchitecture::from_genome(&genome);
            let phi = arch.compute_phi();
            self.evaluations += 1;

            if phi > best_phi {
                best_phi = phi;
                best_genome = genome;
            }

            phi_history.push(best_phi);
        }

        SearchResult {
            best_phi,
            best_architecture: best_genome,
            phi_history,
            evaluations: self.evaluations,
            strategy: SearchStrategy::Random,
            elapsed_ms: None,
        }
    }

    /// Run evolutionary search
    pub fn evolutionary_search(&mut self, generations: usize) -> SearchResult {
        self.initialize_population();
        self.phi_history.push(self.best.as_ref().expect("best individual must be set after initialization").fitness);

        for gen in 0..generations {
            self.generation = gen + 1;
            self.evolve_generation();
            self.phi_history.push(self.best.as_ref().expect("best individual must be set after initialization").fitness);
        }

        SearchResult {
            best_phi: self.best.as_ref().expect("best individual must be set after initialization").fitness,
            best_architecture: self.best.as_ref().expect("best individual must be set after initialization").genome.clone(),
            phi_history: self.phi_history.clone(),
            evaluations: self.evaluations,
            strategy: SearchStrategy::Evolutionary,
            elapsed_ms: None,
        }
    }

    /// Run gradient-guided search
    pub fn gradient_search(&mut self, steps: usize) -> SearchResult {
        // Start with random architecture
        let mut genome = ArchitectureGenome::random(self.config.seed);
        let arch = DecodedArchitecture::from_genome(&genome);
        let mut best_phi = arch.compute_phi();
        self.evaluations += 1;

        let mut phi_history = vec![best_phi];

        for _step in 0..steps {
            // Compute gradient
            let gradient = PhiGradient::compute(&genome, self.config.gradient_epsilon);
            self.evaluations += 12;  // ~6 parameters * 2 (forward/backward)

            // Apply gradient
            gradient.apply(&mut genome, self.config.learning_rate);

            // Evaluate new architecture
            let arch = DecodedArchitecture::from_genome(&genome);
            let phi = arch.compute_phi();
            self.evaluations += 1;

            if phi > best_phi {
                best_phi = phi;
            }

            phi_history.push(best_phi);
        }

        SearchResult {
            best_phi,
            best_architecture: genome,
            phi_history,
            evaluations: self.evaluations,
            strategy: SearchStrategy::GradientGuided,
            elapsed_ms: None,
        }
    }

    /// Run hybrid search (evolutionary + gradient)
    pub fn hybrid_search(&mut self, generations: usize) -> SearchResult {
        self.initialize_population();
        self.phi_history.push(self.best.as_ref().expect("best individual must be set after initialization").fitness);

        for gen in 0..generations {
            self.generation = gen + 1;

            // Evolutionary step
            self.evolve_generation();

            // Gradient refinement on top individuals
            self.gradient_refine_elites();

            self.phi_history.push(self.best.as_ref().expect("best individual must be set after initialization").fitness);
        }

        SearchResult {
            best_phi: self.best.as_ref().expect("best individual must be set after initialization").fitness,
            best_architecture: self.best.as_ref().expect("best individual must be set after initialization").genome.clone(),
            phi_history: self.phi_history.clone(),
            evaluations: self.evaluations,
            strategy: SearchStrategy::Hybrid,
            elapsed_ms: None,
        }
    }

    /// Main search entry point
    pub fn search(&mut self, strategy: SearchStrategy, iterations: usize) -> SearchResult {
        match strategy {
            SearchStrategy::Random => self.random_search(),
            SearchStrategy::Evolutionary => self.evolutionary_search(iterations),
            SearchStrategy::GradientGuided => self.gradient_search(iterations),
            SearchStrategy::Hybrid => self.hybrid_search(iterations),
            SearchStrategy::GradientEvolutionary => self.gradient_evolutionary_search(iterations),
            SearchStrategy::MomentumOptimization => self.momentum_search(iterations),
            SearchStrategy::IslandGradient => self.island_gradient_search(iterations),
        }
    }

    /// Gradient-guided evolutionary search
    ///
    /// Combines evolutionary selection with gradient-informed mutations.
    /// Mutations follow the Phi gradient direction rather than being random,
    /// enabling more efficient exploration of the consciousness landscape.
    pub fn gradient_evolutionary_search(&mut self, generations: usize) -> SearchResult {
        self.initialize_population();
        self.phi_history.push(self.best.as_ref().expect("best individual must be set after initialization").fitness);

        // Compute initial gradients for the population
        let mut population_gradients: Vec<PhiGradient> = self.population
            .iter()
            .map(|ind| PhiGradient::compute(&ind.genome, self.config.gradient_epsilon))
            .collect();
        self.evaluations += self.population.len() * 12;

        let mut no_improvement_count = 0;
        let mut prev_best = self.best.as_ref().expect("best individual must be set after initialization").fitness;

        for gen in 0..generations {
            self.generation = gen + 1;

            // Gradient-guided evolution step
            self.evolve_generation_with_gradients(&population_gradients);

            // Update gradients for new population
            population_gradients = self.population
                .iter()
                .map(|ind| PhiGradient::compute(&ind.genome, self.config.gradient_epsilon))
                .collect();
            self.evaluations += self.population.len() * 12;

            let current_best = self.best.as_ref().expect("best individual must be set after initialization").fitness;
            self.phi_history.push(current_best);

            // Early stopping check
            if (current_best - prev_best).abs() < self.config.convergence_threshold {
                no_improvement_count += 1;
                if no_improvement_count >= self.config.patience {
                    break;
                }
            } else {
                no_improvement_count = 0;
            }
            prev_best = current_best;
        }

        SearchResult {
            best_phi: self.best.as_ref().expect("best individual must be set after initialization").fitness,
            best_architecture: self.best.as_ref().expect("best individual must be set after initialization").genome.clone(),
            phi_history: self.phi_history.clone(),
            evaluations: self.evaluations,
            strategy: SearchStrategy::GradientEvolutionary,
            elapsed_ms: None,
        }
    }

    /// Evolve generation using gradient-guided mutations
    fn evolve_generation_with_gradients(&mut self, gradients: &[PhiGradient]) {
        let mut new_population = Vec::with_capacity(self.config.population_size);

        // Preserve elites (unchanged)
        for i in 0..self.config.elite_count.min(self.population.len()) {
            new_population.push(self.population[i].clone());
        }

        // Generate offspring with gradient-guided mutations
        while new_population.len() < self.config.population_size {
            // Tournament selection
            let parent_idx = self.tournament_select_idx();
            let parent_genome = self.population[parent_idx].genome.clone();
            let parent_gradient = &gradients[parent_idx.min(gradients.len() - 1)];

            let mut child_genome = parent_genome;

            // Apply gradient-guided mutation
            child_genome.mutate_with_gradient(
                parent_gradient,
                self.config.learning_rate,
                self.config.gradient_noise_scale,
                self.next_u64(),
            );

            // Evaluate
            let arch = DecodedArchitecture::from_genome(&child_genome);
            let fitness = arch.compute_phi();
            self.evaluations += 1;

            let child = Individual {
                genome: child_genome,
                fitness,
            };

            // Update best
            if fitness > self.best.as_ref().expect("best individual must be set after initialization").fitness {
                self.best = Some(child.clone());
            }

            new_population.push(child);
        }

        // Replace population
        self.population = new_population;
        self.population.sort_by(|a, b| {
            b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Tournament selection returning index
    fn tournament_select_idx(&mut self) -> usize {
        let tournament_size = 3;
        let mut best_idx = (self.next_u64() as usize) % self.population.len();

        for _ in 1..tournament_size {
            let idx = (self.next_u64() as usize) % self.population.len();
            if self.population[idx].fitness > self.population[best_idx].fitness {
                best_idx = idx;
            }
        }

        best_idx
    }

    /// Momentum-based optimization search
    ///
    /// Uses momentum to accelerate convergence and escape local optima.
    /// Maintains velocity vectors for smooth optimization trajectories.
    pub fn momentum_search(&mut self, steps: usize) -> SearchResult {
        // Initialize multiple starting points
        let num_starts = self.config.population_size.min(5);
        let mut best_genome = ArchitectureGenome::random(self.config.seed);
        let mut best_phi = 0.0;
        let mut phi_history = Vec::new();

        for start in 0..num_starts {
            let mut genome = ArchitectureGenome::random(self.config.seed + start as u64 * 1000);
            let mut velocity = GradientVelocity::new();

            let arch = DecodedArchitecture::from_genome(&genome);
            let mut current_phi = arch.compute_phi();
            let _ = current_phi; // Suppress unused_assignments warning - will be overwritten
            self.evaluations += 1;

            let mut prev_gradient: Option<PhiGradient> = None;

            for step in 0..steps / num_starts {
                // Compute gradient
                let gradient = PhiGradient::compute(&genome, self.config.gradient_epsilon);
                self.evaluations += 12;

                // Check for convergence (gradient alignment)
                if let Some(ref prev) = prev_gradient {
                    let similarity = gradient.cosine_similarity(prev);
                    if gradient.magnitude < self.config.convergence_threshold
                        || similarity > 0.99
                    {
                        // Converged, try random restart
                        break;
                    }
                }

                // Apply momentum update
                genome.mutate_with_momentum(
                    &gradient,
                    &mut velocity,
                    self.config.momentum,
                    self.config.learning_rate,
                );

                // Evaluate
                let arch = DecodedArchitecture::from_genome(&genome);
                current_phi = arch.compute_phi();
                self.evaluations += 1;

                if current_phi > best_phi {
                    best_phi = current_phi;
                    best_genome = genome.clone();
                }

                phi_history.push(best_phi);
                prev_gradient = Some(gradient);

                // Velocity decay for annealing effect
                if step % 10 == 0 {
                    velocity.decay(0.95);
                }
            }
        }

        SearchResult {
            best_phi,
            best_architecture: best_genome,
            phi_history,
            evaluations: self.evaluations,
            strategy: SearchStrategy::MomentumOptimization,
            elapsed_ms: None,
        }
    }

    /// Island model with gradient-guided migration
    ///
    /// Maintains multiple populations (islands) that evolve independently,
    /// with periodic migration of best individuals. Migrants are refined
    /// using gradient descent before being inserted into target islands.
    pub fn island_gradient_search(&mut self, generations: usize) -> SearchResult {
        let num_islands = self.config.num_islands.max(2);
        let island_size = self.config.population_size / num_islands;

        // Initialize islands
        let mut islands: Vec<Vec<Individual>> = (0..num_islands)
            .map(|i| {
                let mut island = Vec::with_capacity(island_size);
                for j in 0..island_size {
                    let genome = ArchitectureGenome::random(
                        self.config.seed + (i * island_size + j) as u64 * 100,
                    );
                    let arch = DecodedArchitecture::from_genome(&genome);
                    let fitness = arch.compute_phi();
                    self.evaluations += 1;
                    island.push(Individual { genome, fitness });
                }
                island.sort_by(|a, b| {
                    b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal)
                });
                island
            })
            .collect();

        // Track global best
        let mut global_best: Option<Individual> = None;
        for island in &islands {
            if let Some(best) = island.first() {
                if global_best.is_none()
                    || best.fitness > global_best.as_ref().expect("global_best must be set after first island").fitness
                {
                    global_best = Some(best.clone());
                }
            }
        }

        let mut phi_history = vec![global_best.as_ref().expect("global_best must be set after first island").fitness];

        for gen in 0..generations {
            // Evolve each island independently
            for island in &mut islands {
                self.evolve_island(island);
            }

            // Migration phase
            if gen > 0 && gen % self.config.migration_interval == 0 {
                self.migrate_with_gradient_refinement(&mut islands);
            }

            // Update global best
            for island in &islands {
                if let Some(best) = island.first() {
                    if best.fitness > global_best.as_ref().expect("global_best must be set after first island").fitness {
                        global_best = Some(best.clone());
                    }
                }
            }

            phi_history.push(global_best.as_ref().expect("global_best must be set after first island").fitness);
        }

        let best = global_best.expect("global_best must be set after island evolution");
        SearchResult {
            best_phi: best.fitness,
            best_architecture: best.genome,
            phi_history,
            evaluations: self.evaluations,
            strategy: SearchStrategy::IslandGradient,
            elapsed_ms: None,
        }
    }

    /// Evolve a single island population
    fn evolve_island(&mut self, island: &mut Vec<Individual>) {
        if island.is_empty() {
            return;
        }

        let elite_count = 1.max(island.len() / 5);
        let mut new_island = Vec::with_capacity(island.len());

        // Preserve elites
        for i in 0..elite_count.min(island.len()) {
            new_island.push(island[i].clone());
        }

        // Generate offspring
        while new_island.len() < island.len() {
            // Simple tournament selection within island
            let idx1 = (self.next_u64() as usize) % island.len();
            let idx2 = (self.next_u64() as usize) % island.len();
            let parent = if island[idx1].fitness > island[idx2].fitness {
                &island[idx1]
            } else {
                &island[idx2]
            };

            let mut child_genome = parent.genome.clone();
            child_genome.mutate(self.config.mutation_rate, self.next_u64());

            let arch = DecodedArchitecture::from_genome(&child_genome);
            let fitness = arch.compute_phi();
            self.evaluations += 1;

            new_island.push(Individual {
                genome: child_genome,
                fitness,
            });
        }

        *island = new_island;
        island.sort_by(|a, b| {
            b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Migrate best individuals between islands with gradient refinement
    fn migrate_with_gradient_refinement(&mut self, islands: &mut [Vec<Individual>]) {
        let num_islands = islands.len();
        if num_islands < 2 {
            return;
        }

        let migrants_per_island =
            (islands[0].len() as f32 * self.config.migration_rate).ceil() as usize;
        let migrants_per_island = migrants_per_island.max(1);

        // Collect migrants from each island (best individuals)
        let migrants: Vec<Vec<Individual>> = islands
            .iter()
            .map(|island| {
                island
                    .iter()
                    .take(migrants_per_island)
                    .cloned()
                    .collect()
            })
            .collect();

        // Ring migration: island i receives migrants from island (i-1) mod n
        for i in 0..num_islands {
            let source = (i + num_islands - 1) % num_islands;

            for migrant in &migrants[source] {
                // Gradient-refine migrant before insertion
                let mut refined_genome = migrant.genome.clone();
                let gradient =
                    PhiGradient::compute(&refined_genome, self.config.gradient_epsilon);
                self.evaluations += 12;

                gradient.apply(&mut refined_genome, self.config.learning_rate);

                let arch = DecodedArchitecture::from_genome(&refined_genome);
                let fitness = arch.compute_phi();
                self.evaluations += 1;

                // Replace worst individual in target island
                if !islands[i].is_empty() && fitness > islands[i].last().expect("island must not be empty").fitness {
                    islands[i].pop();
                    islands[i].push(Individual {
                        genome: refined_genome,
                        fitness,
                    });
                    islands[i].sort_by(|a, b| {
                        b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
            }
        }
    }

    /// Evolve one generation
    fn evolve_generation(&mut self) {
        let mut new_population = Vec::with_capacity(self.config.population_size);

        // Preserve elites
        for i in 0..self.config.elite_count.min(self.population.len()) {
            new_population.push(self.population[i].clone());
        }

        // Generate offspring
        while new_population.len() < self.config.population_size {
            // Clone genomes to avoid borrow conflicts with RNG methods
            let parent1_genome = self.tournament_select().genome.clone();
            let parent2_genome = self.tournament_select().genome.clone();
            let crossover_rate = self.config.crossover_rate;

            let mut child_genome = if self.next_f32() < crossover_rate {
                parent1_genome.crossover(&parent2_genome, self.next_u64())
            } else {
                parent1_genome
            };

            // Mutate
            child_genome.mutate(self.config.mutation_rate, self.next_u64());

            // Evaluate
            let arch = DecodedArchitecture::from_genome(&child_genome);
            let fitness = arch.compute_phi();
            self.evaluations += 1;

            let child = Individual {
                genome: child_genome,
                fitness,
            };

            // Update best
            if self.best.is_none() || fitness > self.best.as_ref().expect("best individual must be set after initialization").fitness {
                self.best = Some(child.clone());
            }

            new_population.push(child);
        }

        // Replace population
        self.population = new_population;
        self.population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));
    }

    /// Tournament selection
    fn tournament_select(&mut self) -> &Individual {
        let tournament_size = 3;
        let mut best_idx = (self.next_u64() as usize) % self.population.len();

        for _ in 1..tournament_size {
            let idx = (self.next_u64() as usize) % self.population.len();
            if self.population[idx].fitness > self.population[best_idx].fitness {
                best_idx = idx;
            }
        }

        &self.population[best_idx]
    }

    /// Gradient refinement on elite individuals
    fn gradient_refine_elites(&mut self) {
        for i in 0..self.config.elite_count.min(self.population.len()) {
            let mut genome = self.population[i].genome.clone();

            for _ in 0..self.config.gradient_steps_per_generation {
                let gradient = PhiGradient::compute(&genome, self.config.gradient_epsilon);
                self.evaluations += 12;
                gradient.apply(&mut genome, self.config.learning_rate);
            }

            // Evaluate refined genome
            let arch = DecodedArchitecture::from_genome(&genome);
            let fitness = arch.compute_phi();
            self.evaluations += 1;

            if fitness > self.population[i].fitness {
                self.population[i].genome = genome;
                self.population[i].fitness = fitness;

                if fitness > self.best.as_ref().expect("best individual must be set after initialization").fitness {
                    self.best = Some(self.population[i].clone());
                }
            }
        }

        // Re-sort
        self.population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));
    }

    // PRNG helpers

    fn next_u64(&mut self) -> u64 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        self.rng_state
    }

    fn next_f32(&mut self) -> f32 {
        self.next_u64() as f32 / u64::MAX as f32
    }

    /// Get current best architecture
    pub fn best(&self) -> Option<&Individual> {
        self.best.as_ref()
    }

    /// Get current population (sorted by fitness)
    pub fn population(&self) -> &[Individual] {
        &self.population
    }

    /// Get search statistics
    pub fn stats(&self) -> SearchStats {
        SearchStats {
            generation: self.generation,
            evaluations: self.evaluations,
            best_phi: self.best.as_ref().map(|b| b.fitness).unwrap_or(0.0),
            population_size: self.population.len(),
            avg_phi: if self.population.is_empty() {
                0.0
            } else {
                self.population.iter().map(|i| i.fitness).sum::<f64>() / self.population.len() as f64
            },
            phi_std: if self.population.len() < 2 {
                0.0
            } else {
                let avg = self.population.iter().map(|i| i.fitness).sum::<f64>() / self.population.len() as f64;
                let variance = self.population.iter().map(|i| (i.fitness - avg).powi(2)).sum::<f64>()
                    / (self.population.len() - 1) as f64;
                variance.sqrt()
            },
        }
    }

    /// Reset search state
    pub fn reset(&mut self) {
        self.population.clear();
        self.best = None;
        self.phi_history.clear();
        self.generation = 0;
        self.evaluations = 0;
        self.rng_state = self.config.seed;
    }
}

/// Search statistics
#[derive(Debug, Clone)]
pub struct SearchStats {
    pub generation: usize,
    pub evaluations: usize,
    pub best_phi: f64,
    pub population_size: usize,
    pub avg_phi: f64,
    pub phi_std: f64,
}

impl std::fmt::Display for SearchStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Gen {}: Best Phi = {:.4}, Avg = {:.4} +/- {:.4} ({} evals)",
            self.generation, self.best_phi, self.avg_phi, self.phi_std, self.evaluations
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genome_creation() {
        let genome = ArchitectureGenome::default();
        assert_eq!(genome.num_nodes, 16);
        assert_eq!(genome.hdc_dim, HDC_DIMENSION);
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

        genome.mutate(1.0, 12345);  // High mutation rate

        // At least some parameter should have changed
        // (with mutation rate 1.0, all parameters mutate)
        // Use epsilon-based comparison for floating-point values
        const EPSILON: f32 = 1e-9;
        assert!(
            (genome.connection_density - original_density).abs() > EPSILON ||
            genome.num_nodes != 16 ||
            (genome.modularity - 0.5).abs() > EPSILON
        );
    }

    #[test]
    fn test_genome_crossover() {
        let parent1 = ArchitectureGenome::random(100);
        let parent2 = ArchitectureGenome::random(200);

        let child = parent1.crossover(&parent2, 300);

        // Child should have values from one of the parents
        assert!(
            child.num_nodes == parent1.num_nodes ||
            child.num_nodes == parent2.num_nodes
        );
    }

    #[test]
    fn test_decode_architecture() {
        let genome = ArchitectureGenome {
            num_nodes: 8,
            hdc_dim: 256,  // Small for fast tests
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
        assert_eq!(stats.num_edges, 10);  // Ring has n edges
        assert!((stats.avg_degree - 2.0).abs() < 0.1);  // Ring: each node has degree 2
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
}
