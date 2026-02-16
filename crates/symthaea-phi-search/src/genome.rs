//! Architecture genome representation for Phi-guided search.

use serde::{Deserialize, Serialize};

use symthaea_core::hdc::HDC_DIMENSION;

use super::phi_gradient::{GradientVelocity, PhiGradient};

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
            num_nodes: 8 + next_usize(&mut state, 57), // 8-64 nodes
            hierarchy_depth: 2 + next_usize(&mut state, 6), // 2-7 levels
            base_tau: 100.0 + next_f32(&mut state) * 1900.0, // 100-2000ms
            tau_ratio: 0.2 + next_f32(&mut state) * 0.6, // 0.2-0.8
            connection_density: 0.1 + next_f32(&mut state) * 0.7, // 0.1-0.8
            modularity: next_f32(&mut state),          // 0.0-1.0
            num_modules: 2 + next_usize(&mut state, 7), // 2-8 modules
            bridge_ratio: 0.05 + next_f32(&mut state) * 0.5, // 0.05-0.55
            topology_type: TopologyGene::random(state),
            binding_strength: 0.3 + next_f32(&mut state) * 0.7, // 0.3-1.0
            bundling_mode: BundlingGene::random(state),
            recurrence: next_f32(&mut state), // 0.0-1.0
            skip_connection_prob: next_f32(&mut state) * 0.3, // 0.0-0.3
            use_attention: next_f32(&mut state) > 0.5,
            hdc_dim: HDC_DIMENSION,
            seed: state,
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
            self.base_tau *= 0.8 + next_f32(&mut state) * 0.4; // +-20%
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
        velocity.v_density =
            momentum * velocity.v_density + learning_rate * gradient.d_density as f32;
        velocity.v_modularity =
            momentum * velocity.v_modularity + learning_rate * gradient.d_modularity as f32;
        velocity.v_bridge_ratio =
            momentum * velocity.v_bridge_ratio + learning_rate * gradient.d_bridge_ratio as f32;
        velocity.v_tau_ratio =
            momentum * velocity.v_tau_ratio + learning_rate * gradient.d_tau_ratio as f32;
        velocity.v_binding_strength = momentum * velocity.v_binding_strength
            + learning_rate * gradient.d_binding_strength as f32;
        velocity.v_recurrence =
            momentum * velocity.v_recurrence + learning_rate * gradient.d_recurrence as f32;

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
            *s as usize % 2 == 0
        };

        Self {
            num_nodes: if choose(&mut state) {
                self.num_nodes
            } else {
                other.num_nodes
            },
            hierarchy_depth: if choose(&mut state) {
                self.hierarchy_depth
            } else {
                other.hierarchy_depth
            },
            base_tau: if choose(&mut state) {
                self.base_tau
            } else {
                other.base_tau
            },
            tau_ratio: if choose(&mut state) {
                self.tau_ratio
            } else {
                other.tau_ratio
            },
            connection_density: if choose(&mut state) {
                self.connection_density
            } else {
                other.connection_density
            },
            modularity: if choose(&mut state) {
                self.modularity
            } else {
                other.modularity
            },
            num_modules: if choose(&mut state) {
                self.num_modules
            } else {
                other.num_modules
            },
            bridge_ratio: if choose(&mut state) {
                self.bridge_ratio
            } else {
                other.bridge_ratio
            },
            topology_type: if choose(&mut state) {
                self.topology_type
            } else {
                other.topology_type
            },
            binding_strength: if choose(&mut state) {
                self.binding_strength
            } else {
                other.binding_strength
            },
            bundling_mode: if choose(&mut state) {
                self.bundling_mode
            } else {
                other.bundling_mode
            },
            recurrence: if choose(&mut state) {
                self.recurrence
            } else {
                other.recurrence
            },
            skip_connection_prob: if choose(&mut state) {
                self.skip_connection_prob
            } else {
                other.skip_connection_prob
            },
            use_attention: if choose(&mut state) {
                self.use_attention
            } else {
                other.use_attention
            },
            hdc_dim: self.hdc_dim, // Always inherit from first parent
            seed,
        }
    }
}
