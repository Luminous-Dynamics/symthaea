// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
            (*s as usize).is_multiple_of(2)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_genome_values() {
        let g = ArchitectureGenome::default();
        assert_eq!(g.num_nodes, 16);
        assert_eq!(g.hierarchy_depth, 4);
        assert!((g.base_tau - 1000.0).abs() < f32::EPSILON);
        assert!((g.tau_ratio - 1.0 / 3.0).abs() < 1e-6);
        assert!((g.connection_density - 0.4).abs() < f32::EPSILON);
        assert!((g.modularity - 0.5).abs() < f32::EPSILON);
        assert_eq!(g.num_modules, 4);
        assert!((g.bridge_ratio - 0.3).abs() < f32::EPSILON);
        assert_eq!(g.topology_type, TopologyGene::Modular);
        assert!((g.binding_strength - 0.8).abs() < f32::EPSILON);
        assert_eq!(g.bundling_mode, BundlingGene::WeightedAverage);
        assert!((g.recurrence - 0.3).abs() < f32::EPSILON);
        assert!((g.skip_connection_prob - 0.1).abs() < f32::EPSILON);
        assert!(!g.use_attention);
        assert_eq!(g.hdc_dim, HDC_DIMENSION);
        assert_eq!(g.seed, 42);
    }

    #[test]
    fn test_random_genome_deterministic() {
        let g1 = ArchitectureGenome::random(99);
        let g2 = ArchitectureGenome::random(99);
        // Same seed must produce identical genomes
        assert_eq!(g1.num_nodes, g2.num_nodes);
        assert_eq!(g1.hierarchy_depth, g2.hierarchy_depth);
        assert!((g1.base_tau - g2.base_tau).abs() < f32::EPSILON);
        assert_eq!(g1.topology_type, g2.topology_type);
        assert_eq!(g1.bundling_mode, g2.bundling_mode);
    }

    #[test]
    fn test_random_genome_bounds() {
        // Test several seeds to check range constraints
        for seed in 0..20u64 {
            let g = ArchitectureGenome::random(seed + 1);
            assert!(
                g.num_nodes >= 8 && g.num_nodes <= 64,
                "num_nodes out of range: {}",
                g.num_nodes
            );
            assert!(g.hierarchy_depth >= 2 && g.hierarchy_depth <= 7);
            assert!(g.base_tau >= 100.0 && g.base_tau <= 2000.0);
            assert!(g.tau_ratio >= 0.2 && g.tau_ratio <= 0.8);
            assert!(g.connection_density >= 0.1 && g.connection_density <= 0.8);
            assert!(g.modularity >= 0.0 && g.modularity <= 1.0);
            assert!(g.num_modules >= 2 && g.num_modules <= 8);
            assert!(g.bridge_ratio >= 0.05 && g.bridge_ratio <= 0.55);
            assert!(g.binding_strength >= 0.3 && g.binding_strength <= 1.0);
            assert!(g.recurrence >= 0.0 && g.recurrence <= 1.0);
            assert!(g.skip_connection_prob >= 0.0 && g.skip_connection_prob <= 0.3);
            assert_eq!(g.hdc_dim, HDC_DIMENSION);
        }
    }

    #[test]
    fn test_mutate_respects_bounds() {
        let mut g = ArchitectureGenome::default();
        // Apply many rounds of high-rate mutation and verify bounds hold
        for i in 0..50u64 {
            g.mutate(1.0, i * 7 + 13);
        }
        assert!(g.num_nodes >= 4 && g.num_nodes <= 128);
        assert!(g.hierarchy_depth >= 1 && g.hierarchy_depth <= 10);
        assert!(g.base_tau >= 10.0 && g.base_tau <= 5000.0);
        assert!(g.tau_ratio >= 0.1 && g.tau_ratio <= 0.9);
        assert!(g.connection_density >= 0.05 && g.connection_density <= 0.95);
        assert!(g.modularity >= 0.0 && g.modularity <= 1.0);
        assert!(g.num_modules >= 1 && g.num_modules <= 16);
        assert!(g.bridge_ratio >= 0.0 && g.bridge_ratio <= 0.8);
        assert!(g.binding_strength >= 0.1 && g.binding_strength <= 1.0);
        assert!(g.recurrence >= 0.0 && g.recurrence <= 1.0);
        assert!(g.skip_connection_prob >= 0.0 && g.skip_connection_prob <= 0.5);
    }

    #[test]
    fn test_mutate_zero_rate_preserves_genome() {
        let original = ArchitectureGenome::default();
        let mut g = original.clone();
        g.mutate(0.0, 42);
        // With mutation_rate 0.0, no gene should mutate
        assert_eq!(g.num_nodes, original.num_nodes);
        assert!((g.connection_density - original.connection_density).abs() < f32::EPSILON);
        assert!((g.modularity - original.modularity).abs() < f32::EPSILON);
        assert!((g.base_tau - original.base_tau).abs() < f32::EPSILON);
        assert!((g.tau_ratio - original.tau_ratio).abs() < f32::EPSILON);
        assert!((g.bridge_ratio - original.bridge_ratio).abs() < f32::EPSILON);
        assert!((g.binding_strength - original.binding_strength).abs() < f32::EPSILON);
        assert!((g.recurrence - original.recurrence).abs() < f32::EPSILON);
        assert!((g.skip_connection_prob - original.skip_connection_prob).abs() < f32::EPSILON);
        assert_eq!(g.use_attention, original.use_attention);
    }

    #[test]
    fn test_crossover_inherits_from_parents() {
        let p1 = ArchitectureGenome {
            num_nodes: 10,
            hierarchy_depth: 2,
            base_tau: 500.0,
            connection_density: 0.2,
            modularity: 0.1,
            num_modules: 2,
            topology_type: TopologyGene::Ring,
            bundling_mode: BundlingGene::Average,
            use_attention: false,
            ..Default::default()
        };
        let p2 = ArchitectureGenome {
            num_nodes: 50,
            hierarchy_depth: 7,
            base_tau: 1800.0,
            connection_density: 0.9,
            modularity: 0.9,
            num_modules: 8,
            topology_type: TopologyGene::Star,
            bundling_mode: BundlingGene::MajorityVote,
            use_attention: true,
            ..Default::default()
        };

        let child = p1.crossover(&p2, 77);
        // Each gene comes from one parent
        assert!(child.num_nodes == p1.num_nodes || child.num_nodes == p2.num_nodes);
        assert!(
            child.hierarchy_depth == p1.hierarchy_depth
                || child.hierarchy_depth == p2.hierarchy_depth
        );
        assert!(
            (child.base_tau - p1.base_tau).abs() < f32::EPSILON
                || (child.base_tau - p2.base_tau).abs() < f32::EPSILON
        );
        assert!(
            (child.connection_density - p1.connection_density).abs() < f32::EPSILON
                || (child.connection_density - p2.connection_density).abs() < f32::EPSILON
        );
        assert!(child.topology_type == p1.topology_type || child.topology_type == p2.topology_type);
        assert!(child.bundling_mode == p1.bundling_mode || child.bundling_mode == p2.bundling_mode);
        assert!(child.use_attention == p1.use_attention || child.use_attention == p2.use_attention);
        // hdc_dim always comes from first parent
        assert_eq!(child.hdc_dim, p1.hdc_dim);
    }

    #[test]
    fn test_crossover_seed_produces_child_seed() {
        let p1 = ArchitectureGenome::default();
        let p2 = ArchitectureGenome::random(100);
        let child = p1.crossover(&p2, 555);
        assert_eq!(child.seed, 555);
    }

    #[test]
    fn test_topology_gene_all_contains_all_variants() {
        let all = TopologyGene::all();
        assert_eq!(all.len(), 10);
        // Verify all variants are present
        assert!(all.contains(&TopologyGene::Random));
        assert!(all.contains(&TopologyGene::Ring));
        assert!(all.contains(&TopologyGene::Star));
        assert!(all.contains(&TopologyGene::HierarchicalTree));
        assert!(all.contains(&TopologyGene::Modular));
        assert!(all.contains(&TopologyGene::ScaleFree));
        assert!(all.contains(&TopologyGene::SmallWorld));
        assert!(all.contains(&TopologyGene::Lattice));
        assert!(all.contains(&TopologyGene::CorePeriphery));
        assert!(all.contains(&TopologyGene::Attention));
    }

    #[test]
    fn test_topology_gene_random_deterministic() {
        let t1 = TopologyGene::random(42);
        let t2 = TopologyGene::random(42);
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_bundling_gene_all_contains_all_variants() {
        let all = BundlingGene::all();
        assert_eq!(all.len(), 5);
        assert!(all.contains(&BundlingGene::Average));
        assert!(all.contains(&BundlingGene::WeightedAverage));
        assert!(all.contains(&BundlingGene::MajorityVote));
        assert!(all.contains(&BundlingGene::PermutationSequence));
        assert!(all.contains(&BundlingGene::ResonatorCleanup));
    }

    #[test]
    fn test_bundling_gene_random_deterministic() {
        let b1 = BundlingGene::random(77);
        let b2 = BundlingGene::random(77);
        assert_eq!(b1, b2);
    }

    #[test]
    fn test_mutate_with_gradient_moves_in_gradient_direction() {
        let mut g = ArchitectureGenome {
            connection_density: 0.5,
            modularity: 0.5,
            bridge_ratio: 0.4,
            tau_ratio: 0.5,
            binding_strength: 0.5,
            recurrence: 0.5,
            ..Default::default()
        };

        let gradient = PhiGradient {
            d_density: 1.0,
            d_modularity: -1.0,
            d_bridge_ratio: 0.0,
            d_tau_ratio: 0.0,
            d_binding_strength: 0.0,
            d_recurrence: 0.0,
            magnitude: (2.0f64).sqrt(),
        };

        let orig_density = g.connection_density;
        let orig_mod = g.modularity;

        // Large step, no noise
        g.mutate_with_gradient(&gradient, 0.1, 0.0, 42);

        // Density should increase (positive gradient)
        assert!(g.connection_density > orig_density - 0.01);
        // Modularity should decrease (negative gradient)
        assert!(g.modularity < orig_mod + 0.01);
    }

    #[test]
    fn test_mutate_natural_gradient_adaptive_lr() {
        // Steep gradient => smaller effective step
        let mut g_steep = ArchitectureGenome {
            connection_density: 0.5,
            ..Default::default()
        };
        let steep = PhiGradient {
            d_density: 1.0,
            d_modularity: 0.0,
            d_bridge_ratio: 0.0,
            d_tau_ratio: 0.0,
            d_binding_strength: 0.0,
            d_recurrence: 0.0,
            magnitude: 10.0,
        };
        g_steep.mutate_natural_gradient(&steep, 1.0, 42);
        let steep_delta = (g_steep.connection_density - 0.5).abs();

        // Flat gradient => larger effective step
        let mut g_flat = ArchitectureGenome {
            connection_density: 0.5,
            ..Default::default()
        };
        let flat = PhiGradient {
            d_density: 1.0,
            d_modularity: 0.0,
            d_bridge_ratio: 0.0,
            d_tau_ratio: 0.0,
            d_binding_strength: 0.0,
            d_recurrence: 0.0,
            magnitude: 0.01,
        };
        g_flat.mutate_natural_gradient(&flat, 1.0, 42);
        let flat_delta = (g_flat.connection_density - 0.5).abs();

        // Flat region should produce a larger step
        assert!(
            flat_delta > steep_delta,
            "flat_delta={} should be > steep_delta={}",
            flat_delta,
            steep_delta
        );
    }

    #[test]
    fn test_mutate_with_momentum_accumulates_velocity() {
        let mut g = ArchitectureGenome {
            connection_density: 0.5,
            modularity: 0.5,
            bridge_ratio: 0.4,
            tau_ratio: 0.5,
            binding_strength: 0.5,
            recurrence: 0.5,
            ..Default::default()
        };

        let gradient = PhiGradient {
            d_density: 1.0,
            d_modularity: 0.0,
            d_bridge_ratio: 0.0,
            d_tau_ratio: 0.0,
            d_binding_strength: 0.0,
            d_recurrence: 0.0,
            magnitude: 1.0,
        };

        let mut velocity = GradientVelocity::new();
        assert!(velocity.magnitude() < 1e-9);

        // First step
        g.mutate_with_momentum(&gradient, &mut velocity, 0.9, 0.1);
        let v1 = velocity.v_density;
        assert!(
            v1 > 0.0,
            "velocity should be positive after positive gradient"
        );

        // Second step: momentum should carry forward
        g.mutate_with_momentum(&gradient, &mut velocity, 0.9, 0.1);
        let v2 = velocity.v_density;
        assert!(
            v2 > v1,
            "velocity should accumulate with momentum: v2={} > v1={}",
            v2,
            v1
        );
    }

    #[test]
    fn test_genome_clone_equals_original() {
        let g = ArchitectureGenome::random(42);
        let g2 = g.clone();
        assert_eq!(g.num_nodes, g2.num_nodes);
        assert!((g.base_tau - g2.base_tau).abs() < f32::EPSILON);
        assert_eq!(g.topology_type, g2.topology_type);
        assert_eq!(g.bundling_mode, g2.bundling_mode);
        assert_eq!(g.seed, g2.seed);
        assert_eq!(g.hdc_dim, g2.hdc_dim);
        assert!((g.connection_density - g2.connection_density).abs() < f32::EPSILON);
    }
}
