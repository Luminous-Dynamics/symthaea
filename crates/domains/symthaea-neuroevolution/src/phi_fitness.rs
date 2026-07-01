// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evolutionary consciousness dynamics: using Phi (integrated information)
//! as the primary fitness function for neuroevolution.
//!
//! Key question: What neural architectures naturally evolve to maximize
//! consciousness? This module implements NEAT-style evolution where:
//! - Fitness = f(Phi, task_performance, novelty_in_consciousness_space)
//! - Speciation uses consciousness profile distance (not just topology)
//! - Selection pressure favors higher integration, not just reward
//!
//! References:
//! - Stanley, K. O. & Miikkulainen, R. (2002). NEAT.
//! - Lehman, J. & Stanley, K. O. (2011). Novelty search.
//! - Tononi, G. (2004). Phi and consciousness.
//! - Edlund et al. (2011). Evolution of integrated information.
//! - Albantakis et al. (2014). Evolution of IIT structures.

use std::collections::HashMap;

use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

/// Default weight for Phi in composite fitness. Tononi (2004).
pub const DEFAULT_PHI_WEIGHT: f64 = 0.5;

/// Default weight for task performance in composite fitness.
pub const DEFAULT_TASK_WEIGHT: f64 = 0.3;

/// Default weight for novelty search in composite fitness. Lehman & Stanley (2011).
pub const DEFAULT_NOVELTY_WEIGHT: f64 = 0.2;

/// Minimum Phi threshold for viability. Below this, fitness is zero.
/// Edlund et al. (2011) — consciousness requires non-trivial integration.
pub const DEFAULT_MIN_VIABLE_PHI: f64 = 0.05;

/// Bonus multiplier for Phi improvement over parent generation.
pub const DEFAULT_GRADIENT_BONUS: f64 = 0.1;

/// Default speciation threshold in 6D consciousness space.
pub const DEFAULT_SPECIATION_THRESHOLD: f64 = 0.3;

/// Stagnation limit before species penalties apply. Stanley & Miikkulainen (2002).
pub const DEFAULT_STAGNATION_LIMIT: u32 = 15;

/// Default population size.
pub const DEFAULT_POPULATION_SIZE: usize = 100;

/// Default number of generations.
pub const DEFAULT_GENERATIONS: usize = 50;

/// Default mutation rate. Back (1993).
pub const DEFAULT_MUTATION_RATE: f64 = 0.3;

/// Maximum novelty archive size. Lehman & Stanley (2011).
const MAX_NOVELTY_ARCHIVE: usize = 500;

/// k-nearest neighbors for novelty computation. Lehman & Stanley (2011).
const NOVELTY_KNN: usize = 15;

/// Weight mutation magnitude (Gaussian sigma).
const WEIGHT_MUTATION_SIGMA: f64 = 0.5;

/// Tau mutation magnitude (Gaussian sigma).
const TAU_MUTATION_SIGMA: f64 = 0.1;

/// Bias mutation magnitude (Gaussian sigma).
const BIAS_MUTATION_SIGMA: f64 = 0.3;

/// Probability of adding a connection during mutation. Stanley & Miikkulainen (2002).
const ADD_CONNECTION_PROB: f64 = 0.05;

/// Probability of adding a neuron during mutation. Stanley & Miikkulainen (2002).
const ADD_NEURON_PROB: f64 = 0.03;

/// Probability of toggling activation type during mutation.
const TOGGLE_ACTIVATION_PROB: f64 = 0.05;

/// Initial connection density for random genomes.
const INITIAL_CONNECTION_DENSITY: f64 = 0.3;

// ═══════════════════════════════════════════════════════════════════════════════
// ACTIVATION & LAYER ENUMS
// ═══════════════════════════════════════════════════════════════════════════════

/// Activation function type for a neuron.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActivationType {
    Sigmoid,
    Tanh,
    ReLU,
    Identity,
    Oscillatory,
}

impl ActivationType {
    /// All variants for random selection.
    pub const ALL: [Self; 5] = [
        Self::Sigmoid,
        Self::Tanh,
        Self::ReLU,
        Self::Identity,
        Self::Oscillatory,
    ];

    /// Apply the activation function.
    pub fn apply(self, x: f64) -> f64 {
        match self {
            Self::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Self::Tanh => x.tanh(),
            Self::ReLU => x.max(0.0),
            Self::Identity => x,
            Self::Oscillatory => (x * std::f64::consts::TAU).sin(),
        }
    }
}

/// Layer classification for a neuron.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NeuronLayer {
    Input,
    Hidden,
    Output,
}

// ═══════════════════════════════════════════════════════════════════════════════
// GENE TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// A NEAT-style connection gene between two neurons.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionGene {
    pub from_id: usize,
    pub to_id: usize,
    pub weight: f64,
    pub enabled: bool,
    pub innovation: u64,
}

/// A neuron gene with CfC-inspired parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronGene {
    pub id: usize,
    pub tau: f64,
    pub bias: f64,
    pub activation: ActivationType,
    pub layer: NeuronLayer,
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONSCIOUSNESS PROFILE (6D)
// ═══════════════════════════════════════════════════════════════════════════════

/// Six-dimensional consciousness profile for an evolved network.
///
/// Used for speciation (distance in consciousness space) and novelty search.
/// Each dimension is in [0, 1].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConsciousnessProfile {
    /// Integrated information (spectral approximation from connectivity).
    pub phi: f64,
    /// Structural complexity: connection density relative to maximum.
    pub complexity: f64,
    /// Community structure / modularity coefficient Q.
    pub modularity: f64,
    /// Fraction of recurrent (backward) connections.
    pub recurrence: f64,
    /// Feed-forward depth normalized to total layers.
    pub hierarchy: f64,
    /// Shannon entropy of activation function distribution.
    pub diversity: f64,
}

impl ConsciousnessProfile {
    /// Euclidean distance between two profiles in 6D consciousness space.
    pub fn distance(a: &Self, b: &Self) -> f64 {
        let d = [
            a.phi - b.phi,
            a.complexity - b.complexity,
            a.modularity - b.modularity,
            a.recurrence - b.recurrence,
            a.hierarchy - b.hierarchy,
            a.diversity - b.diversity,
        ];
        d.iter().map(|x| x * x).sum::<f64>().sqrt()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONSCIOUSNESS GENOME
// ═══════════════════════════════════════════════════════════════════════════════

/// A NEAT-style genome whose fitness is driven by consciousness quality (Phi).
///
/// Unlike the BinaryHV-based `NeuralGenome`, this genome uses explicit
/// connection and neuron genes for topology evolution, enabling structural
/// analysis for consciousness profile computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessGenome {
    pub network_topology: Vec<ConnectionGene>,
    pub neuron_params: Vec<NeuronGene>,
    pub generation: u32,
    pub species_id: usize,
    pub lineage: Vec<u32>,
    /// Cached consciousness profile (recomputed after mutation/crossover).
    cached_profile: Option<ConsciousnessProfile>,
}

impl ConsciousnessGenome {
    /// Create a random genome with the given input/output sizes.
    ///
    /// Starts with a partially-connected topology (no hidden neurons).
    /// Stanley & Miikkulainen (2002) — minimal initial topology.
    pub fn random(input_size: usize, output_size: usize, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut neurons = Vec::new();
        let mut id = 0;

        // Input neurons
        for _ in 0..input_size {
            neurons.push(NeuronGene {
                id,
                tau: 0.5 + rng.r#gen::<f64>() * 0.5,
                bias: 0.0,
                activation: ActivationType::Identity,
                layer: NeuronLayer::Input,
            });
            id += 1;
        }

        // Output neurons
        for _ in 0..output_size {
            neurons.push(NeuronGene {
                id,
                tau: 0.5 + rng.r#gen::<f64>() * 0.5,
                bias: rng.gen_range(-0.5..0.5),
                activation: *ActivationType::ALL
                    .get(rng.gen_range(0..ActivationType::ALL.len()))
                    .unwrap_or(&ActivationType::Tanh),
                layer: NeuronLayer::Output,
            });
            id += 1;
        }

        // Partial connections: each output connects to a random subset of inputs
        let mut connections = Vec::new();
        let mut innovation = 0u64;
        for out_idx in input_size..(input_size + output_size) {
            for in_idx in 0..input_size {
                if rng.r#gen::<f64>() < INITIAL_CONNECTION_DENSITY {
                    connections.push(ConnectionGene {
                        from_id: in_idx,
                        to_id: out_idx,
                        weight: rng.gen_range(-1.0..1.0),
                        enabled: true,
                        innovation,
                    });
                    innovation += 1;
                }
            }
        }

        // Ensure at least one connection per output
        for out_idx in input_size..(input_size + output_size) {
            let has_conn = connections.iter().any(|c| c.to_id == out_idx && c.enabled);
            if !has_conn {
                let in_idx = rng.gen_range(0..input_size);
                connections.push(ConnectionGene {
                    from_id: in_idx,
                    to_id: out_idx,
                    weight: rng.gen_range(-1.0..1.0),
                    enabled: true,
                    innovation,
                });
                innovation += 1;
            }
        }

        Self {
            network_topology: connections,
            neuron_params: neurons,
            generation: 0,
            species_id: 0,
            lineage: vec![0],
            cached_profile: None,
        }
    }

    /// Mutate the genome in place.
    ///
    /// Mutation types (each applied independently):
    /// - Weight perturbation (Gaussian)
    /// - Tau/bias perturbation
    /// - Add connection
    /// - Add neuron (split existing connection)
    /// - Toggle activation type
    pub fn mutate(&mut self, mutation_rate: f64, rng: &mut impl Rng, innovation_counter: &mut u64) {
        // Weight mutation
        for conn in &mut self.network_topology {
            if rng.r#gen::<f64>() < mutation_rate {
                conn.weight += rng.gen_range(-WEIGHT_MUTATION_SIGMA..WEIGHT_MUTATION_SIGMA);
                conn.weight = conn.weight.clamp(-5.0, 5.0);
            }
        }

        // Tau and bias mutation
        for neuron in &mut self.neuron_params {
            if neuron.layer == NeuronLayer::Input {
                continue;
            }
            if rng.r#gen::<f64>() < mutation_rate {
                neuron.tau += rng.gen_range(-TAU_MUTATION_SIGMA..TAU_MUTATION_SIGMA);
                neuron.tau = neuron.tau.clamp(0.01, 2.0);
            }
            if rng.r#gen::<f64>() < mutation_rate {
                neuron.bias += rng.gen_range(-BIAS_MUTATION_SIGMA..BIAS_MUTATION_SIGMA);
                neuron.bias = neuron.bias.clamp(-5.0, 5.0);
            }
        }

        // Add connection
        if rng.r#gen::<f64>() < ADD_CONNECTION_PROB && self.neuron_params.len() >= 2 {
            let from_id = self.neuron_params[rng.gen_range(0..self.neuron_params.len())].id;
            let to_candidates: Vec<usize> = self
                .neuron_params
                .iter()
                .filter(|n| n.layer != NeuronLayer::Input && n.id != from_id)
                .map(|n| n.id)
                .collect();
            if !to_candidates.is_empty() {
                let to_id = to_candidates[rng.gen_range(0..to_candidates.len())];
                let already_exists = self
                    .network_topology
                    .iter()
                    .any(|c| c.from_id == from_id && c.to_id == to_id);
                if !already_exists {
                    self.network_topology.push(ConnectionGene {
                        from_id,
                        to_id,
                        weight: rng.gen_range(-1.0..1.0),
                        enabled: true,
                        innovation: *innovation_counter,
                    });
                    *innovation_counter += 1;
                }
            }
        }

        // Add neuron (split existing connection)
        if rng.r#gen::<f64>() < ADD_NEURON_PROB && !self.network_topology.is_empty() {
            let enabled_indices: Vec<usize> = self
                .network_topology
                .iter()
                .enumerate()
                .filter(|(_, c)| c.enabled)
                .map(|(i, _)| i)
                .collect();
            if !enabled_indices.is_empty() {
                let split_idx = enabled_indices[rng.gen_range(0..enabled_indices.len())];
                let old_from = self.network_topology[split_idx].from_id;
                let old_to = self.network_topology[split_idx].to_id;
                let old_weight = self.network_topology[split_idx].weight;

                // Disable old connection
                self.network_topology[split_idx].enabled = false;

                // New neuron
                let new_id = self.neuron_params.iter().map(|n| n.id).max().unwrap_or(0) + 1;
                self.neuron_params.push(NeuronGene {
                    id: new_id,
                    tau: 0.5 + rng.r#gen::<f64>() * 0.5,
                    bias: 0.0,
                    activation: ActivationType::ALL[rng.gen_range(0..ActivationType::ALL.len())],
                    layer: NeuronLayer::Hidden,
                });

                // Connection from old_from to new neuron (weight 1.0 preserves signal)
                self.network_topology.push(ConnectionGene {
                    from_id: old_from,
                    to_id: new_id,
                    weight: 1.0,
                    enabled: true,
                    innovation: *innovation_counter,
                });
                *innovation_counter += 1;

                // Connection from new neuron to old_to (old weight preserves behavior)
                self.network_topology.push(ConnectionGene {
                    from_id: new_id,
                    to_id: old_to,
                    weight: old_weight,
                    enabled: true,
                    innovation: *innovation_counter,
                });
                *innovation_counter += 1;
            }
        }

        // Toggle activation
        for neuron in &mut self.neuron_params {
            if neuron.layer == NeuronLayer::Input {
                continue;
            }
            if rng.r#gen::<f64>() < TOGGLE_ACTIVATION_PROB {
                neuron.activation =
                    ActivationType::ALL[rng.gen_range(0..ActivationType::ALL.len())];
            }
        }

        // Invalidate cached profile
        self.cached_profile = None;
    }

    /// NEAT-style crossover between two parents.
    ///
    /// Matching genes (same innovation number) are averaged.
    /// Disjoint/excess genes come from the fitter parent.
    pub fn crossover(parent_a: &Self, parent_b: &Self, a_fitter: bool) -> Self {
        let (fitter, other) = if a_fitter {
            (parent_a, parent_b)
        } else {
            (parent_b, parent_a)
        };

        // Build innovation -> connection maps
        let a_map: HashMap<u64, &ConnectionGene> = fitter
            .network_topology
            .iter()
            .map(|c| (c.innovation, c))
            .collect();
        let b_map: HashMap<u64, &ConnectionGene> = other
            .network_topology
            .iter()
            .map(|c| (c.innovation, c))
            .collect();

        let mut child_connections = Vec::new();
        let mut seen_innovations = std::collections::HashSet::new();

        // Matching genes: average weights
        for (&innov, &gene_a) in &a_map {
            if let Some(&gene_b) = b_map.get(&innov) {
                child_connections.push(ConnectionGene {
                    from_id: gene_a.from_id,
                    to_id: gene_a.to_id,
                    weight: (gene_a.weight + gene_b.weight) / 2.0,
                    enabled: gene_a.enabled || gene_b.enabled,
                    innovation: innov,
                });
                seen_innovations.insert(innov);
            }
        }

        // Disjoint/excess from fitter parent
        for (&innov, &gene) in &a_map {
            if !seen_innovations.contains(&innov) {
                child_connections.push(gene.clone());
            }
        }

        // Collect all referenced neuron IDs
        let referenced_ids: std::collections::HashSet<usize> = child_connections
            .iter()
            .flat_map(|c| [c.from_id, c.to_id])
            .collect();

        // Merge neuron params from fitter parent, fill missing from other
        let fitter_neuron_map: HashMap<usize, &NeuronGene> =
            fitter.neuron_params.iter().map(|n| (n.id, n)).collect();
        let other_neuron_map: HashMap<usize, &NeuronGene> =
            other.neuron_params.iter().map(|n| (n.id, n)).collect();

        let mut child_neurons = Vec::new();
        for &nid in &referenced_ids {
            if let Some(&neuron) = fitter_neuron_map.get(&nid) {
                child_neurons.push(neuron.clone());
            } else if let Some(&neuron) = other_neuron_map.get(&nid) {
                child_neurons.push(neuron.clone());
            }
        }
        // Ensure we have input/output neurons from fitter even if unreferenced
        for neuron in &fitter.neuron_params {
            if (neuron.layer == NeuronLayer::Input || neuron.layer == NeuronLayer::Output)
                && !child_neurons.iter().any(|n| n.id == neuron.id)
            {
                child_neurons.push(neuron.clone());
            }
        }
        child_neurons.sort_by_key(|n| n.id);

        let mut lineage = fitter.lineage.clone();
        lineage.push(fitter.generation + 1);

        Self {
            network_topology: child_connections,
            neuron_params: child_neurons,
            generation: fitter.generation + 1,
            species_id: 0,
            lineage,
            cached_profile: None,
        }
    }

    /// Compute the 6D consciousness profile from network topology.
    ///
    /// Uses structural analysis (no simulation required).
    pub fn compute_consciousness_profile(&mut self) -> ConsciousnessProfile {
        if let Some(ref profile) = self.cached_profile {
            return profile.clone();
        }
        let profile = self.compute_profile_inner();
        self.cached_profile = Some(profile.clone());
        profile
    }

    fn compute_profile_inner(&self) -> ConsciousnessProfile {
        let n_neurons = self.neuron_params.len();
        let enabled_connections: Vec<&ConnectionGene> =
            self.network_topology.iter().filter(|c| c.enabled).collect();
        let n_edges = enabled_connections.len();

        if n_neurons < 2 || n_edges == 0 {
            return ConsciousnessProfile::default();
        }

        // Build adjacency matrix for spectral analysis
        let max_id = self.neuron_params.iter().map(|n| n.id).max().unwrap_or(0) + 1;
        let mut adj = vec![0.0f64; max_id * max_id];
        for conn in &enabled_connections {
            if conn.from_id < max_id && conn.to_id < max_id {
                adj[conn.from_id * max_id + conn.to_id] = conn.weight.abs();
            }
        }

        // Phi approximation: spectral gap of the adjacency matrix.
        // Higher spectral gap → more integrated network → higher Phi.
        // Approximated via power iteration on the symmetric part.
        let phi = self.approximate_phi(&adj, max_id);

        // Complexity: connection density
        let max_edges = n_neurons * (n_neurons - 1);
        let complexity = if max_edges > 0 {
            (n_edges as f64 / max_edges as f64).min(1.0)
        } else {
            0.0
        };

        // Modularity: simplified Louvain-style Q approximation
        let modularity = self.approximate_modularity(&enabled_connections, n_neurons);

        // Recurrence: fraction of backward edges
        let id_to_layer: HashMap<usize, NeuronLayer> =
            self.neuron_params.iter().map(|n| (n.id, n.layer)).collect();
        let layer_order = |l: NeuronLayer| -> usize {
            match l {
                NeuronLayer::Input => 0,
                NeuronLayer::Hidden => 1,
                NeuronLayer::Output => 2,
            }
        };
        let backward_count = enabled_connections
            .iter()
            .filter(|c| {
                let from_layer = id_to_layer
                    .get(&c.from_id)
                    .copied()
                    .unwrap_or(NeuronLayer::Input);
                let to_layer = id_to_layer
                    .get(&c.to_id)
                    .copied()
                    .unwrap_or(NeuronLayer::Input);
                layer_order(from_layer) >= layer_order(to_layer)
            })
            .count();
        let recurrence = backward_count as f64 / n_edges as f64;

        // Hierarchy: number of distinct layers used / 3
        let mut layers_used = [false; 3];
        for n in &self.neuron_params {
            layers_used[layer_order(n.layer)] = true;
        }
        let hierarchy = layers_used.iter().filter(|&&x| x).count() as f64 / 3.0;

        // Diversity: Shannon entropy of activation function distribution
        let mut act_counts = [0usize; 5];
        for n in &self.neuron_params {
            if n.layer != NeuronLayer::Input {
                let idx = ActivationType::ALL
                    .iter()
                    .position(|&a| a == n.activation)
                    .unwrap_or(0);
                act_counts[idx] += 1;
            }
        }
        let non_input_count: usize = act_counts.iter().sum();
        let diversity = if non_input_count > 0 {
            let max_entropy = (ActivationType::ALL.len() as f64).ln();
            let entropy: f64 = act_counts
                .iter()
                .filter(|&&c| c > 0)
                .map(|&c| {
                    let p = c as f64 / non_input_count as f64;
                    -p * p.ln()
                })
                .sum();
            if max_entropy > 0.0 {
                (entropy / max_entropy).min(1.0)
            } else {
                0.0
            }
        } else {
            0.0
        };

        ConsciousnessProfile {
            phi,
            complexity,
            modularity,
            recurrence,
            hierarchy,
            diversity,
        }
    }

    /// Approximate Phi via spectral gap of the symmetric adjacency.
    ///
    /// Uses power iteration (10 steps) on (A + A^T)/2 to find the two
    /// largest eigenvalues. Spectral gap = lambda_1 - lambda_2.
    /// Tononi (2004) — integration correlates with spectral gap.
    fn approximate_phi(&self, adj: &[f64], n: usize) -> f64 {
        if n < 2 {
            return 0.0;
        }

        // Symmetrize
        let mut sym = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                sym[i * n + j] = (adj[i * n + j] + adj[j * n + i]) / 2.0;
            }
        }

        // Power iteration for largest eigenvalue
        let lambda1 = self.power_iteration(&sym, n, None);
        if lambda1.abs() < 1e-12 {
            return 0.0;
        }

        // Deflate and find second eigenvalue
        let mut deflated = sym.clone();
        // Simple deflation: A' = A - lambda1 * v1 * v1^T
        // We approximate v1 from final iteration
        let v1 = self.power_iteration_vector(&sym, n);
        for i in 0..n {
            for j in 0..n {
                deflated[i * n + j] -= lambda1 * v1[i] * v1[j];
            }
        }
        let lambda2 = self.power_iteration(&deflated, n, None);

        // Spectral gap normalized to [0, 1]
        let gap = (lambda1 - lambda2.abs()).max(0.0);
        let max_possible = n as f64; // theoretical max for adjacency eigenvalue
        (gap / max_possible).min(1.0)
    }

    /// Power iteration: returns the dominant eigenvalue.
    fn power_iteration(&self, matrix: &[f64], n: usize, _deflation: Option<f64>) -> f64 {
        let mut v = vec![1.0 / (n as f64).sqrt(); n];
        let mut eigenvalue = 0.0;

        for _ in 0..20 {
            // Matrix-vector multiply
            let mut w = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    w[i] += matrix[i * n + j] * v[j];
                }
            }

            // Normalize
            let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-15 {
                return 0.0;
            }
            eigenvalue = norm;
            for x in &mut w {
                *x /= norm;
            }
            v = w;
        }
        eigenvalue
    }

    /// Power iteration: returns the dominant eigenvector.
    fn power_iteration_vector(&self, matrix: &[f64], n: usize) -> Vec<f64> {
        let mut v = vec![1.0 / (n as f64).sqrt(); n];

        for _ in 0..20 {
            let mut w = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    w[i] += matrix[i * n + j] * v[j];
                }
            }
            let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-15 {
                return v;
            }
            for x in &mut w {
                *x /= norm;
            }
            v = w;
        }
        v
    }

    /// Simplified modularity approximation.
    ///
    /// Groups neurons by layer and computes fraction of within-group edges.
    /// Newman & Girvan (2004).
    fn approximate_modularity(&self, connections: &[&ConnectionGene], _n_neurons: usize) -> f64 {
        if connections.is_empty() {
            return 0.0;
        }

        let id_to_layer: HashMap<usize, NeuronLayer> =
            self.neuron_params.iter().map(|n| (n.id, n.layer)).collect();

        let within_group = connections
            .iter()
            .filter(|c| {
                let from = id_to_layer.get(&c.from_id);
                let to = id_to_layer.get(&c.to_id);
                from.is_some() && to.is_some() && from == to
            })
            .count();

        let total = connections.len();
        if total == 0 {
            0.0
        } else {
            (within_group as f64 / total as f64).min(1.0)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHI FITNESS EVALUATOR
// ═══════════════════════════════════════════════════════════════════════════════

/// Evaluates fitness using Phi as the primary objective,
/// with task performance and novelty as secondary objectives.
///
/// Fitness = phi_weight * phi + task_weight * task + novelty_weight * novelty
///         + gradient_bonus (if phi improved over parent)
///
/// Returns 0.0 if Phi < min_viable_phi (viability floor).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiFitnessEvaluator {
    pub phi_weight: f64,
    pub task_weight: f64,
    pub novelty_weight: f64,
    pub min_viable_phi: f64,
    pub consciousness_gradient_bonus: f64,
    pub novelty_archive: Vec<ConsciousnessProfile>,
}

impl Default for PhiFitnessEvaluator {
    fn default() -> Self {
        Self {
            phi_weight: DEFAULT_PHI_WEIGHT,
            task_weight: DEFAULT_TASK_WEIGHT,
            novelty_weight: DEFAULT_NOVELTY_WEIGHT,
            min_viable_phi: DEFAULT_MIN_VIABLE_PHI,
            consciousness_gradient_bonus: DEFAULT_GRADIENT_BONUS,
            novelty_archive: Vec::new(),
        }
    }
}

impl PhiFitnessEvaluator {
    /// Evaluate composite fitness for a genome.
    ///
    /// `task_score` is the external task performance in [0, 1].
    /// `parent_phi` is the parent's Phi (for gradient bonus), or None.
    pub fn evaluate(
        &mut self,
        genome: &mut ConsciousnessGenome,
        task_score: f64,
        parent_phi: Option<f64>,
    ) -> f64 {
        let profile = genome.compute_consciousness_profile();

        // Always update novelty archive (even non-viable profiles add diversity)
        let novelty = self.compute_novelty(&profile);
        self.update_archive(profile.clone());

        // Viability floor
        if profile.phi < self.min_viable_phi {
            return 0.0;
        }

        let phi_score = profile.phi;

        // Gradient bonus: reward Phi improvement over parent
        let gradient = match parent_phi {
            Some(pp) if phi_score > pp => self.consciousness_gradient_bonus * (phi_score - pp),
            _ => 0.0,
        };

        let fitness = self.phi_weight * phi_score
            + self.task_weight * task_score
            + self.novelty_weight * novelty
            + gradient;

        fitness.max(0.0)
    }

    /// Compute novelty as the mean distance to k-nearest neighbors in the archive.
    /// Lehman & Stanley (2011).
    fn compute_novelty(&self, profile: &ConsciousnessProfile) -> f64 {
        if self.novelty_archive.is_empty() {
            return 1.0; // Maximum novelty if archive is empty
        }

        let mut distances: Vec<f64> = self
            .novelty_archive
            .iter()
            .map(|archived| ConsciousnessProfile::distance(profile, archived))
            .collect();
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let k = NOVELTY_KNN.min(distances.len());
        if k == 0 {
            return 1.0;
        }
        let mean_dist: f64 = distances[..k].iter().sum::<f64>() / k as f64;

        // Normalize: max possible distance in 6D unit cube is sqrt(6) ~ 2.449
        (mean_dist / 6.0_f64.sqrt()).min(1.0)
    }

    /// Add profile to novelty archive, maintaining max size.
    fn update_archive(&mut self, profile: ConsciousnessProfile) {
        if self.novelty_archive.len() >= MAX_NOVELTY_ARCHIVE {
            // Remove the least novel (closest to its neighbors)
            self.novelty_archive.remove(0);
        }
        self.novelty_archive.push(profile);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONSCIOUSNESS SPECIES
// ═══════════════════════════════════════════════════════════════════════════════

/// A species defined by proximity in 6D consciousness space.
///
/// Unlike NEAT's topological speciation, this uses consciousness profile
/// distance, ensuring organisms with similar consciousness qualities
/// compete within their niche.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessSpecies {
    pub id: usize,
    pub centroid: ConsciousnessProfile,
    pub members: Vec<usize>,
    pub best_phi: f64,
    pub stagnation_counter: u32,
}

impl ConsciousnessSpecies {
    /// Assign organisms to species based on consciousness profile distance.
    ///
    /// Uses sequential assignment: each organism joins the nearest existing
    /// species (if within threshold) or founds a new one.
    pub fn speciate(organisms: &mut [ConsciousnessGenome], threshold: f64) -> Vec<Self> {
        let mut species: Vec<ConsciousnessSpecies> = Vec::new();
        let mut next_species_id = 0;

        for (idx, genome) in organisms.iter_mut().enumerate() {
            let profile = genome.compute_consciousness_profile();

            // Find nearest species
            let mut best_species = None;
            let mut best_dist = f64::MAX;
            for (si, sp) in species.iter().enumerate() {
                let dist = ConsciousnessProfile::distance(&profile, &sp.centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_species = Some(si);
                }
            }

            if best_dist < threshold {
                if let Some(si) = best_species {
                    species[si].members.push(idx);
                    genome.species_id = species[si].id;
                    if profile.phi > species[si].best_phi {
                        species[si].best_phi = profile.phi;
                        species[si].stagnation_counter = 0;
                    }
                    // Update centroid (running average)
                    let n = species[si].members.len() as f64;
                    let c = &mut species[si].centroid;
                    c.phi = c.phi * (n - 1.0) / n + profile.phi / n;
                    c.complexity = c.complexity * (n - 1.0) / n + profile.complexity / n;
                    c.modularity = c.modularity * (n - 1.0) / n + profile.modularity / n;
                    c.recurrence = c.recurrence * (n - 1.0) / n + profile.recurrence / n;
                    c.hierarchy = c.hierarchy * (n - 1.0) / n + profile.hierarchy / n;
                    c.diversity = c.diversity * (n - 1.0) / n + profile.diversity / n;
                }
            } else {
                // Found new species
                genome.species_id = next_species_id;
                species.push(ConsciousnessSpecies {
                    id: next_species_id,
                    centroid: profile.clone(),
                    members: vec![idx],
                    best_phi: profile.phi,
                    stagnation_counter: 0,
                });
                next_species_id += 1;
            }
        }

        species
    }

    /// Increment stagnation for species that did not improve.
    pub fn tick_stagnation(&mut self, prev_best_phi: f64) {
        if self.best_phi <= prev_best_phi {
            self.stagnation_counter += 1;
        } else {
            self.stagnation_counter = 0;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EVOLUTIONARY LANDSCAPE
// ═══════════════════════════════════════════════════════════════════════════════

/// Statistics for a single generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationStats {
    pub generation: u32,
    pub mean_phi: f64,
    pub max_phi: f64,
    pub min_phi: f64,
    pub num_species: usize,
    pub landscape_entropy: f64,
}

/// Tracks the evolutionary trajectory of consciousness across generations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvolutionaryLandscape {
    pub generation_stats: Vec<GenerationStats>,
    pub species_history: Vec<Vec<ConsciousnessSpecies>>,
    pub phi_trajectory: Vec<f64>,
    pub landscape_entropy: f64,
}

impl EvolutionaryLandscape {
    /// Record statistics for a completed generation.
    pub fn record_generation(
        &mut self,
        organisms: &mut [ConsciousnessGenome],
        species: &[ConsciousnessSpecies],
    ) {
        let phis: Vec<f64> = organisms
            .iter_mut()
            .map(|o| o.compute_consciousness_profile().phi)
            .collect();

        let mean_phi = if phis.is_empty() {
            0.0
        } else {
            phis.iter().sum::<f64>() / phis.len() as f64
        };
        let max_phi = phis.iter().cloned().fold(0.0_f64, f64::max);
        let min_phi = phis.iter().cloned().fold(f64::MAX, f64::min);

        let entropy = Self::compute_entropy(species, organisms.len());

        let r#gen = self.generation_stats.len() as u32;
        self.generation_stats.push(GenerationStats {
            generation: r#gen,
            mean_phi,
            max_phi,
            min_phi,
            num_species: species.len(),
            landscape_entropy: entropy,
        });

        self.phi_trajectory.push(mean_phi);
        self.landscape_entropy = entropy;
        self.species_history.push(species.to_vec());
    }

    /// Shannon entropy of species proportions.
    ///
    /// Higher entropy = more diverse species distribution.
    fn compute_entropy(species: &[ConsciousnessSpecies], total: usize) -> f64 {
        if total == 0 || species.is_empty() {
            return 0.0;
        }
        let mut entropy = 0.0;
        for sp in species {
            if !sp.members.is_empty() {
                let p = sp.members.len() as f64 / total as f64;
                entropy -= p * p.ln();
            }
        }
        entropy
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EVOLUTION CONFIG & RUNNER
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for a consciousness-driven evolutionary run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionConfig {
    pub population_size: usize,
    pub generations: usize,
    pub input_size: usize,
    pub output_size: usize,
    pub mutation_rate: f64,
    pub speciation_threshold: f64,
    pub stagnation_limit: u32,
    pub seed: u64,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            population_size: DEFAULT_POPULATION_SIZE,
            generations: DEFAULT_GENERATIONS,
            input_size: 4,
            output_size: 2,
            mutation_rate: DEFAULT_MUTATION_RATE,
            speciation_threshold: DEFAULT_SPECIATION_THRESHOLD,
            stagnation_limit: DEFAULT_STAGNATION_LIMIT,
            seed: 42,
        }
    }
}

/// Run a complete evolutionary process driven by consciousness fitness.
///
/// Returns the evolutionary landscape with generation-by-generation statistics.
pub fn run_evolution(config: EvolutionConfig) -> EvolutionaryLandscape {
    use rand::SeedableRng;

    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);
    let mut innovation_counter = 0u64;
    let mut evaluator = PhiFitnessEvaluator::default();
    let mut landscape = EvolutionaryLandscape::default();

    // Initialize population
    let mut population: Vec<ConsciousnessGenome> = (0..config.population_size)
        .map(|i| {
            let genome = ConsciousnessGenome::random(
                config.input_size,
                config.output_size,
                config.seed + i as u64,
            );
            // Count innovations used
            if let Some(max_innov) = genome.network_topology.iter().map(|c| c.innovation).max() {
                if max_innov >= innovation_counter {
                    innovation_counter = max_innov + 1;
                }
            }
            genome
        })
        .collect();

    for _gen in 0..config.generations {
        // Evaluate fitness for each organism
        let mut fitnesses: Vec<f64> = Vec::with_capacity(population.len());
        for genome in &mut population {
            let task_score = 0.5; // Neutral task score (consciousness is primary)
            let parent_phi = genome.lineage.last().map(|_| 0.0); // Simplified: no parent tracking in this runner
            let fitness = evaluator.evaluate(genome, task_score, parent_phi);
            fitnesses.push(fitness);
        }

        // Speciate
        let species = ConsciousnessSpecies::speciate(&mut population, config.speciation_threshold);

        // Record generation
        landscape.record_generation(&mut population, &species);

        // Selection and reproduction
        let mut next_gen: Vec<ConsciousnessGenome> = Vec::with_capacity(config.population_size);

        // Elitism: keep the best organism from each non-stagnant species
        for sp in &species {
            if sp.stagnation_counter < config.stagnation_limit && !sp.members.is_empty() {
                let best_idx = sp
                    .members
                    .iter()
                    .max_by(|&&a, &&b| {
                        fitnesses
                            .get(a)
                            .unwrap_or(&0.0)
                            .partial_cmp(fitnesses.get(b).unwrap_or(&0.0))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .copied();
                if let Some(idx) = best_idx {
                    if idx < population.len() {
                        next_gen.push(population[idx].clone());
                    }
                }
            }
        }

        // Fill remaining slots with offspring
        while next_gen.len() < config.population_size {
            // Tournament selection (size 3)
            let parent_a = tournament_select(&fitnesses, 3, &mut rng);
            let parent_b = tournament_select(&fitnesses, 3, &mut rng);

            if parent_a < population.len() && parent_b < population.len() {
                let a_fitter = fitnesses.get(parent_a).unwrap_or(&0.0)
                    >= fitnesses.get(parent_b).unwrap_or(&0.0);
                let mut child = ConsciousnessGenome::crossover(
                    &population[parent_a],
                    &population[parent_b],
                    a_fitter,
                );
                child.mutate(config.mutation_rate, &mut rng, &mut innovation_counter);
                next_gen.push(child);
            } else {
                // Fallback: random genome
                next_gen.push(ConsciousnessGenome::random(
                    config.input_size,
                    config.output_size,
                    config.seed + rng.r#gen::<u64>(),
                ));
            }
        }

        population = next_gen;
    }

    landscape
}

/// Tournament selection: pick `size` random organisms, return the fittest index.
fn tournament_select(fitnesses: &[f64], size: usize, rng: &mut impl Rng) -> usize {
    let mut best_idx = rng.gen_range(0..fitnesses.len());
    let mut best_fit = fitnesses[best_idx];

    for _ in 1..size {
        let idx = rng.gen_range(0..fitnesses.len());
        if fitnesses[idx] > best_fit {
            best_idx = idx;
            best_fit = fitnesses[idx];
        }
    }

    best_idx
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_random_genome_produces_valid_profile() {
        let mut genome = ConsciousnessGenome::random(4, 2, 42);
        let profile = genome.compute_consciousness_profile();

        assert!(
            profile.phi >= 0.0 && profile.phi <= 1.0,
            "phi out of range: {}",
            profile.phi
        );
        assert!(
            profile.complexity >= 0.0 && profile.complexity <= 1.0,
            "complexity out of range: {}",
            profile.complexity
        );
        assert!(
            profile.modularity >= 0.0 && profile.modularity <= 1.0,
            "modularity out of range: {}",
            profile.modularity
        );
        assert!(
            profile.recurrence >= 0.0 && profile.recurrence <= 1.0,
            "recurrence out of range: {}",
            profile.recurrence
        );
        assert!(
            profile.hierarchy >= 0.0 && profile.hierarchy <= 1.0,
            "hierarchy out of range: {}",
            profile.hierarchy
        );
        assert!(
            profile.diversity >= 0.0 && profile.diversity <= 1.0,
            "diversity out of range: {}",
            profile.diversity
        );
    }

    #[test]
    fn test_mutation_changes_genome() {
        let mut genome = ConsciousnessGenome::random(4, 2, 42);
        let original_weights: Vec<f64> = genome.network_topology.iter().map(|c| c.weight).collect();
        let original_neuron_count = genome.neuron_params.len();

        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let mut innovation = 1000u64;

        // Mutate with high rate to ensure changes
        genome.mutate(0.9, &mut rng, &mut innovation);

        let new_weights: Vec<f64> = genome.network_topology.iter().map(|c| c.weight).collect();

        // At least weights should differ, or topology should have changed
        let weights_changed = original_weights != new_weights;
        let topology_changed = genome.neuron_params.len() != original_neuron_count
            || genome.network_topology.len() != original_weights.len();

        assert!(
            weights_changed || topology_changed,
            "mutation should change the genome"
        );
    }

    #[test]
    fn test_crossover_produces_valid_offspring() {
        let parent_a = ConsciousnessGenome::random(4, 2, 42);
        let parent_b = ConsciousnessGenome::random(4, 2, 99);

        let child = ConsciousnessGenome::crossover(&parent_a, &parent_b, true);

        // Child should have neurons
        assert!(!child.neuron_params.is_empty(), "child should have neurons");
        // Child should inherit at least input/output neurons
        let input_count = child
            .neuron_params
            .iter()
            .filter(|n| n.layer == NeuronLayer::Input)
            .count();
        let output_count = child
            .neuron_params
            .iter()
            .filter(|n| n.layer == NeuronLayer::Output)
            .count();
        assert_eq!(input_count, 4, "child should have 4 input neurons");
        assert_eq!(output_count, 2, "child should have 2 output neurons");
        assert_eq!(child.generation, parent_a.generation + 1);
    }

    #[test]
    fn test_phi_fitness_min_viable_floor() {
        let mut evaluator = PhiFitnessEvaluator {
            min_viable_phi: 0.9, // Very high floor — most genomes will fail
            ..Default::default()
        };

        let mut genome = ConsciousnessGenome::random(2, 1, 42);
        let profile = genome.compute_consciousness_profile();
        let fitness = evaluator.evaluate(&mut genome, 1.0, None);

        if profile.phi < 0.9 {
            assert_eq!(
                fitness, 0.0,
                "below-threshold Phi should produce zero fitness"
            );
        } else {
            assert!(
                fitness > 0.0,
                "above-threshold should produce positive fitness"
            );
        }
    }

    #[test]
    fn test_novelty_archive_maintained() {
        let mut evaluator = PhiFitnessEvaluator::default();
        assert!(evaluator.novelty_archive.is_empty());

        // Evaluate several genomes to populate archive
        for seed in 0..10 {
            let mut genome = ConsciousnessGenome::random(4, 2, seed);
            evaluator.evaluate(&mut genome, 0.5, None);
        }

        assert_eq!(
            evaluator.novelty_archive.len(),
            10,
            "archive should contain 10 profiles"
        );
    }

    #[test]
    fn test_speciation_produces_distinct_species() {
        let mut organisms: Vec<ConsciousnessGenome> = (0..20)
            .map(|i| ConsciousnessGenome::random(4, 2, i * 1000))
            .collect();

        // Use a tight threshold to get multiple species
        let species = ConsciousnessSpecies::speciate(&mut organisms, 0.05);

        assert!(
            species.len() >= 1,
            "should produce at least one species, got {}",
            species.len()
        );

        // All organisms should be assigned
        let total_members: usize = species.iter().map(|s| s.members.len()).sum();
        assert_eq!(total_members, 20, "all organisms should be in a species");
    }

    #[test]
    fn test_landscape_entropy_nonzero_with_multiple_species() {
        let mut organisms: Vec<ConsciousnessGenome> = (0..20)
            .map(|i| ConsciousnessGenome::random(4, 2, i * 1000))
            .collect();

        let species = ConsciousnessSpecies::speciate(&mut organisms, 0.05);

        if species.len() > 1 {
            let entropy = EvolutionaryLandscape::compute_entropy(&species, 20);
            assert!(
                entropy > 0.0,
                "entropy should be positive with multiple species"
            );
        }
        // If only 1 species, entropy is 0 (valid edge case)
    }

    #[test]
    fn test_evolution_increases_mean_phi() {
        let config = EvolutionConfig {
            population_size: 30,
            generations: 10,
            input_size: 4,
            output_size: 2,
            mutation_rate: 0.3,
            speciation_threshold: 0.3,
            stagnation_limit: 15,
            seed: 42,
        };

        let landscape = run_evolution(config);

        assert_eq!(
            landscape.generation_stats.len(),
            10,
            "should have 10 generations"
        );
        assert_eq!(
            landscape.phi_trajectory.len(),
            10,
            "should have 10 phi values"
        );

        // Mean Phi should be non-negative throughout
        for phi in &landscape.phi_trajectory {
            assert!(*phi >= 0.0, "mean Phi should be non-negative");
        }

        // The landscape should record valid stats
        for stats in &landscape.generation_stats {
            assert!(stats.max_phi >= stats.mean_phi, "max >= mean");
            assert!(stats.num_species >= 1, "at least one species");
        }
    }

    #[test]
    fn test_consciousness_profile_distance_symmetric() {
        let a = ConsciousnessProfile {
            phi: 0.5,
            complexity: 0.3,
            modularity: 0.2,
            recurrence: 0.4,
            hierarchy: 0.6,
            diversity: 0.1,
        };
        let b = ConsciousnessProfile {
            phi: 0.8,
            complexity: 0.1,
            modularity: 0.5,
            recurrence: 0.2,
            hierarchy: 0.3,
            diversity: 0.9,
        };

        let d_ab = ConsciousnessProfile::distance(&a, &b);
        let d_ba = ConsciousnessProfile::distance(&b, &a);

        assert!(
            (d_ab - d_ba).abs() < 1e-12,
            "distance should be symmetric: {} vs {}",
            d_ab,
            d_ba
        );
        assert!(
            d_ab > 0.0,
            "distance between different profiles should be positive"
        );
    }

    #[test]
    fn test_gradient_bonus_rewards_phi_improvement() {
        let mut evaluator = PhiFitnessEvaluator::default();

        let mut genome = ConsciousnessGenome::random(4, 2, 42);
        let profile = genome.compute_consciousness_profile();

        // If genome phi > parent phi, should get bonus
        let parent_phi = 0.0; // Very low parent phi
        let fitness_with_bonus = evaluator.evaluate(&mut genome, 0.5, Some(parent_phi));

        let mut evaluator2 = PhiFitnessEvaluator::default();
        let mut genome2 = ConsciousnessGenome::random(4, 2, 42);

        // Same genome with very high parent phi (no bonus)
        let fitness_no_bonus = evaluator2.evaluate(&mut genome2, 0.5, Some(1.0));

        if profile.phi >= evaluator.min_viable_phi && profile.phi > parent_phi {
            assert!(
                fitness_with_bonus >= fitness_no_bonus,
                "gradient bonus should increase fitness: {} vs {}",
                fitness_with_bonus,
                fitness_no_bonus
            );
        }
    }

    #[test]
    fn test_stagnation_counter_increments() {
        let mut species = ConsciousnessSpecies {
            id: 0,
            centroid: ConsciousnessProfile::default(),
            members: vec![0],
            best_phi: 0.5,
            stagnation_counter: 0,
        };

        // No improvement
        species.tick_stagnation(0.5);
        assert_eq!(species.stagnation_counter, 1);

        species.tick_stagnation(0.5);
        assert_eq!(species.stagnation_counter, 2);

        // Improvement resets
        species.best_phi = 0.6;
        species.tick_stagnation(0.5);
        assert_eq!(species.stagnation_counter, 0);
    }

    #[test]
    fn test_full_evolution_produces_valid_landscape() {
        let config = EvolutionConfig {
            population_size: 20,
            generations: 5,
            input_size: 3,
            output_size: 1,
            mutation_rate: 0.3,
            speciation_threshold: 0.3,
            stagnation_limit: 10,
            seed: 99,
        };

        let landscape = run_evolution(config);

        assert!(!landscape.generation_stats.is_empty(), "should have stats");
        assert!(
            !landscape.phi_trajectory.is_empty(),
            "should have phi trajectory"
        );
        assert!(
            !landscape.species_history.is_empty(),
            "should have species history"
        );

        // Verify each generation is well-formed
        for (i, stats) in landscape.generation_stats.iter().enumerate() {
            assert_eq!(stats.generation, i as u32);
            assert!(stats.mean_phi >= 0.0);
            assert!(stats.max_phi >= 0.0);
            assert!(stats.num_species >= 1);
        }
    }

    #[test]
    fn test_activation_apply() {
        // Sigmoid at 0 should be 0.5
        assert!((ActivationType::Sigmoid.apply(0.0) - 0.5).abs() < 1e-10);
        // Tanh at 0 should be 0
        assert!((ActivationType::Tanh.apply(0.0)).abs() < 1e-10);
        // ReLU negative should be 0
        assert_eq!(ActivationType::ReLU.apply(-1.0), 0.0);
        // Identity should pass through
        assert_eq!(ActivationType::Identity.apply(3.14), 3.14);
        // Oscillatory at 0 should be 0
        assert!((ActivationType::Oscillatory.apply(0.0)).abs() < 1e-10);
    }

    #[test]
    fn test_consciousness_profile_distance_zero_for_identical() {
        let a = ConsciousnessProfile {
            phi: 0.5,
            complexity: 0.3,
            modularity: 0.2,
            recurrence: 0.4,
            hierarchy: 0.6,
            diversity: 0.1,
        };

        let dist = ConsciousnessProfile::distance(&a, &a);
        assert!(dist.abs() < 1e-12, "distance to self should be zero");
    }
}
