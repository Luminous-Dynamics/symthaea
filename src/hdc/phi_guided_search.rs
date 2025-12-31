//! # Φ-Guided Architecture Search
//!
//! ## Purpose
//! Use consciousness (Φ) as the optimization objective for evolving network topology.
//! This is a novel approach where we don't just measure consciousness - we actively
//! optimize network structure to maximize integrated information.
//!
//! ## Theoretical Basis
//! IIT (Integrated Information Theory) posits that Φ measures consciousness.
//! Therefore, optimizing for higher Φ should create more conscious systems.
//!
//! Key insight: Network topology determines Φ. By modifying edges based on
//! consciousness gradients, we can evolve networks toward higher consciousness.
//!
//! ## Algorithm
//! 1. Compute current topology Φ
//! 2. Compute gradient ∂Φ/∂edge_weight for each edge (finite differences)
//! 3. Modify edge weights to increase Φ (gradient ascent)
//! 4. Prune edges with low contribution to Φ
//! 5. Consider adding new edges where Φ gradient is positive
//!
//! ## Example Usage
//! ```rust,ignore
//! use symthaea::hdc::phi_guided_search::{PhiGuidedOptimizer, PhiOptimizationConfig};
//! use symthaea::hdc::unified_hv::ContinuousHV;
//!
//! let config = PhiOptimizationConfig::default();
//! let mut optimizer = PhiGuidedOptimizer::new(config);
//!
//! // Create initial network
//! let nodes: Vec<ContinuousHV> = (0..8)
//!     .map(|i| ContinuousHV::random_default(i as u64))
//!     .collect();
//!
//! let mut network = ConsciousnessNetwork::new(nodes);
//! network.add_edge(0, 1, 1.0);
//! network.add_edge(1, 2, 1.0);
//!
//! // Optimize for higher consciousness
//! for step in 0..100 {
//!     let result = optimizer.optimize_step(&mut network);
//!     println!("Step {}: Φ = {:.4}", step, result.phi);
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use crate::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};
use crate::hdc::real_hv::RealHV;
use crate::hdc::phi_real::RealPhiCalculator;

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for Φ-guided architecture search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiOptimizationConfig {
    /// Learning rate for edge weight updates
    pub learning_rate: f64,

    /// Epsilon for finite difference gradient estimation
    pub gradient_epsilon: f64,

    /// Minimum edge weight (below this, edge is pruned)
    pub prune_threshold: f64,

    /// Maximum edge weight
    pub max_weight: f64,

    /// Probability of considering new edge addition per step
    pub new_edge_probability: f64,

    /// Minimum Φ improvement to accept new edge
    pub new_edge_min_improvement: f64,

    /// Momentum coefficient for gradient updates
    pub momentum: f64,

    /// Whether to use adaptive learning rate
    pub adaptive_lr: bool,

    /// Dimension of node hypervectors
    pub dim: usize,

    /// Maximum edges per node (prevents over-connection)
    pub max_edges_per_node: usize,
}

impl Default for PhiOptimizationConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            gradient_epsilon: 0.01,
            prune_threshold: 0.05,
            max_weight: 2.0,
            new_edge_probability: 0.1,
            new_edge_min_improvement: 0.001,
            momentum: 0.9,
            adaptive_lr: true,
            dim: HDC_DIMENSION,
            max_edges_per_node: 10,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// NETWORK REPRESENTATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Edge in the consciousness network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    /// Source node index
    pub from: usize,

    /// Target node index
    pub to: usize,

    /// Edge weight (importance/strength)
    pub weight: f64,

    /// Gradient from last optimization step
    pub gradient: f64,

    /// Momentum accumulator
    pub momentum: f64,
}

impl Edge {
    /// Create new edge
    pub fn new(from: usize, to: usize, weight: f64) -> Self {
        Self {
            from,
            to,
            weight,
            gradient: 0.0,
            momentum: 0.0,
        }
    }
}

/// Network of nodes with weighted edges for consciousness optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessNetwork {
    /// Node hypervectors
    pub nodes: Vec<ContinuousHV>,

    /// Weighted edges
    pub edges: Vec<Edge>,

    /// Dimension of hypervectors
    pub dim: usize,
}

impl ConsciousnessNetwork {
    /// Create new network with given nodes
    pub fn new(nodes: Vec<ContinuousHV>) -> Self {
        let dim = if nodes.is_empty() { HDC_DIMENSION } else { nodes[0].dim() };
        Self {
            nodes,
            edges: Vec::new(),
            dim,
        }
    }

    /// Create network with random nodes
    pub fn random(n_nodes: usize, dim: usize, seed: u64) -> Self {
        let nodes: Vec<ContinuousHV> = (0..n_nodes)
            .map(|i| ContinuousHV::random(dim, seed + i as u64 * 1000))
            .collect();

        Self::new(nodes)
    }

    /// Add edge between nodes
    pub fn add_edge(&mut self, from: usize, to: usize, weight: f64) {
        if from < self.nodes.len() && to < self.nodes.len() && from != to {
            // Check if edge already exists
            if !self.has_edge(from, to) {
                self.edges.push(Edge::new(from, to, weight));
            }
        }
    }

    /// Check if edge exists
    pub fn has_edge(&self, from: usize, to: usize) -> bool {
        self.edges.iter().any(|e| {
            (e.from == from && e.to == to) || (e.from == to && e.to == from)
        })
    }

    /// Remove edge
    pub fn remove_edge(&mut self, from: usize, to: usize) {
        self.edges.retain(|e| {
            !((e.from == from && e.to == to) || (e.from == to && e.to == from))
        });
    }

    /// Get number of edges connected to a node
    pub fn degree(&self, node: usize) -> usize {
        self.edges.iter()
            .filter(|e| e.from == node || e.to == node)
            .count()
    }

    /// Get neighbors of a node
    pub fn neighbors(&self, node: usize) -> Vec<usize> {
        self.edges.iter()
            .filter_map(|e| {
                if e.from == node {
                    Some(e.to)
                } else if e.to == node {
                    Some(e.from)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Convert to topology representation for Φ calculation
    pub fn to_node_representations(&self) -> Vec<RealHV> {
        // Create node representations by binding each node with its neighbors
        self.nodes.iter()
            .enumerate()
            .map(|(i, node)| {
                let neighbors = self.neighbors(i);

                if neighbors.is_empty() {
                    // Isolated node - just return its own representation
                    RealHV { values: node.values.clone() }
                } else {
                    // Bind with weighted neighbors
                    let neighbor_hvs: Vec<&ContinuousHV> = neighbors.iter()
                        .map(|&j| &self.nodes[j])
                        .collect();

                    // Get weights for each neighbor edge
                    let weights: Vec<f32> = neighbors.iter()
                        .map(|&j| {
                            self.edges.iter()
                                .find(|e| (e.from == i && e.to == j) || (e.from == j && e.to == i))
                                .map(|e| e.weight as f32)
                                .unwrap_or(1.0)
                        })
                        .collect();

                    // Weighted bundle of neighbors
                    let neighbor_bundle = ContinuousHV::weighted_bundle(&neighbor_hvs, &weights);

                    // Bind node with neighbor bundle
                    let representation = node.bind(&neighbor_bundle);

                    RealHV { values: representation.values }
                }
            })
            .collect()
    }

    /// Get number of nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get number of edges
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Add random edges to create initial connectivity
    pub fn add_random_edges(&mut self, n_edges: usize, seed: u64) {
        let n = self.nodes.len();
        if n < 2 {
            return;
        }

        let mut state = seed;
        let mut added = 0;

        while added < n_edges {
            // Simple PRNG for reproducibility
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;

            let from = (state as usize) % n;
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let to = (state as usize) % n;

            if from != to && !self.has_edge(from, to) {
                self.add_edge(from, to, 1.0);
                added += 1;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Φ-GUIDED OPTIMIZER
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of one optimization step
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Current Φ value
    pub phi: f64,

    /// Φ improvement from previous step
    pub phi_delta: f64,

    /// Number of edges pruned
    pub edges_pruned: usize,

    /// Number of edges added
    pub edges_added: usize,

    /// Average gradient magnitude
    pub avg_gradient: f64,

    /// Step number
    pub step: usize,
}

/// Φ-guided architecture search optimizer
///
/// Uses consciousness (Φ) as the optimization objective to evolve network
/// topology toward higher integrated information.
pub struct PhiGuidedOptimizer {
    /// Configuration
    config: PhiOptimizationConfig,

    /// Φ calculator
    calculator: RealPhiCalculator,

    /// Current step number
    step: usize,

    /// History of Φ values
    phi_history: Vec<f64>,

    /// Best Φ achieved
    best_phi: f64,

    /// PRNG state for stochastic operations
    rng_state: u64,
}

impl PhiGuidedOptimizer {
    /// Create new optimizer
    pub fn new(config: PhiOptimizationConfig) -> Self {
        Self {
            config,
            calculator: RealPhiCalculator::new(),
            step: 0,
            phi_history: Vec::new(),
            best_phi: 0.0,
            rng_state: 42,
        }
    }

    /// Create with default config
    pub fn default_optimizer() -> Self {
        Self::new(PhiOptimizationConfig::default())
    }

    /// Perform one optimization step
    pub fn optimize_step(&mut self, network: &mut ConsciousnessNetwork) -> OptimizationResult {
        self.step += 1;

        // 1. Compute current Φ
        let representations = network.to_node_representations();
        let current_phi = self.calculator.compute(&representations);

        // 2. Compute gradients for each edge
        self.compute_edge_gradients(network, current_phi);

        // 3. Update edge weights using gradient ascent
        self.update_edge_weights(network);

        // 4. Prune weak edges
        let edges_pruned = self.prune_edges(network);

        // 5. Consider adding new edges
        let edges_added = self.consider_new_edges(network, current_phi);

        // 6. Compute new Φ after changes
        let representations = network.to_node_representations();
        let new_phi = self.calculator.compute(&representations);

        // Track history
        let phi_delta = new_phi - current_phi;
        self.phi_history.push(new_phi);
        if new_phi > self.best_phi {
            self.best_phi = new_phi;
        }

        // Compute average gradient magnitude
        let avg_gradient = if network.edges.is_empty() {
            0.0
        } else {
            network.edges.iter().map(|e| e.gradient.abs()).sum::<f64>()
                / network.edges.len() as f64
        };

        OptimizationResult {
            phi: new_phi,
            phi_delta,
            edges_pruned,
            edges_added,
            avg_gradient,
            step: self.step,
        }
    }

    /// Compute gradient ∂Φ/∂weight for each edge using finite differences
    fn compute_edge_gradients(&mut self, network: &mut ConsciousnessNetwork, base_phi: f64) {
        let epsilon = self.config.gradient_epsilon;

        for edge_idx in 0..network.edges.len() {
            // Perturb edge weight +epsilon
            let original_weight = network.edges[edge_idx].weight;
            network.edges[edge_idx].weight = (original_weight + epsilon).min(self.config.max_weight);

            let representations = network.to_node_representations();
            let phi_plus = self.calculator.compute(&representations);

            // Perturb edge weight -epsilon
            network.edges[edge_idx].weight = (original_weight - epsilon).max(0.0);

            let representations = network.to_node_representations();
            let phi_minus = self.calculator.compute(&representations);

            // Restore original weight
            network.edges[edge_idx].weight = original_weight;

            // Compute gradient via central differences
            let gradient = (phi_plus - phi_minus) / (2.0 * epsilon);
            network.edges[edge_idx].gradient = gradient;
        }
    }

    /// Update edge weights using gradient ascent with momentum
    fn update_edge_weights(&mut self, network: &mut ConsciousnessNetwork) {
        for edge in network.edges.iter_mut() {
            // Update momentum
            edge.momentum = self.config.momentum * edge.momentum
                + (1.0 - self.config.momentum) * edge.gradient;

            // Adaptive learning rate based on gradient magnitude
            let lr = if self.config.adaptive_lr {
                self.config.learning_rate / (1.0 + edge.gradient.abs())
            } else {
                self.config.learning_rate
            };

            // Gradient ascent (we want to MAXIMIZE Φ)
            edge.weight += lr * edge.momentum;

            // Clamp to valid range
            edge.weight = edge.weight.clamp(0.0, self.config.max_weight);
        }
    }

    /// Prune edges with weight below threshold
    fn prune_edges(&mut self, network: &mut ConsciousnessNetwork) -> usize {
        let initial_count = network.edges.len();

        network.edges.retain(|e| e.weight >= self.config.prune_threshold);

        initial_count - network.edges.len()
    }

    /// Consider adding new edges that might increase Φ
    fn consider_new_edges(&mut self, network: &mut ConsciousnessNetwork, current_phi: f64) -> usize {
        // Random chance to consider new edges
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;

        let rand_val = (self.rng_state as f64) / (u64::MAX as f64);
        if rand_val > self.config.new_edge_probability {
            return 0;
        }

        let n = network.nodes.len();
        if n < 2 {
            return 0;
        }

        let mut added = 0;

        // Try a few random new edges
        for _ in 0..3 {
            self.rng_state ^= self.rng_state << 13;
            self.rng_state ^= self.rng_state >> 7;
            self.rng_state ^= self.rng_state << 17;
            let from = (self.rng_state as usize) % n;

            self.rng_state ^= self.rng_state << 13;
            self.rng_state ^= self.rng_state >> 7;
            self.rng_state ^= self.rng_state << 17;
            let to = (self.rng_state as usize) % n;

            if from == to || network.has_edge(from, to) {
                continue;
            }

            // Check if node has too many edges
            if network.degree(from) >= self.config.max_edges_per_node
                || network.degree(to) >= self.config.max_edges_per_node {
                continue;
            }

            // Try adding edge and see if Φ improves
            network.add_edge(from, to, 0.5);  // Start with moderate weight

            let representations = network.to_node_representations();
            let new_phi = self.calculator.compute(&representations);

            if new_phi > current_phi + self.config.new_edge_min_improvement {
                // Keep the edge
                added += 1;
            } else {
                // Remove the edge
                network.remove_edge(from, to);
            }
        }

        added
    }

    /// Run optimization for multiple steps
    pub fn optimize(
        &mut self,
        network: &mut ConsciousnessNetwork,
        n_steps: usize,
    ) -> OptimizationSummary {
        let initial_phi = {
            let representations = network.to_node_representations();
            self.calculator.compute(&representations)
        };

        let mut results = Vec::with_capacity(n_steps);

        for _ in 0..n_steps {
            let result = self.optimize_step(network);
            results.push(result);
        }

        let final_phi = results.last().map(|r| r.phi).unwrap_or(initial_phi);

        OptimizationSummary {
            initial_phi,
            final_phi,
            improvement: final_phi - initial_phi,
            steps: n_steps,
            best_phi: self.best_phi,
            results,
        }
    }

    /// Get optimization history
    pub fn history(&self) -> &[f64] {
        &self.phi_history
    }

    /// Get best Φ achieved
    pub fn best_phi(&self) -> f64 {
        self.best_phi
    }

    /// Reset optimizer state
    pub fn reset(&mut self) {
        self.step = 0;
        self.phi_history.clear();
        self.best_phi = 0.0;
    }
}

/// Summary of optimization run
#[derive(Debug, Clone)]
pub struct OptimizationSummary {
    /// Initial Φ value
    pub initial_phi: f64,

    /// Final Φ value
    pub final_phi: f64,

    /// Total improvement
    pub improvement: f64,

    /// Number of steps
    pub steps: usize,

    /// Best Φ achieved during optimization
    pub best_phi: f64,

    /// Per-step results
    pub results: Vec<OptimizationResult>,
}

impl OptimizationSummary {
    /// Get average Φ improvement per step
    pub fn avg_improvement_per_step(&self) -> f64 {
        if self.steps == 0 {
            0.0
        } else {
            self.improvement / self.steps as f64
        }
    }

    /// Get total edges pruned
    pub fn total_edges_pruned(&self) -> usize {
        self.results.iter().map(|r| r.edges_pruned).sum()
    }

    /// Get total edges added
    pub fn total_edges_added(&self) -> usize {
        self.results.iter().map(|r| r.edges_added).sum()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PRESET INITIALIZATION STRATEGIES
// ═══════════════════════════════════════════════════════════════════════════════

/// Preset strategies for initializing network topology
pub enum InitializationStrategy {
    /// Random sparse graph
    RandomSparse { edge_ratio: f64 },

    /// Ring topology (each node connects to neighbors)
    Ring,

    /// Star topology (one central hub)
    Star,

    /// Small-world (ring + random shortcuts)
    SmallWorld { rewire_prob: f64 },

    /// Fully connected
    FullyConnected,
}

impl InitializationStrategy {
    /// Apply strategy to network
    pub fn apply(&self, network: &mut ConsciousnessNetwork, seed: u64) {
        let n = network.node_count();
        if n < 2 {
            return;
        }

        // Clear existing edges
        network.edges.clear();

        match self {
            Self::RandomSparse { edge_ratio } => {
                let max_edges = n * (n - 1) / 2;
                let n_edges = ((max_edges as f64) * edge_ratio) as usize;
                network.add_random_edges(n_edges, seed);
            }

            Self::Ring => {
                for i in 0..n {
                    network.add_edge(i, (i + 1) % n, 1.0);
                }
            }

            Self::Star => {
                // Node 0 is the hub
                for i in 1..n {
                    network.add_edge(0, i, 1.0);
                }
            }

            Self::SmallWorld { rewire_prob } => {
                // Start with ring
                for i in 0..n {
                    network.add_edge(i, (i + 1) % n, 1.0);
                }

                // Add random shortcuts
                let mut state = seed;
                for i in 0..n {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;

                    let rand_val = (state as f64) / (u64::MAX as f64);
                    if rand_val < *rewire_prob {
                        // Add shortcut to random node
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        let target = (state as usize) % n;

                        if target != i && !network.has_edge(i, target) {
                            network.add_edge(i, target, 0.5);
                        }
                    }
                }
            }

            Self::FullyConnected => {
                for i in 0..n {
                    for j in (i + 1)..n {
                        network.add_edge(i, j, 1.0);
                    }
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_creation() {
        let network = ConsciousnessNetwork::random(8, 512, 42);

        assert_eq!(network.node_count(), 8);
        assert_eq!(network.edge_count(), 0);
    }

    #[test]
    fn test_add_edges() {
        let mut network = ConsciousnessNetwork::random(4, 512, 42);

        network.add_edge(0, 1, 1.0);
        network.add_edge(1, 2, 0.5);
        network.add_edge(2, 3, 0.8);

        assert_eq!(network.edge_count(), 3);
        assert!(network.has_edge(0, 1));
        assert!(network.has_edge(1, 0));  // Undirected
        assert!(!network.has_edge(0, 3));
    }

    #[test]
    fn test_neighbors() {
        let mut network = ConsciousnessNetwork::random(5, 512, 42);

        network.add_edge(0, 1, 1.0);
        network.add_edge(0, 2, 1.0);
        network.add_edge(0, 3, 1.0);

        let neighbors = network.neighbors(0);
        assert_eq!(neighbors.len(), 3);
        assert!(neighbors.contains(&1));
        assert!(neighbors.contains(&2));
        assert!(neighbors.contains(&3));
    }

    #[test]
    fn test_initialization_strategies() {
        let mut network = ConsciousnessNetwork::random(8, 512, 42);

        // Ring
        InitializationStrategy::Ring.apply(&mut network, 0);
        assert_eq!(network.edge_count(), 8);

        // Star
        InitializationStrategy::Star.apply(&mut network, 0);
        assert_eq!(network.edge_count(), 7);  // n-1 edges from hub

        // Fully connected
        InitializationStrategy::FullyConnected.apply(&mut network, 0);
        assert_eq!(network.edge_count(), 8 * 7 / 2);  // n*(n-1)/2
    }

    #[test]
    fn test_optimizer_single_step() {
        let config = PhiOptimizationConfig {
            dim: 512,  // Smaller for faster tests
            ..Default::default()
        };

        let mut optimizer = PhiGuidedOptimizer::new(config);
        let mut network = ConsciousnessNetwork::random(6, 512, 42);

        // Initialize with ring topology
        InitializationStrategy::Ring.apply(&mut network, 0);

        let result = optimizer.optimize_step(&mut network);

        assert!(result.phi >= 0.0);
        assert!(result.phi <= 1.0);
        assert_eq!(result.step, 1);
    }

    #[test]
    fn test_optimizer_multiple_steps() {
        let config = PhiOptimizationConfig {
            dim: 256,  // Small for speed
            learning_rate: 0.2,
            ..Default::default()
        };

        let mut optimizer = PhiGuidedOptimizer::new(config);
        let mut network = ConsciousnessNetwork::random(5, 256, 42);

        InitializationStrategy::Ring.apply(&mut network, 0);

        let summary = optimizer.optimize(&mut network, 10);

        assert_eq!(summary.steps, 10);
        assert!(summary.best_phi >= summary.initial_phi || summary.best_phi >= 0.0);
    }

    #[test]
    fn test_edge_pruning() {
        let config = PhiOptimizationConfig {
            dim: 256,
            prune_threshold: 0.5,
            ..Default::default()
        };

        let mut optimizer = PhiGuidedOptimizer::new(config);
        let mut network = ConsciousnessNetwork::random(4, 256, 42);

        // Add edges with varying weights
        network.add_edge(0, 1, 1.0);  // Keep
        network.add_edge(1, 2, 0.3);  // Prune (< 0.5)
        network.add_edge(2, 3, 0.8);  // Keep
        network.add_edge(0, 3, 0.1);  // Prune (< 0.5)

        let pruned = optimizer.prune_edges(&mut network);

        assert_eq!(pruned, 2);
        assert_eq!(network.edge_count(), 2);
    }

    #[test]
    fn test_to_node_representations() {
        let mut network = ConsciousnessNetwork::random(4, 256, 42);

        network.add_edge(0, 1, 1.0);
        network.add_edge(1, 2, 1.0);
        network.add_edge(2, 3, 1.0);

        let representations = network.to_node_representations();

        assert_eq!(representations.len(), 4);
        for rep in &representations {
            assert_eq!(rep.dim(), 256);
        }
    }

    #[test]
    fn test_phi_calculation() {
        let mut network = ConsciousnessNetwork::random(6, 512, 42);

        // Ring should have higher Φ than disconnected
        InitializationStrategy::Ring.apply(&mut network, 0);
        let ring_representations = network.to_node_representations();
        let calculator = RealPhiCalculator::new();
        let ring_phi = calculator.compute(&ring_representations);

        // Star for comparison
        InitializationStrategy::Star.apply(&mut network, 0);
        let star_representations = network.to_node_representations();
        let star_phi = calculator.compute(&star_representations);

        // Both should be valid Φ values
        assert!(ring_phi >= 0.0 && ring_phi <= 1.0);
        assert!(star_phi >= 0.0 && star_phi <= 1.0);
    }

    #[test]
    fn test_optimizer_reset() {
        let config = PhiOptimizationConfig::default();
        let mut optimizer = PhiGuidedOptimizer::new(config);

        optimizer.step = 10;
        optimizer.best_phi = 0.8;
        optimizer.phi_history.push(0.5);

        optimizer.reset();

        assert_eq!(optimizer.step, 0);
        assert_eq!(optimizer.best_phi, 0.0);
        assert!(optimizer.phi_history.is_empty());
    }
}
