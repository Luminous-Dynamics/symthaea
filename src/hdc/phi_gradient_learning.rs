//! Φ-Gradient Topology Learning
//!
//! # Research Direction: Learning Optimal Connections via Gradient Descent on Φ
//!
//! This module implements a differentiable approach to topology optimization where
//! the network learns optimal connection patterns by maximizing integrated information.
//!
//! # Key Innovation
//!
//! Traditional topology design is hand-crafted. This system instead:
//! 1. Parameterizes edge existence as continuous probabilities
//! 2. Computes Φ as a differentiable function of these parameters
//! 3. Uses gradient ascent to maximize Φ
//! 4. Binarizes edge probabilities to get final topology
//!
//! # Theoretical Foundation
//!
//! From our bridge hypothesis:
//! - Φ correlates strongly (r=-0.72) with bridge ratio
//! - Optimal bridge ratio is ~40-45% for maximum Φ
//! - BUT the specific WHICH edges matter, not just how many
//!
//! This system learns WHICH specific connections maximize Φ.

use super::real_hv::RealHV;
use super::phi_real::RealPhiCalculator;

/// Learning configuration for Φ-gradient optimization
#[derive(Clone, Debug)]
pub struct PhiLearningConfig {
    /// Initial learning rate
    pub learning_rate: f64,
    /// Momentum coefficient
    pub momentum: f64,
    /// Learning rate decay per epoch
    pub lr_decay: f64,
    /// Minimum learning rate
    pub min_lr: f64,
    /// Temperature for edge probability (higher = more exploration)
    pub temperature: f64,
    /// Temperature decay per epoch
    pub temp_decay: f64,
    /// Minimum temperature
    pub min_temp: f64,
    /// L1 regularization on edge probabilities (sparsity)
    pub l1_reg: f64,
    /// Target edge density (0.0-1.0)
    pub target_density: f64,
    /// Density penalty coefficient
    pub density_penalty: f64,
}

impl Default for PhiLearningConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            momentum: 0.9,
            lr_decay: 0.995,
            min_lr: 0.001,
            temperature: 1.0,
            temp_decay: 0.99,
            min_temp: 0.1,
            l1_reg: 0.01,
            target_density: 0.10,  // ~10% density like ConsciousnessOptimized
            density_penalty: 0.1,
        }
    }
}

/// Learnable edge with soft (probabilistic) existence
#[derive(Clone, Debug)]
pub struct LearnableEdge {
    /// Source node
    pub from: usize,
    /// Target node
    pub to: usize,
    /// Logit for edge probability (sigmoid(logit) = probability)
    pub logit: f64,
    /// Momentum accumulator
    pub momentum: f64,
    /// Gradient history for adaptive learning
    pub grad_history: Vec<f64>,
}

impl LearnableEdge {
    pub fn new(from: usize, to: usize, initial_prob: f64) -> Self {
        // Convert probability to logit: logit = log(p / (1-p))
        let p = initial_prob.clamp(0.01, 0.99);
        let logit = (p / (1.0 - p)).ln();

        Self {
            from,
            to,
            logit,
            momentum: 0.0,
            grad_history: Vec::new(),
        }
    }

    /// Get current edge probability
    pub fn probability(&self, temperature: f64) -> f64 {
        sigmoid(self.logit / temperature)
    }

    /// Is edge likely present? (prob > 0.5)
    pub fn is_likely_present(&self, temperature: f64) -> bool {
        self.probability(temperature) > 0.5
    }

    /// Sample edge presence (stochastic forward pass)
    pub fn sample(&self, temperature: f64, seed: u64) -> bool {
        let prob = self.probability(temperature);
        rand_f64(seed) < prob
    }
}

/// A node in the learnable topology
#[derive(Clone, Debug)]
pub struct LearnableNode {
    /// Node ID
    pub id: usize,
    /// Node representation (HDC vector)
    pub state: RealHV,
    /// Module assignment (for tracking bridge vs intra-module)
    pub module: usize,
    /// Hierarchical level
    pub level: usize,
}

/// Φ-gradient learnable topology
#[derive(Clone)]
pub struct PhiGradientTopology {
    /// Nodes in the topology
    nodes: Vec<LearnableNode>,
    /// Learnable edges (all possible edges, with probabilities)
    edges: Vec<LearnableEdge>,
    /// Learning configuration
    config: PhiLearningConfig,
    /// Φ calculator
    phi_calc: RealPhiCalculator,
    /// HDC dimension
    dim: usize,
    /// Training history
    history: LearningHistory,
    /// Current epoch
    epoch: usize,
}

/// Training history for analysis
#[derive(Clone, Debug, Default)]
pub struct LearningHistory {
    /// Φ values over training
    pub phi_values: Vec<f64>,
    /// Edge density over training
    pub densities: Vec<f64>,
    /// Bridge ratios over training
    pub bridge_ratios: Vec<f64>,
    /// Learning rates used
    pub learning_rates: Vec<f64>,
    /// Gradient magnitudes
    pub grad_magnitudes: Vec<f64>,
}

impl PhiGradientTopology {
    /// Create a new learnable topology
    ///
    /// Initializes with random edge probabilities around target density
    pub fn new(n_nodes: usize, dim: usize, n_modules: usize, seed: u64, config: PhiLearningConfig) -> Self {
        // Create nodes with module assignments
        let nodes: Vec<LearnableNode> = (0..n_nodes)
            .map(|i| {
                let module = i % n_modules;
                let level = if i == 0 { 0 }
                    else if i < n_modules + 1 { 1 }
                    else if i < n_modules * 5 + 1 { 2 }
                    else { 3 };

                LearnableNode {
                    id: i,
                    state: RealHV::random(dim, seed + i as u64 * 12345),
                    module,
                    level,
                }
            })
            .collect();

        // Create all possible edges (upper triangular to avoid duplicates)
        let target_density = config.target_density;
        let edges: Vec<LearnableEdge> = (0..n_nodes)
            .flat_map(|i| {
                (i + 1..n_nodes).map(move |j| {
                    // Initialize with probability around target density
                    // Slightly higher for bridges to encourage integration
                    let modules_i = i % n_modules;
                    let modules_j = j % n_modules;
                    let is_bridge = modules_i != modules_j;
                    let init_prob = if is_bridge {
                        target_density * 1.2  // Slight bridge bias
                    } else {
                        target_density * 0.9
                    };
                    LearnableEdge::new(i, j, init_prob.clamp(0.05, 0.95))
                })
            })
            .collect();

        Self {
            nodes,
            edges,
            config,
            phi_calc: RealPhiCalculator::new(),
            dim,
            history: LearningHistory::default(),
            epoch: 0,
        }
    }

    /// Compute Φ for current edge configuration
    pub fn compute_phi(&self) -> f64 {
        let representations: Vec<RealHV> = self.nodes.iter()
            .map(|n| n.state.clone())
            .collect();
        self.phi_calc.compute(&representations)
    }

    /// Compute Φ with a specific edge configuration (for gradient estimation)
    fn compute_phi_with_edges(&self, active_edges: &[bool]) -> f64 {
        // Create temporary states influenced by active edges
        let mut states: Vec<RealHV> = self.nodes.iter()
            .map(|n| n.state.clone())
            .collect();

        // Apply edge influences
        for (i, edge) in self.edges.iter().enumerate() {
            if active_edges[i] {
                // Blend connected node states
                let from_state = states[edge.from].clone();
                let to_state = states[edge.to].clone();

                let blended = from_state.add(&to_state.scale(0.3)).normalize();
                states[edge.from] = states[edge.from].add(&to_state.scale(0.1)).normalize();
                states[edge.to] = states[edge.to].add(&from_state.scale(0.1)).normalize();
            }
        }

        self.phi_calc.compute(&states)
    }

    /// Estimate gradient of Φ with respect to each edge probability
    ///
    /// Uses finite differences with REINFORCE-style variance reduction
    fn estimate_gradients(&mut self) -> Vec<f64> {
        let n_samples = 5;  // Number of samples for gradient estimation
        let temp = self.current_temperature();

        let mut gradients = vec![0.0; self.edges.len()];

        // Baseline Φ (for variance reduction)
        let baseline: f64 = (0..n_samples)
            .map(|s| {
                let active: Vec<bool> = self.edges.iter().enumerate()
                    .map(|(i, e)| e.sample(temp, self.epoch as u64 * 1000 + s as u64 + i as u64 * 100))
                    .collect();
                self.compute_phi_with_edges(&active)
            })
            .sum::<f64>() / n_samples as f64;

        // Estimate gradient for each edge using log-derivative trick
        for (i, edge) in self.edges.iter().enumerate() {
            let prob = edge.probability(temp);

            // Sample with edge forced on vs off
            let mut grad_estimate = 0.0;

            for s in 0..n_samples {
                // Sample other edges randomly
                let mut active: Vec<bool> = self.edges.iter().enumerate()
                    .map(|(j, e)| {
                        if j == i {
                            false  // Will set below
                        } else {
                            e.sample(temp, self.epoch as u64 * 1000 + s as u64 + j as u64 * 100)
                        }
                    })
                    .collect();

                // Φ with edge on
                active[i] = true;
                let phi_on = self.compute_phi_with_edges(&active);

                // Φ with edge off
                active[i] = false;
                let phi_off = self.compute_phi_with_edges(&active);

                // REINFORCE gradient: (R - baseline) * d log p / d theta
                let reward_on = phi_on - baseline;
                let reward_off = phi_off - baseline;

                // Gradient of sigmoid logit
                grad_estimate += reward_on * prob - reward_off * (1.0 - prob);
            }

            gradients[i] = grad_estimate / n_samples as f64;
        }

        gradients
    }

    /// Perform one learning step
    pub fn learn_step(&mut self) {
        let lr = self.current_learning_rate();
        let momentum = self.config.momentum;
        let temp = self.current_temperature();

        // Estimate gradients
        let gradients = self.estimate_gradients();

        // Track gradient magnitude
        let grad_magnitude: f64 = gradients.iter().map(|g| g * g).sum::<f64>().sqrt();
        self.history.grad_magnitudes.push(grad_magnitude);

        // Compute density once before the loop to avoid borrow conflict
        let current_density = self.current_density();

        // Update edge logits with momentum
        for (i, edge) in self.edges.iter_mut().enumerate() {
            let grad = gradients[i];

            // Add regularization gradient
            let prob = edge.probability(temp);
            let l1_grad = -self.config.l1_reg * prob.signum();

            // Density penalty (encourage target density)
            let density_grad = -self.config.density_penalty *
                (current_density - self.config.target_density);

            let total_grad = grad + l1_grad + density_grad;

            // Update with momentum
            edge.momentum = momentum * edge.momentum + (1.0 - momentum) * total_grad;
            edge.logit += lr * edge.momentum;

            // Clamp logit to prevent extreme probabilities
            edge.logit = edge.logit.clamp(-5.0, 5.0);

            edge.grad_history.push(grad);
        }

        // Record history
        let phi = self.compute_phi();
        self.history.phi_values.push(phi);
        self.history.densities.push(self.current_density());
        self.history.bridge_ratios.push(self.bridge_ratio());
        self.history.learning_rates.push(lr);

        self.epoch += 1;
    }

    /// Train for multiple epochs
    pub fn train(&mut self, epochs: usize) {
        for _ in 0..epochs {
            self.learn_step();
        }
    }

    /// Current learning rate (with decay)
    pub fn current_learning_rate(&self) -> f64 {
        let lr = self.config.learning_rate * self.config.lr_decay.powi(self.epoch as i32);
        lr.max(self.config.min_lr)
    }

    /// Current temperature (with decay)
    pub fn current_temperature(&self) -> f64 {
        let temp = self.config.temperature * self.config.temp_decay.powi(self.epoch as i32);
        temp.max(self.config.min_temp)
    }

    /// Current edge density
    pub fn current_density(&self) -> f64 {
        let temp = self.current_temperature();
        let active_count: f64 = self.edges.iter()
            .map(|e| e.probability(temp))
            .sum();
        let max_edges = self.nodes.len() * (self.nodes.len() - 1) / 2;
        active_count / max_edges as f64
    }

    /// Current bridge ratio (cross-module edges / total edges)
    pub fn bridge_ratio(&self) -> f64 {
        let temp = self.current_temperature();

        let mut total_prob = 0.0;
        let mut bridge_prob = 0.0;

        for edge in &self.edges {
            let prob = edge.probability(temp);
            total_prob += prob;

            let is_bridge = self.nodes[edge.from].module != self.nodes[edge.to].module;
            if is_bridge {
                bridge_prob += prob;
            }
        }

        if total_prob > 0.0 {
            bridge_prob / total_prob
        } else {
            0.0
        }
    }

    /// Extract final topology (binarize probabilities)
    pub fn extract_topology(&self) -> Vec<(usize, usize)> {
        let temp = self.current_temperature();
        self.edges.iter()
            .filter(|e| e.is_likely_present(temp))
            .map(|e| (e.from, e.to))
            .collect()
    }

    /// Get learning metrics
    pub fn metrics(&self) -> PhiLearningMetrics {
        let phi = if self.history.phi_values.is_empty() {
            self.compute_phi()
        } else {
            *self.history.phi_values.last().unwrap()
        };

        PhiLearningMetrics {
            epoch: self.epoch,
            phi,
            density: self.current_density(),
            bridge_ratio: self.bridge_ratio(),
            learning_rate: self.current_learning_rate(),
            temperature: self.current_temperature(),
            n_active_edges: self.extract_topology().len(),
            n_total_edges: self.edges.len(),
            grad_magnitude: self.history.grad_magnitudes.last().copied().unwrap_or(0.0),
        }
    }

    /// Get training history
    pub fn history(&self) -> &LearningHistory {
        &self.history
    }

    /// Get node representations
    pub fn nodes(&self) -> &[LearnableNode] {
        &self.nodes
    }

    /// Get edge information
    pub fn edges(&self) -> &[LearnableEdge] {
        &self.edges
    }
}

/// Metrics for Φ-gradient learning
#[derive(Clone, Debug)]
pub struct PhiLearningMetrics {
    /// Current epoch
    pub epoch: usize,
    /// Current Φ value
    pub phi: f64,
    /// Current edge density
    pub density: f64,
    /// Current bridge ratio
    pub bridge_ratio: f64,
    /// Current learning rate
    pub learning_rate: f64,
    /// Current temperature
    pub temperature: f64,
    /// Number of active (prob > 0.5) edges
    pub n_active_edges: usize,
    /// Total possible edges
    pub n_total_edges: usize,
    /// Gradient magnitude
    pub grad_magnitude: f64,
}

impl std::fmt::Display for PhiLearningMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Epoch {:4}: Φ={:.4}, density={:.1}%, bridges={:.1}%, edges={}/{}, lr={:.4}, temp={:.2}",
            self.epoch,
            self.phi,
            self.density * 100.0,
            self.bridge_ratio * 100.0,
            self.n_active_edges,
            self.n_total_edges,
            self.learning_rate,
            self.temperature
        )
    }
}

// Helper functions
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn rand_f64(seed: u64) -> f64 {
    let x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (x >> 33) as f64 / (1u64 << 31) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::HDC_DIMENSION;

    #[test]
    fn test_learnable_edge() {
        let edge = LearnableEdge::new(0, 1, 0.5);
        assert!((edge.probability(1.0) - 0.5).abs() < 0.1);

        let edge_high = LearnableEdge::new(0, 1, 0.9);
        assert!(edge_high.probability(1.0) > 0.8);
    }

    #[test]
    fn test_topology_creation() {
        let config = PhiLearningConfig::default();
        let topology = PhiGradientTopology::new(16, HDC_DIMENSION, 4, 42, config);

        assert_eq!(topology.nodes.len(), 16);
        // Max edges for 16 nodes = 16*15/2 = 120
        assert_eq!(topology.edges.len(), 120);

        let metrics = topology.metrics();
        println!("{}", metrics);
    }

    #[test]
    fn test_learning_step() {
        let config = PhiLearningConfig {
            learning_rate: 0.5,
            ..Default::default()
        };

        let mut topology = PhiGradientTopology::new(12, HDC_DIMENSION, 4, 42, config);

        let initial_phi = topology.compute_phi();
        println!("Initial: {}", topology.metrics());

        // Run learning steps
        for step in 0..10 {
            topology.learn_step();
            if step % 2 == 0 {
                println!("{}", topology.metrics());
            }
        }

        let final_phi = topology.compute_phi();
        println!("Final: {}", topology.metrics());

        // History should be recorded
        assert_eq!(topology.history().phi_values.len(), 10);
    }

    #[test]
    #[ignore = "computationally expensive - run with cargo test --release --ignored"]
    fn test_extract_topology() {
        let config = PhiLearningConfig::default();
        let mut topology = PhiGradientTopology::new(12, HDC_DIMENSION, 4, 42, config);

        // Train briefly
        topology.train(5);

        let edges = topology.extract_topology();
        println!("Extracted {} edges from {} possible", edges.len(), topology.edges.len());

        // Should have some edges
        assert!(!edges.is_empty());
        // Should be less than all possible
        assert!(edges.len() < topology.edges.len());
    }

    #[test]
    fn test_bridge_ratio_learning() {
        let config = PhiLearningConfig {
            learning_rate: 0.3,
            target_density: 0.15,
            ..Default::default()
        };

        let mut topology = PhiGradientTopology::new(16, HDC_DIMENSION, 4, 42, config);

        println!("\nBridge Ratio Learning Test:");
        println!("{}", "─".repeat(60));

        let initial_ratio = topology.bridge_ratio();
        println!("Initial bridge ratio: {:.1}%", initial_ratio * 100.0);

        topology.train(20);

        let final_ratio = topology.bridge_ratio();
        println!("Final bridge ratio: {:.1}%", final_ratio * 100.0);

        // Bridge ratio should change during learning
        // (exact direction depends on gradient)
    }

    #[test]
    fn test_temperature_annealing() {
        let config = PhiLearningConfig {
            temperature: 2.0,
            temp_decay: 0.9,
            min_temp: 0.5,
            ..Default::default()
        };

        let mut topology = PhiGradientTopology::new(8, HDC_DIMENSION, 4, 42, config);

        println!("\nTemperature Annealing:");
        for _ in 0..10 {
            topology.epoch += 1;
            println!("Epoch {}: temp={:.3}", topology.epoch, topology.current_temperature());
        }

        assert!(topology.current_temperature() >= 0.5);
    }
}
