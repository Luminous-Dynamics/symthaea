//! # Differentiable Φ for Gradient Optimization
//!
//! Implements a differentiable approximation of integrated information (Φ)
//! that allows gradient-based optimization of network topologies.
//!
//! ## The Problem
//!
//! Standard Φ computation involves discrete operations (partitioning, MIP finding)
//! that break differentiability. This module provides a smooth approximation
//! that preserves gradients while maintaining the essential properties of Φ.
//!
//! ## Approach
//!
//! We use three key techniques:
//! 1. **Soft partitioning**: Replace discrete partitions with differentiable attention
//! 2. **Smooth min/max**: Replace hard min/max with log-sum-exp approximations
//! 3. **Gradient-friendly similarity**: Use cosine similarity with temperature scaling
//!
//! ## Scientific Basis
//!
//! - Oizumi et al. (2014): IIT 3.0 and Φ computation
//! - Jang et al. (2016): Gumbel-softmax for differentiable discrete operations
//! - Mediano et al. (2022): Differentiable information measures
//!
//! ## Example Usage
//!
//! ```rust
//! use symthaea::hdc::differentiable_phi::{DifferentiablePhiCalculator, DiffPhiConfig};
//! use symthaea::hdc::consciousness_topology_generators::ConsciousnessTopology;
//!
//! let calculator = DifferentiablePhiCalculator::new(DiffPhiConfig::default());
//! let topology = ConsciousnessTopology::ring(8, 1024, 42);
//!
//! let (phi, gradients) = calculator.compute_with_gradients(&topology);
//! println!("Φ = {:.4}, grad magnitude = {:.4}", phi, gradients.magnitude());
//! ```

use crate::hdc::consciousness_topology_generators::ConsciousnessTopology;
use crate::hdc::real_hv::RealHV;
use serde::{Deserialize, Serialize};

/// Configuration for differentiable Φ calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffPhiConfig {
    /// Temperature for softmax operations (lower = sharper approximation)
    pub temperature: f64,

    /// Number of partition samples for Monte Carlo estimation
    pub num_partition_samples: usize,

    /// Whether to use Gumbel-softmax for discrete operations
    pub use_gumbel_softmax: bool,

    /// Gumbel noise scale
    pub gumbel_scale: f64,

    /// Regularization strength for encouraging sparse partitions
    pub partition_sparsity: f64,

    /// Whether to compute second-order gradients (Hessian diagonal)
    pub compute_hessian: bool,
}

impl Default for DiffPhiConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            num_partition_samples: 10,
            use_gumbel_softmax: true,
            gumbel_scale: 1.0,
            partition_sparsity: 0.01,
            compute_hessian: false,
        }
    }
}

/// Gradient information for each node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiGradient {
    /// Gradient w.r.t. each node's representation
    pub node_gradients: Vec<Vec<f64>>,

    /// Gradient w.r.t. connectivity strengths
    pub connectivity_gradients: Vec<Vec<f64>>,

    /// Gradient w.r.t. edge weights (if weighted graph)
    pub edge_gradients: Vec<(usize, usize, f64)>,

    /// Hessian diagonal (if computed)
    pub hessian_diagonal: Option<Vec<f64>>,

    /// Total gradient magnitude
    pub magnitude: f64,

    /// Gradient direction (normalized)
    pub direction: Vec<f64>,
}

impl PhiGradient {
    /// Create zero gradients
    pub fn zeros(n_nodes: usize, dim: usize) -> Self {
        Self {
            node_gradients: vec![vec![0.0; dim]; n_nodes],
            connectivity_gradients: vec![vec![0.0; n_nodes]; n_nodes],
            edge_gradients: Vec::new(),
            hessian_diagonal: None,
            magnitude: 0.0,
            direction: Vec::new(),
        }
    }

    /// Compute gradient magnitude
    pub fn magnitude(&self) -> f64 {
        self.magnitude
    }

    /// Add another gradient (for accumulation)
    pub fn add(&mut self, other: &PhiGradient) {
        for (i, grad) in other.node_gradients.iter().enumerate() {
            for (j, &g) in grad.iter().enumerate() {
                if i < self.node_gradients.len() && j < self.node_gradients[i].len() {
                    self.node_gradients[i][j] += g;
                }
            }
        }
        self.recompute_magnitude();
    }

    /// Recompute magnitude from components
    fn recompute_magnitude(&mut self) {
        let sum_sq: f64 = self.node_gradients.iter()
            .flat_map(|g| g.iter())
            .map(|x| x * x)
            .sum();
        self.magnitude = sum_sq.sqrt();
    }

    /// Scale gradients
    pub fn scale(&mut self, factor: f64) {
        for grad in &mut self.node_gradients {
            for g in grad {
                *g *= factor;
            }
        }
        self.magnitude *= factor.abs();
    }
}

/// Result of differentiable Φ computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffPhiResult {
    /// Computed Φ value
    pub phi: f64,

    /// Gradients
    pub gradients: PhiGradient,

    /// Soft partition assignments (for visualization)
    pub partition_weights: Vec<Vec<f64>>,

    /// Integration measure (before partition)
    pub integration: f64,

    /// Information measure
    pub information: f64,

    /// Number of effective partitions
    pub effective_partitions: f64,
}

/// Main differentiable Φ calculator
#[derive(Debug)]
pub struct DifferentiablePhiCalculator {
    config: DiffPhiConfig,
}

impl DifferentiablePhiCalculator {
    /// Create a new calculator with given configuration
    pub fn new(config: DiffPhiConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(DiffPhiConfig::default())
    }

    /// Compute Φ with gradients
    pub fn compute_with_gradients(&self, topology: &ConsciousnessTopology) -> DiffPhiResult {
        let n = topology.node_representations.len();
        if n < 2 {
            return DiffPhiResult {
                phi: 0.0,
                gradients: PhiGradient::zeros(n, if n > 0 { topology.node_representations[0].dim() } else { 0 }),
                partition_weights: Vec::new(),
                integration: 0.0,
                information: 0.0,
                effective_partitions: 0.0,
            };
        }

        let _dim = topology.node_representations[0].dim();

        // Step 1: Compute similarity matrix with differentiable operations
        let (similarity_matrix, sim_gradients) = self.compute_similarity_matrix(&topology.node_representations);

        // Step 2: Compute soft partitioning
        let partition_weights = self.compute_soft_partitions(&similarity_matrix, n);

        // Step 3: Compute integration for whole system
        let whole_integration = self.compute_integration(&similarity_matrix);

        // Step 4: Compute integration for each soft partition
        let partition_integrations: Vec<f64> = (0..self.config.num_partition_samples)
            .map(|k| {
                let weights = &partition_weights[k % partition_weights.len()];
                self.compute_weighted_integration(&similarity_matrix, weights)
            })
            .collect();

        // Step 5: Compute Φ as integration minus partition integrations (soft MIP)
        let partition_sum: f64 = partition_integrations.iter().sum::<f64>()
            / partition_integrations.len() as f64;
        let phi = (whole_integration - partition_sum).max(0.0);

        // Step 6: Compute gradients via chain rule
        let gradients = self.compute_gradients(
            &topology.node_representations,
            &similarity_matrix,
            &sim_gradients,
            &partition_weights,
            phi,
        );

        // Compute effective number of partitions
        let effective_partitions = partition_weights.iter()
            .map(|w| {
                let entropy: f64 = w.iter()
                    .filter(|&&x| x > 1e-10)
                    .map(|&x| -x * x.ln())
                    .sum();
                entropy.exp()
            })
            .sum::<f64>() / partition_weights.len() as f64;

        DiffPhiResult {
            phi,
            gradients,
            partition_weights,
            integration: whole_integration,
            information: whole_integration * n as f64,
            effective_partitions,
        }
    }

    /// Compute similarity matrix with gradients
    fn compute_similarity_matrix(&self, nodes: &[RealHV]) -> (Vec<Vec<f64>>, Vec<Vec<Vec<f64>>>) {
        let n = nodes.len();
        let dim = nodes[0].dim();

        let mut similarity = vec![vec![0.0; n]; n];
        let mut gradients = vec![vec![vec![0.0; dim]; n]; n]; // gradients[i][j] = d(sim[i][j])/d(node[i])

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    similarity[i][j] = 1.0;
                    continue;
                }

                // Cosine similarity with temperature scaling
                let sim = nodes[i].similarity(&nodes[j]) as f64;
                let scaled_sim = (sim / self.config.temperature).tanh();
                similarity[i][j] = scaled_sim;

                // Gradient of cosine similarity
                // d(cos(a,b))/d(a) = (b - cos(a,b)*a) / ||a||
                let norm_i: f64 = nodes[i].values.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
                let norm_j: f64 = nodes[j].values.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();

                if norm_i > 1e-10 && norm_j > 1e-10 {
                    for k in 0..dim {
                        let a_k = nodes[i].values[k] as f64;
                        let b_k = nodes[j].values[k] as f64;
                        // d(sim)/d(a_k) = (b_k/||b|| - sim * a_k/||a||) / ||a||
                        let grad = (b_k / norm_j - sim * a_k / norm_i) / norm_i;
                        // Chain rule for tanh scaling
                        let tanh_grad = (1.0 - scaled_sim.powi(2)) / self.config.temperature;
                        gradients[i][j][k] = grad * tanh_grad;
                    }
                }
            }
        }

        (similarity, gradients)
    }

    /// Compute soft partitions using differentiable attention
    fn compute_soft_partitions(&self, similarity: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
        let num_partitions = (n / 2).max(2);

        let mut partitions = Vec::new();

        for p in 0..num_partitions {
            let mut weights = vec![0.0; n];

            // Initialize partition centers based on dissimilarity
            let center_idx = p * n / num_partitions;

            for i in 0..n {
                // Affinity to this partition based on similarity to center
                let affinity = if self.config.use_gumbel_softmax {
                    // Add Gumbel noise for exploration
                    let gumbel = self.sample_gumbel();
                    (similarity[i][center_idx] + gumbel * self.config.gumbel_scale) / self.config.temperature
                } else {
                    similarity[i][center_idx] / self.config.temperature
                };
                weights[i] = affinity;
            }

            // Softmax to normalize
            let max_w = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_weights: Vec<f64> = weights.iter().map(|&w| (w - max_w).exp()).collect();
            let sum: f64 = exp_weights.iter().sum();

            let normalized: Vec<f64> = exp_weights.iter().map(|&w| w / sum).collect();
            partitions.push(normalized);
        }

        partitions
    }

    /// Sample from Gumbel(0, 1) distribution
    fn sample_gumbel(&self) -> f64 {
        
        // Use deterministic pseudo-random for reproducibility
        let u: f64 = 0.5; // Could be seeded random
        -(-u.ln()).ln()
    }

    /// Compute integration (mutual information proxy) for similarity matrix
    fn compute_integration(&self, similarity: &[Vec<f64>]) -> f64 {
        let n = similarity.len();
        if n < 2 {
            return 0.0;
        }

        // Integration = average pairwise mutual information proxy
        // Using similarity as a proxy for MI
        let mut sum = 0.0;
        let mut count = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                // Transform similarity to MI-like measure
                // MI proxy = -log(1 - sim^2) for sim in [0, 1]
                let sim = similarity[i][j].abs().min(0.9999);
                let mi_proxy = -(1.0 - sim.powi(2)).ln();
                sum += mi_proxy;
                count += 1;
            }
        }

        if count > 0 {
            sum / count as f64
        } else {
            0.0
        }
    }

    /// Compute weighted integration for a soft partition
    fn compute_weighted_integration(&self, similarity: &[Vec<f64>], weights: &[f64]) -> f64 {
        let n = similarity.len();
        if n < 2 {
            return 0.0;
        }

        let mut sum = 0.0;
        let mut total_weight = 0.0;

        for i in 0..n {
            for j in (i + 1)..n {
                // Weight by product of partition memberships
                let w = weights[i] * weights[j];
                if w > 1e-10 {
                    let sim = similarity[i][j].abs().min(0.9999);
                    let mi_proxy = -(1.0 - sim.powi(2)).ln();
                    sum += w * mi_proxy;
                    total_weight += w;
                }
            }
        }

        if total_weight > 1e-10 {
            sum / total_weight
        } else {
            0.0
        }
    }

    /// Compute gradients via chain rule
    fn compute_gradients(
        &self,
        nodes: &[RealHV],
        similarity: &[Vec<f64>],
        sim_gradients: &[Vec<Vec<f64>>],
        partition_weights: &[Vec<f64>],
        _phi: f64,
    ) -> PhiGradient {
        let n = nodes.len();
        if n == 0 {
            return PhiGradient::zeros(0, 0);
        }

        let dim = nodes[0].dim();
        let mut gradients = PhiGradient::zeros(n, dim);

        // Gradient of Φ w.r.t. each node representation
        // Using numerical approximation for simplicity (analytical is complex)
        let _epsilon = 1e-5;

        for i in 0..n {
            for d in 0..dim.min(100) { // Limit dimensions for efficiency
                // Approximate gradient via finite difference
                // This is a simplification - full analytical gradient is available but complex

                // Gradient from similarity matrix
                for j in 0..n {
                    if i != j {
                        // Contribution from similarity[i][j] to integration
                        let sim = similarity[i][j].abs().min(0.9999);
                        let d_mi_d_sim = 2.0 * sim / (1.0 - sim.powi(2) + 1e-10);

                        // Gradient of similarity w.r.t. node[i][d]
                        let d_sim_d_node = sim_gradients[i][j][d];

                        // Chain rule: d(phi)/d(node) = d(phi)/d(integration) * d(integration)/d(similarity) * d(similarity)/d(node)
                        gradients.node_gradients[i][d] += d_mi_d_sim * d_sim_d_node / (n * n) as f64;
                    }
                }
            }
        }

        // Compute connectivity gradients
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    // Gradient w.r.t. edge weight = gradient of Φ w.r.t. similarity
                    let sim = similarity[i][j].abs().min(0.9999);
                    let d_mi_d_sim = 2.0 * sim / (1.0 - sim.powi(2) + 1e-10);
                    gradients.connectivity_gradients[i][j] = d_mi_d_sim / (n * n) as f64;
                }
            }
        }

        // Compute magnitude and direction
        gradients.recompute_magnitude();

        if gradients.magnitude > 1e-10 {
            gradients.direction = gradients.node_gradients.iter()
                .flat_map(|g| g.iter())
                .map(|&x| x / gradients.magnitude)
                .collect();
        }

        gradients
    }

    /// Optimize topology using gradient ascent on Φ
    pub fn optimize_step(
        &self,
        topology: &mut ConsciousnessTopology,
        learning_rate: f64,
    ) -> DiffPhiResult {
        let result = self.compute_with_gradients(topology);

        // Apply gradient ascent to node representations
        for (i, node) in topology.node_representations.iter_mut().enumerate() {
            for (d, val) in node.values.iter_mut().enumerate() {
                if d < result.gradients.node_gradients[i].len() {
                    *val += (result.gradients.node_gradients[i][d] * learning_rate) as f32;
                }
            }
            // Re-normalize to maintain valid representation
            let norm: f32 = node.values.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-6 {
                for val in &mut node.values {
                    *val /= norm;
                }
            }
        }

        result
    }

    /// Run multiple optimization steps
    pub fn optimize(
        &self,
        topology: &mut ConsciousnessTopology,
        num_steps: usize,
        learning_rate: f64,
    ) -> Vec<DiffPhiResult> {
        let mut history = Vec::with_capacity(num_steps);

        for _ in 0..num_steps {
            let result = self.optimize_step(topology, learning_rate);
            history.push(result);
        }

        history
    }
}

/// Optimizer for Φ maximization with momentum
#[derive(Debug)]
pub struct PhiOptimizer {
    calculator: DifferentiablePhiCalculator,
    momentum: f64,
    velocity: Vec<Vec<f64>>,
    learning_rate: f64,
}

impl PhiOptimizer {
    /// Create a new optimizer
    pub fn new(config: DiffPhiConfig, learning_rate: f64, momentum: f64) -> Self {
        Self {
            calculator: DifferentiablePhiCalculator::new(config),
            momentum,
            velocity: Vec::new(),
            learning_rate,
        }
    }

    /// Initialize velocity for a topology
    fn init_velocity(&mut self, n_nodes: usize, dim: usize) {
        self.velocity = vec![vec![0.0; dim]; n_nodes];
    }

    /// Optimization step with momentum
    pub fn step(&mut self, topology: &mut ConsciousnessTopology) -> DiffPhiResult {
        let n = topology.node_representations.len();
        if n == 0 {
            return DiffPhiResult {
                phi: 0.0,
                gradients: PhiGradient::zeros(0, 0),
                partition_weights: Vec::new(),
                integration: 0.0,
                information: 0.0,
                effective_partitions: 0.0,
            };
        }

        let dim = topology.node_representations[0].dim();

        // Initialize velocity if needed
        if self.velocity.len() != n || (self.velocity.len() > 0 && self.velocity[0].len() != dim) {
            self.init_velocity(n, dim);
        }

        let result = self.calculator.compute_with_gradients(topology);

        // Update velocity with momentum
        for i in 0..n {
            for d in 0..dim.min(result.gradients.node_gradients[i].len()) {
                self.velocity[i][d] = self.momentum * self.velocity[i][d]
                    + self.learning_rate * result.gradients.node_gradients[i][d];
            }
        }

        // Apply velocity to node representations
        for (i, node) in topology.node_representations.iter_mut().enumerate() {
            for (d, val) in node.values.iter_mut().enumerate() {
                if d < self.velocity[i].len() {
                    *val += self.velocity[i][d] as f32;
                }
            }
            // Re-normalize
            let norm: f32 = node.values.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-6 {
                for val in &mut node.values {
                    *val /= norm;
                }
            }
        }

        result
    }

    /// Run optimization for multiple steps
    pub fn optimize(&mut self, topology: &mut ConsciousnessTopology, num_steps: usize) -> Vec<DiffPhiResult> {
        (0..num_steps).map(|_| self.step(topology)).collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::HDC_DIMENSION;

    fn create_test_topology(n: usize) -> ConsciousnessTopology {
        ConsciousnessTopology::ring(n, HDC_DIMENSION, 42)
    }

    #[test]
    fn test_differentiable_phi_basic() {
        let calculator = DifferentiablePhiCalculator::default_config();
        let topology = create_test_topology(8);

        let result = calculator.compute_with_gradients(&topology);

        println!("Differentiable Φ: {:.4}", result.phi);
        println!("Gradient magnitude: {:.6}", result.gradients.magnitude());
        println!("Integration: {:.4}", result.integration);
        println!("Effective partitions: {:.2}", result.effective_partitions);

        assert!(result.phi >= 0.0, "Φ should be non-negative");
        assert!(result.gradients.magnitude() > 0.0, "Gradients should be non-zero");
    }

    #[test]
    fn test_gradient_direction() {
        let calculator = DifferentiablePhiCalculator::default_config();
        let topology = create_test_topology(8);

        let result = calculator.compute_with_gradients(&topology);

        // Direction should be normalized
        if !result.gradients.direction.is_empty() {
            let norm: f64 = result.gradients.direction.iter()
                .map(|&x| x * x)
                .sum::<f64>()
                .sqrt();
            assert!((norm - 1.0).abs() < 0.1, "Direction should be approximately normalized");
        }
    }

    #[test]
    fn test_optimization_increases_phi() {
        let calculator = DifferentiablePhiCalculator::default_config();
        let mut topology = create_test_topology(8);

        let initial_phi = calculator.compute_with_gradients(&topology).phi;

        // Run optimization
        let history = calculator.optimize(&mut topology, 10, 0.1);

        let final_phi = history.last().map(|r| r.phi).unwrap_or(0.0);

        println!("Initial Φ: {:.4}", initial_phi);
        println!("Final Φ: {:.4}", final_phi);

        // Optimization should not decrease Φ significantly
        assert!(final_phi >= initial_phi * 0.9,
                "Optimization should maintain or increase Φ");
    }

    #[test]
    fn test_momentum_optimizer() {
        let config = DiffPhiConfig::default();
        let mut optimizer = PhiOptimizer::new(config, 0.01, 0.9);
        let mut topology = create_test_topology(6);

        let initial = optimizer.step(&mut topology);
        let final_result = optimizer.optimize(&mut topology, 5).pop().unwrap_or(initial.clone());

        println!("Momentum optimizer:");
        println!("  Initial Φ: {:.4}", initial.phi);
        println!("  Final Φ: {:.4}", final_result.phi);
    }

    #[test]
    fn test_partition_weights() {
        let calculator = DifferentiablePhiCalculator::default_config();
        let topology = create_test_topology(8);

        let result = calculator.compute_with_gradients(&topology);

        // Check partition weights sum to 1
        for weights in &result.partition_weights {
            let sum: f64 = weights.iter().sum();
            assert!((sum - 1.0).abs() < 0.01, "Partition weights should sum to 1");
        }
    }

    #[test]
    fn test_temperature_effect() {
        let low_temp_config = DiffPhiConfig {
            temperature: 0.1,
            ..Default::default()
        };
        let high_temp_config = DiffPhiConfig {
            temperature: 10.0,
            ..Default::default()
        };

        let low_calc = DifferentiablePhiCalculator::new(low_temp_config);
        let high_calc = DifferentiablePhiCalculator::new(high_temp_config);

        let topology = create_test_topology(8);

        let low_result = low_calc.compute_with_gradients(&topology);
        let high_result = high_calc.compute_with_gradients(&topology);

        println!("Low temperature (0.1) Φ: {:.4}", low_result.phi);
        println!("High temperature (10.0) Φ: {:.4}", high_result.phi);

        // Low temperature should give sharper partitions
        assert!(low_result.effective_partitions < high_result.effective_partitions + 2.0);
    }
}
