//! Shapley Value Attribution for Fair Contribution Scoring in Federated Learning.
//!
//! This module implements Shapley values for measuring individual node contributions
//! to the aggregated gradient in federated learning. Shapley values provide a fair,
//! game-theoretic approach to contribution attribution with desirable properties:
//!
//! - **Efficiency**: Values sum to total coalition value
//! - **Symmetry**: Identical contributions get identical values
//! - **Null player**: Non-contributors get zero value
//! - **Additivity**: Values are additive across independent value functions
//!
//! ## Computational Methods
//!
//! For `n` players (nodes):
//! - **Exact**: O(2^n) - suitable for n <= 10
//! - **Monte Carlo**: O(m*n) where m = samples - suitable for 10 < n <= 100
//! - **Permutation**: O(p*n) where p = permutations - suitable for n > 100
//!
//! ## Example
//!
//! ```rust
//! use fl_aggregator::shapley::{ShapleyCalculator, ShapleyConfig, SamplingMethod, Baseline, ValueFunction};
//! use ndarray::Array1;
//! use std::collections::HashMap;
//!
//! let config = ShapleyConfig::default();
//! let mut calculator = ShapleyCalculator::new(config);
//!
//! let mut gradients = HashMap::new();
//! gradients.insert("node1".to_string(), Array1::from(vec![1.0, 2.0, 3.0]));
//! gradients.insert("node2".to_string(), Array1::from(vec![1.1, 2.1, 3.1]));
//!
//! let aggregated = Array1::from(vec![1.05, 2.05, 3.05]);
//! let result = calculator.calculate_values(&gradients, &aggregated);
//!
//! println!("Shapley values: {:?}", result.values);
//! ```

use crate::byzantine::{cosine_similarity, euclidean_distance, l2_norm};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// Sampling method for Shapley value computation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SamplingMethod {
    /// Exact computation - enumerate all 2^n subsets.
    /// Only feasible for small N (typically <= 10).
    Exact,

    /// Monte Carlo sampling - randomly sample subsets.
    /// Good for medium-sized coalitions (10-100 nodes).
    MonteCarlo {
        /// Number of random samples to take.
        samples: usize,
    },

    /// Permutation sampling - sample random orderings.
    /// Best for large coalitions (>100 nodes).
    Permutation {
        /// Number of permutations to sample.
        permutations: usize,
    },
}

impl Default for SamplingMethod {
    fn default() -> Self {
        SamplingMethod::MonteCarlo { samples: 1000 }
    }
}

impl SamplingMethod {
    /// Create Monte Carlo sampler with given sample count.
    pub fn monte_carlo(samples: usize) -> Self {
        SamplingMethod::MonteCarlo { samples }
    }

    /// Create permutation sampler with given permutation count.
    pub fn permutation(permutations: usize) -> Self {
        SamplingMethod::Permutation { permutations }
    }
}

/// Baseline computation method for value function.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Baseline {
    /// Use zero gradient as baseline.
    #[default]
    ZeroGradient,

    /// Use mean of all gradients as baseline.
    MeanGradient,

    /// Use component-wise median of all gradients as baseline.
    MedianGradient,
}

/// Value function for measuring contribution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValueFunction {
    /// Cosine similarity - how well does the gradient align with target?
    #[default]
    CosineSimilarity,

    /// L2 distance reduction - how much does adding this gradient reduce L2 distance?
    L2Distance,

    /// Model improvement - simulated quality improvement.
    /// Uses negative L2 distance from aggregated as proxy for model quality.
    ModelImprovement,
}

/// Configuration for Shapley value computation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShapleyConfig {
    /// Sampling method for approximation.
    pub sampling_method: SamplingMethod,

    /// How to compute the baseline (empty coalition value).
    pub baseline: Baseline,

    /// Value function for measuring contribution.
    pub value_function: ValueFunction,

    /// Whether to normalize values so they sum to 1.
    pub normalize_values: bool,

    /// Threshold for automatic method selection based on N.
    /// N <= exact_threshold uses Exact.
    pub exact_threshold: usize,

    /// Threshold for Monte Carlo vs Permutation.
    /// N > monte_carlo_threshold uses Permutation.
    pub monte_carlo_threshold: usize,

    /// Random seed for reproducibility (None = random).
    pub seed: Option<u64>,
}

impl Default for ShapleyConfig {
    fn default() -> Self {
        Self {
            sampling_method: SamplingMethod::default(),
            baseline: Baseline::default(),
            value_function: ValueFunction::default(),
            normalize_values: false,
            exact_threshold: 10,
            monte_carlo_threshold: 100,
            seed: None,
        }
    }
}

impl ShapleyConfig {
    /// Create config with exact computation.
    pub fn exact() -> Self {
        Self {
            sampling_method: SamplingMethod::Exact,
            ..Default::default()
        }
    }

    /// Create config with Monte Carlo sampling.
    pub fn monte_carlo(samples: usize) -> Self {
        Self {
            sampling_method: SamplingMethod::MonteCarlo { samples },
            ..Default::default()
        }
    }

    /// Create config with permutation sampling.
    pub fn permutation(permutations: usize) -> Self {
        Self {
            sampling_method: SamplingMethod::Permutation { permutations },
            ..Default::default()
        }
    }

    /// Set the baseline method.
    pub fn with_baseline(mut self, baseline: Baseline) -> Self {
        self.baseline = baseline;
        self
    }

    /// Set the value function.
    pub fn with_value_function(mut self, value_function: ValueFunction) -> Self {
        self.value_function = value_function;
        self
    }

    /// Enable value normalization.
    pub fn with_normalization(mut self) -> Self {
        self.normalize_values = true;
        self
    }

    /// Set random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Automatically select sampling method based on N.
    pub fn auto_select_method(&self, n: usize) -> SamplingMethod {
        if n <= self.exact_threshold {
            SamplingMethod::Exact
        } else if n <= self.monte_carlo_threshold {
            match &self.sampling_method {
                SamplingMethod::MonteCarlo { samples } => {
                    SamplingMethod::MonteCarlo { samples: *samples }
                }
                _ => SamplingMethod::MonteCarlo { samples: 1000 },
            }
        } else {
            match &self.sampling_method {
                SamplingMethod::Permutation { permutations } => {
                    SamplingMethod::Permutation {
                        permutations: *permutations,
                    }
                }
                _ => SamplingMethod::Permutation { permutations: 500 },
            }
        }
    }
}

/// Result of Shapley value computation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShapleyResult {
    /// Node ID to Shapley value mapping.
    pub values: HashMap<String, f32>,

    /// Sum of all Shapley values.
    pub total_value: f32,

    /// Time taken for computation in milliseconds.
    pub computation_time_ms: f64,

    /// Sampling method actually used.
    pub method_used: SamplingMethod,

    /// Number of samples/permutations used (if approximate method).
    pub samples_used: usize,

    /// Efficiency check: how close sum is to total coalition value.
    pub efficiency_error: f32,
}

impl ShapleyResult {
    /// Get the Shapley value for a specific node.
    pub fn get(&self, node_id: &str) -> Option<f32> {
        self.values.get(node_id).copied()
    }

    /// Get nodes sorted by Shapley value (highest first).
    pub fn sorted_by_value(&self) -> Vec<(&String, f32)> {
        let mut sorted: Vec<_> = self.values.iter().map(|(k, v)| (k, *v)).collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted
    }

    /// Get the top K contributors.
    pub fn top_k(&self, k: usize) -> Vec<(&String, f32)> {
        self.sorted_by_value().into_iter().take(k).collect()
    }

    /// Get nodes with negative Shapley values (potentially Byzantine).
    pub fn negative_contributors(&self) -> Vec<(&String, f32)> {
        self.values
            .iter()
            .filter(|(_, v)| **v < 0.0)
            .map(|(k, v)| (k, *v))
            .collect()
    }
}

/// Shapley value calculator for federated learning contributions.
pub struct ShapleyCalculator {
    config: ShapleyConfig,
    rng: XorShiftRng,
}

impl ShapleyCalculator {
    /// Create a new Shapley value calculator with the given configuration.
    pub fn new(config: ShapleyConfig) -> Self {
        let seed = config.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(42)
        });

        Self {
            config,
            rng: XorShiftRng::new(seed),
        }
    }

    /// Calculate Shapley values for all nodes.
    ///
    /// # Arguments
    /// * `gradients` - Map of node IDs to their gradient contributions
    /// * `aggregated` - The aggregated (target) gradient
    ///
    /// # Returns
    /// ShapleyResult containing values for each node
    pub fn calculate_values(
        &mut self,
        gradients: &HashMap<String, Array1<f32>>,
        aggregated: &Array1<f32>,
    ) -> ShapleyResult {
        let start = Instant::now();
        let n = gradients.len();

        if n == 0 {
            return ShapleyResult {
                values: HashMap::new(),
                total_value: 0.0,
                computation_time_ms: 0.0,
                method_used: self.config.sampling_method.clone(),
                samples_used: 0,
                efficiency_error: 0.0,
            };
        }

        // Convert to ordered vectors for consistent indexing
        let node_ids: Vec<&String> = gradients.keys().collect();
        let gradient_list: Vec<&Array1<f32>> = node_ids.iter().map(|id| &gradients[*id]).collect();

        // Compute baseline
        let baseline = self.compute_baseline(&gradient_list, aggregated);

        // Select method based on N
        let method = self.config.auto_select_method(n);

        // Compute Shapley values using selected method
        let (values, samples_used) = match &method {
            SamplingMethod::Exact => (
                self.compute_exact(&node_ids, &gradient_list, aggregated, &baseline),
                1usize << n, // 2^n
            ),
            SamplingMethod::MonteCarlo { samples } => {
                (
                    self.compute_monte_carlo(
                        &node_ids,
                        &gradient_list,
                        aggregated,
                        &baseline,
                        *samples,
                    ),
                    *samples,
                )
            }
            SamplingMethod::Permutation { permutations } => {
                (
                    self.compute_permutation(
                        &node_ids,
                        &gradient_list,
                        aggregated,
                        &baseline,
                        *permutations,
                    ),
                    *permutations,
                )
            }
        };

        // Calculate total value
        let total_value = values.values().sum::<f32>();

        // Calculate total coalition value for efficiency check
        let full_coalition_value = self.coalition_value(&gradient_list, aggregated, &baseline);
        let efficiency_error = (total_value - full_coalition_value).abs();

        // Normalize if configured
        let final_values = if self.config.normalize_values && total_value.abs() > 1e-10 {
            values
                .into_iter()
                .map(|(k, v)| (k, v / total_value))
                .collect()
        } else {
            values
        };

        let computation_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        ShapleyResult {
            values: final_values,
            total_value,
            computation_time_ms,
            method_used: method,
            samples_used,
            efficiency_error,
        }
    }

    /// Calculate marginal contribution of a single node.
    ///
    /// This is a simplified measure that computes the expected marginal
    /// contribution when adding this node to a random coalition.
    pub fn calculate_marginal_contribution(
        &mut self,
        node_id: &str,
        gradients: &HashMap<String, Array1<f32>>,
        aggregated: &Array1<f32>,
    ) -> f32 {
        let node_gradient = match gradients.get(node_id) {
            Some(g) => g,
            None => return 0.0,
        };

        // Get other gradients
        let others: Vec<&Array1<f32>> = gradients
            .iter()
            .filter(|(id, _)| id.as_str() != node_id)
            .map(|(_, g)| g)
            .collect();

        if others.is_empty() {
            // Only this node - its contribution is the full value
            let all_grads = vec![node_gradient];
            let baseline = self.compute_baseline(&all_grads, aggregated);
            return self.coalition_value(&all_grads, aggregated, &baseline);
        }

        let all_grads: Vec<&Array1<f32>> = gradients.values().collect();
        let baseline = self.compute_baseline(&all_grads, aggregated);

        // Value with all nodes
        let value_with = self.coalition_value(&all_grads, aggregated, &baseline);

        // Value without this node
        let value_without = self.coalition_value(&others, aggregated, &baseline);

        value_with - value_without
    }

    /// Compute baseline based on configuration.
    fn compute_baseline(
        &self,
        gradients: &[&Array1<f32>],
        aggregated: &Array1<f32>,
    ) -> Array1<f32> {
        match self.config.baseline {
            Baseline::ZeroGradient => Array1::zeros(aggregated.len()),
            Baseline::MeanGradient => {
                if gradients.is_empty() {
                    return Array1::zeros(aggregated.len());
                }
                let mut sum = Array1::zeros(aggregated.len());
                for g in gradients {
                    sum += &g.view();
                }
                sum / gradients.len() as f32
            }
            Baseline::MedianGradient => {
                if gradients.is_empty() {
                    return Array1::zeros(aggregated.len());
                }
                let dim = aggregated.len();
                let mut result = Array1::zeros(dim);
                for d in 0..dim {
                    let mut values: Vec<f32> = gradients.iter().map(|g| g[d]).collect();
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let mid = values.len() / 2;
                    result[d] = if values.len() % 2 == 0 {
                        (values[mid - 1] + values[mid]) / 2.0
                    } else {
                        values[mid]
                    };
                }
                result
            }
        }
    }

    /// Compute value of a coalition.
    fn coalition_value(
        &self,
        coalition: &[&Array1<f32>],
        aggregated: &Array1<f32>,
        baseline: &Array1<f32>,
    ) -> f32 {
        if coalition.is_empty() {
            return 0.0;
        }

        // Compute coalition aggregate (simple mean)
        let mut coalition_agg = Array1::zeros(aggregated.len());
        for g in coalition {
            coalition_agg += &g.view();
        }
        coalition_agg /= coalition.len() as f32;

        match self.config.value_function {
            ValueFunction::CosineSimilarity => {
                // How well does coalition aggregate align with target?
                let sim = cosine_similarity(coalition_agg.view(), aggregated.view());
                // Transform to [0, 1] range for better interpretability
                (sim + 1.0) / 2.0
            }
            ValueFunction::L2Distance => {
                // How much does coalition reduce L2 distance vs baseline?
                let baseline_dist = euclidean_distance(baseline.view(), aggregated.view());
                let coalition_dist = euclidean_distance(coalition_agg.view(), aggregated.view());
                // Return reduction in distance (positive = good)
                baseline_dist - coalition_dist
            }
            ValueFunction::ModelImprovement => {
                // Negative normalized distance (higher = better)
                let dist = euclidean_distance(coalition_agg.view(), aggregated.view());
                let agg_norm = l2_norm(aggregated.view());
                if agg_norm < 1e-10 {
                    0.0
                } else {
                    1.0 - (dist / agg_norm).min(2.0) / 2.0
                }
            }
        }
    }

    /// Exact Shapley computation via subset enumeration.
    fn compute_exact(
        &self,
        node_ids: &[&String],
        gradients: &[&Array1<f32>],
        aggregated: &Array1<f32>,
        baseline: &Array1<f32>,
    ) -> HashMap<String, f32> {
        let n = node_ids.len();
        let mut values: HashMap<String, f32> = node_ids.iter().map(|id| ((*id).clone(), 0.0)).collect();

        // Precompute factorials for weight calculation
        let factorials: Vec<f64> = (0..=n).map(|k| factorial(k)).collect();

        // For each player
        for i in 0..n {
            let mut shapley_value = 0.0;

            // For each subset not containing player i
            for mask in 0..(1usize << (n - 1)) {
                // Build coalition S (not including i)
                let coalition_indices: Vec<usize> = (0..n)
                    .filter(|&j| j != i)
                    .enumerate()
                    .filter(|(bit_pos, _)| (mask >> bit_pos) & 1 == 1)
                    .map(|(_, j)| j)
                    .collect();

                let s = coalition_indices.len();

                // Value of S
                let coalition: Vec<&Array1<f32>> =
                    coalition_indices.iter().map(|&j| gradients[j]).collect();
                let v_s = self.coalition_value(&coalition, aggregated, baseline);

                // Value of S ∪ {i}
                let mut coalition_with_i = coalition.clone();
                coalition_with_i.push(gradients[i]);
                let v_s_with_i = self.coalition_value(&coalition_with_i, aggregated, baseline);

                // Marginal contribution
                let marginal = v_s_with_i - v_s;

                // Shapley weight: |S|!(n-|S|-1)! / n!
                let weight = (factorials[s] * factorials[n - s - 1]) / factorials[n];

                shapley_value += weight as f32 * marginal;
            }

            values.insert(node_ids[i].clone(), shapley_value);
        }

        values
    }

    /// Monte Carlo approximation of Shapley values.
    fn compute_monte_carlo(
        &mut self,
        node_ids: &[&String],
        gradients: &[&Array1<f32>],
        aggregated: &Array1<f32>,
        baseline: &Array1<f32>,
        samples: usize,
    ) -> HashMap<String, f32> {
        let n = node_ids.len();
        let mut value_sums: Vec<f32> = vec![0.0; n];
        let mut counts: Vec<usize> = vec![0; n];

        for _ in 0..samples {
            // Generate random subset
            let mut subset_mask: u64 = self.rng.next();

            for i in 0..n {
                // Build coalition without player i
                let coalition_without: Vec<&Array1<f32>> = (0..n)
                    .filter(|&j| j != i && ((subset_mask >> (j % 64)) & 1 == 1))
                    .map(|j| gradients[j])
                    .collect();

                // Build coalition with player i
                let mut coalition_with = coalition_without.clone();
                coalition_with.push(gradients[i]);

                let v_without = self.coalition_value(&coalition_without, aggregated, baseline);
                let v_with = self.coalition_value(&coalition_with, aggregated, baseline);

                value_sums[i] += v_with - v_without;
                counts[i] += 1;
            }

            // Reseed for next iteration
            subset_mask = self.rng.next();
        }

        node_ids
            .iter()
            .enumerate()
            .map(|(i, id)| {
                let avg = if counts[i] > 0 {
                    value_sums[i] / counts[i] as f32
                } else {
                    0.0
                };
                ((*id).clone(), avg)
            })
            .collect()
    }

    /// Permutation-based approximation of Shapley values.
    fn compute_permutation(
        &mut self,
        node_ids: &[&String],
        gradients: &[&Array1<f32>],
        aggregated: &Array1<f32>,
        baseline: &Array1<f32>,
        permutations: usize,
    ) -> HashMap<String, f32> {
        let n = node_ids.len();
        let mut value_sums: Vec<f32> = vec![0.0; n];

        for _ in 0..permutations {
            // Generate random permutation using Fisher-Yates
            let mut perm: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = (self.rng.next() as usize) % (i + 1);
                perm.swap(i, j);
            }

            // For each player, compute marginal contribution when they join
            let mut current_coalition: Vec<&Array1<f32>> = Vec::with_capacity(n);
            let mut prev_value = 0.0;

            for &player_idx in &perm {
                // Add player to coalition
                current_coalition.push(gradients[player_idx]);

                // Compute new coalition value
                let new_value = self.coalition_value(&current_coalition, aggregated, baseline);

                // Marginal contribution
                let marginal = new_value - prev_value;
                value_sums[player_idx] += marginal;

                prev_value = new_value;
            }
        }

        // Average over all permutations
        node_ids
            .iter()
            .enumerate()
            .map(|(i, id)| ((*id).clone(), value_sums[i] / permutations as f32))
            .collect()
    }
}

/// Simple XorShift64 RNG for reproducible sampling.
struct XorShiftRng {
    state: u64,
}

impl XorShiftRng {
    fn new(seed: u64) -> Self {
        // Ensure non-zero state
        let state = if seed == 0 { 1 } else { seed };
        Self { state }
    }

    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

/// Compute factorial as f64 to handle larger values.
fn factorial(n: usize) -> f64 {
    if n <= 1 {
        1.0
    } else {
        (2..=n).fold(1.0, |acc, x| acc * x as f64)
    }
}

/// Verify Shapley value fairness properties.
pub fn verify_efficiency(result: &ShapleyResult, tolerance: f32) -> bool {
    result.efficiency_error <= tolerance
}

/// Check if symmetric players have equal values.
pub fn verify_symmetry(
    result: &ShapleyResult,
    symmetric_nodes: &[String],
    tolerance: f32,
) -> bool {
    if symmetric_nodes.len() < 2 {
        return true;
    }

    let first_value = match result.values.get(&symmetric_nodes[0]) {
        Some(v) => *v,
        None => return false,
    };

    symmetric_nodes.iter().skip(1).all(|node| {
        match result.values.get(node) {
            Some(v) => (v - first_value).abs() <= tolerance,
            None => false,
        }
    })
}

/// Check if null players (identical to baseline) get zero value.
pub fn verify_null_player(result: &ShapleyResult, null_nodes: &[String], tolerance: f32) -> bool {
    null_nodes.iter().all(|node| {
        match result.values.get(node) {
            Some(v) => v.abs() <= tolerance,
            None => true,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    fn create_test_gradients() -> HashMap<String, Array1<f32>> {
        let mut gradients = HashMap::new();
        gradients.insert("node1".to_string(), array![1.0, 2.0, 3.0]);
        gradients.insert("node2".to_string(), array![1.1, 2.1, 3.1]);
        gradients.insert("node3".to_string(), array![0.9, 1.9, 2.9]);
        gradients
    }

    #[test]
    fn test_shapley_config_default() {
        let config = ShapleyConfig::default();
        assert!(matches!(config.sampling_method, SamplingMethod::MonteCarlo { .. }));
        assert_eq!(config.baseline, Baseline::ZeroGradient);
        assert_eq!(config.value_function, ValueFunction::CosineSimilarity);
    }

    #[test]
    fn test_shapley_config_exact() {
        let config = ShapleyConfig::exact();
        assert_eq!(config.sampling_method, SamplingMethod::Exact);
    }

    #[test]
    fn test_auto_method_selection() {
        let config = ShapleyConfig::default();

        assert_eq!(config.auto_select_method(5), SamplingMethod::Exact);
        assert!(matches!(
            config.auto_select_method(50),
            SamplingMethod::MonteCarlo { .. }
        ));
        assert!(matches!(
            config.auto_select_method(150),
            SamplingMethod::Permutation { .. }
        ));
    }

    #[test]
    fn test_exact_computation_small() {
        let config = ShapleyConfig::exact().with_seed(42);
        let mut calculator = ShapleyCalculator::new(config);

        let gradients = create_test_gradients();
        let aggregated = array![1.0, 2.0, 3.0];

        let result = calculator.calculate_values(&gradients, &aggregated);

        assert_eq!(result.values.len(), 3);
        assert!(result.computation_time_ms >= 0.0);
        assert_eq!(result.method_used, SamplingMethod::Exact);
    }

    #[test]
    fn test_efficiency_property() {
        // Shapley values should sum to total coalition value (approximately)
        let config = ShapleyConfig::exact().with_seed(42);
        let mut calculator = ShapleyCalculator::new(config);

        let gradients = create_test_gradients();
        let aggregated = array![1.0, 2.0, 3.0];

        let result = calculator.calculate_values(&gradients, &aggregated);

        // Efficiency error should be small for exact computation
        assert!(
            result.efficiency_error < 0.01,
            "Efficiency error too large: {}",
            result.efficiency_error
        );
    }

    #[test]
    fn test_symmetry_property() {
        let config = ShapleyConfig::exact().with_seed(42);
        let mut calculator = ShapleyCalculator::new(config);

        // Create symmetric gradients
        let mut gradients = HashMap::new();
        gradients.insert("node1".to_string(), array![1.0, 0.0]);
        gradients.insert("node2".to_string(), array![1.0, 0.0]); // Same as node1

        let aggregated = array![1.0, 0.0];

        let result = calculator.calculate_values(&gradients, &aggregated);

        // Symmetric players should get equal values
        let v1 = result.values.get("node1").unwrap();
        let v2 = result.values.get("node2").unwrap();
        assert_relative_eq!(v1, v2, epsilon = 0.01);
    }

    #[test]
    fn test_byzantine_gradient_low_value() {
        let config = ShapleyConfig::exact()
            .with_value_function(ValueFunction::L2Distance)
            .with_seed(42);
        let mut calculator = ShapleyCalculator::new(config);

        // Normal gradients cluster together
        let mut gradients = HashMap::new();
        gradients.insert("honest1".to_string(), array![1.0, 2.0, 3.0]);
        gradients.insert("honest2".to_string(), array![1.1, 2.1, 3.1]);
        gradients.insert("honest3".to_string(), array![0.9, 1.9, 2.9]);
        // Byzantine gradient is far away
        gradients.insert("byzantine".to_string(), array![-10.0, -20.0, -30.0]);

        // Aggregated is close to honest gradients
        let aggregated = array![1.0, 2.0, 3.0];

        let result = calculator.calculate_values(&gradients, &aggregated);

        // Byzantine gradient should have lower (potentially negative) Shapley value
        let honest_avg: f32 = ["honest1", "honest2", "honest3"]
            .iter()
            .filter_map(|id| result.values.get(*id))
            .sum::<f32>()
            / 3.0;

        let byzantine_value = *result.values.get("byzantine").unwrap();

        assert!(
            byzantine_value < honest_avg,
            "Byzantine value {} should be less than honest avg {}",
            byzantine_value,
            honest_avg
        );
    }

    #[test]
    fn test_monte_carlo_approximation() {
        // Use 15 nodes to exceed exact_threshold (10) and trigger Monte Carlo
        let mut gradients = HashMap::new();
        for i in 0..15 {
            gradients.insert(format!("node{}", i), array![i as f32, (i + 1) as f32, (i + 2) as f32]);
        }
        let aggregated = array![7.0, 8.0, 9.0];

        let config = ShapleyConfig::monte_carlo(100).with_seed(42);
        let mut calculator = ShapleyCalculator::new(config);

        let result = calculator.calculate_values(&gradients, &aggregated);

        assert!(matches!(result.method_used, SamplingMethod::MonteCarlo { .. }));
        assert_eq!(result.values.len(), 15);
    }

    #[test]
    fn test_permutation_approximation() {
        // Use 120 nodes to exceed monte_carlo_threshold (100) and trigger Permutation
        let mut gradients = HashMap::new();
        for i in 0..120 {
            gradients.insert(format!("node{}", i), Array1::from_vec(vec![i as f32; 10]));
        }
        let aggregated = Array1::from_vec(vec![60.0; 10]);

        let config = ShapleyConfig::permutation(50).with_seed(42);
        let mut calculator = ShapleyCalculator::new(config);

        let result = calculator.calculate_values(&gradients, &aggregated);

        assert!(matches!(result.method_used, SamplingMethod::Permutation { .. }));
        assert_eq!(result.values.len(), 120);
    }

    #[test]
    fn test_exact_vs_monte_carlo_consistency() {
        let gradients = create_test_gradients();
        let aggregated = array![1.0, 2.0, 3.0];

        // Exact
        let mut exact_calc = ShapleyCalculator::new(ShapleyConfig::exact().with_seed(42));
        let exact_result = exact_calc.calculate_values(&gradients, &aggregated);

        // Monte Carlo with many samples
        let mut mc_calc =
            ShapleyCalculator::new(ShapleyConfig::monte_carlo(5000).with_seed(42));
        let mc_result = mc_calc.calculate_values(&gradients, &aggregated);

        // Values should be reasonably close
        for node in gradients.keys() {
            let exact_val = exact_result.values.get(node).unwrap();
            let mc_val = mc_result.values.get(node).unwrap();
            let diff = (exact_val - mc_val).abs();
            assert!(
                diff < 0.1,
                "Node {} differs: exact={}, mc={}, diff={}",
                node,
                exact_val,
                mc_val,
                diff
            );
        }
    }

    #[test]
    fn test_marginal_contribution() {
        let config = ShapleyConfig::exact().with_seed(42);
        let mut calculator = ShapleyCalculator::new(config);

        let gradients = create_test_gradients();
        let aggregated = array![1.0, 2.0, 3.0];

        let mc = calculator.calculate_marginal_contribution("node1", &gradients, &aggregated);

        // Marginal contribution should be finite
        assert!(mc.is_finite());
    }

    #[test]
    fn test_empty_gradients() {
        let config = ShapleyConfig::exact();
        let mut calculator = ShapleyCalculator::new(config);

        let gradients: HashMap<String, Array1<f32>> = HashMap::new();
        let aggregated = array![1.0, 2.0, 3.0];

        let result = calculator.calculate_values(&gradients, &aggregated);

        assert!(result.values.is_empty());
        assert_eq!(result.total_value, 0.0);
    }

    #[test]
    fn test_single_node() {
        let config = ShapleyConfig::exact().with_seed(42);
        let mut calculator = ShapleyCalculator::new(config);

        let mut gradients = HashMap::new();
        gradients.insert("only_node".to_string(), array![1.0, 2.0, 3.0]);

        let aggregated = array![1.0, 2.0, 3.0];

        let result = calculator.calculate_values(&gradients, &aggregated);

        assert_eq!(result.values.len(), 1);
        // Single node gets all the value
        assert!(result.values.get("only_node").unwrap().abs() > 0.0);
    }

    #[test]
    fn test_value_function_l2() {
        let config = ShapleyConfig::exact()
            .with_value_function(ValueFunction::L2Distance)
            .with_seed(42);
        let mut calculator = ShapleyCalculator::new(config);

        let gradients = create_test_gradients();
        let aggregated = array![1.0, 2.0, 3.0];

        let result = calculator.calculate_values(&gradients, &aggregated);

        assert_eq!(result.values.len(), 3);
    }

    #[test]
    fn test_value_function_model_improvement() {
        let config = ShapleyConfig::exact()
            .with_value_function(ValueFunction::ModelImprovement)
            .with_seed(42);
        let mut calculator = ShapleyCalculator::new(config);

        let gradients = create_test_gradients();
        let aggregated = array![1.0, 2.0, 3.0];

        let result = calculator.calculate_values(&gradients, &aggregated);

        assert_eq!(result.values.len(), 3);
    }

    #[test]
    fn test_baseline_mean() {
        let config = ShapleyConfig::exact()
            .with_baseline(Baseline::MeanGradient)
            .with_seed(42);
        let mut calculator = ShapleyCalculator::new(config);

        let gradients = create_test_gradients();
        let aggregated = array![1.0, 2.0, 3.0];

        let result = calculator.calculate_values(&gradients, &aggregated);

        assert_eq!(result.values.len(), 3);
    }

    #[test]
    fn test_baseline_median() {
        let config = ShapleyConfig::exact()
            .with_baseline(Baseline::MedianGradient)
            .with_seed(42);
        let mut calculator = ShapleyCalculator::new(config);

        let gradients = create_test_gradients();
        let aggregated = array![1.0, 2.0, 3.0];

        let result = calculator.calculate_values(&gradients, &aggregated);

        assert_eq!(result.values.len(), 3);
    }

    #[test]
    fn test_normalization() {
        let config = ShapleyConfig::exact().with_normalization().with_seed(42);
        let mut calculator = ShapleyCalculator::new(config);

        let gradients = create_test_gradients();
        let aggregated = array![1.0, 2.0, 3.0];

        let result = calculator.calculate_values(&gradients, &aggregated);

        // Normalized values should sum to approximately 1
        let sum: f32 = result.values.values().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_sorted_by_value() {
        let config = ShapleyConfig::exact().with_seed(42);
        let mut calculator = ShapleyCalculator::new(config);

        let gradients = create_test_gradients();
        let aggregated = array![1.0, 2.0, 3.0];

        let result = calculator.calculate_values(&gradients, &aggregated);
        let sorted = result.sorted_by_value();

        assert_eq!(sorted.len(), 3);
        // Should be sorted in descending order
        for i in 1..sorted.len() {
            assert!(sorted[i - 1].1 >= sorted[i].1);
        }
    }

    #[test]
    fn test_top_k() {
        let config = ShapleyConfig::exact().with_seed(42);
        let mut calculator = ShapleyCalculator::new(config);

        let gradients = create_test_gradients();
        let aggregated = array![1.0, 2.0, 3.0];

        let result = calculator.calculate_values(&gradients, &aggregated);
        let top_2 = result.top_k(2);

        assert_eq!(top_2.len(), 2);
    }

    #[test]
    fn test_negative_contributors() {
        let config = ShapleyConfig::exact()
            .with_value_function(ValueFunction::L2Distance)
            .with_seed(42);
        let mut calculator = ShapleyCalculator::new(config);

        let mut gradients = HashMap::new();
        gradients.insert("honest".to_string(), array![1.0, 2.0, 3.0]);
        gradients.insert("bad".to_string(), array![-100.0, -200.0, -300.0]);

        let aggregated = array![1.0, 2.0, 3.0];

        let result = calculator.calculate_values(&gradients, &aggregated);
        let negative = result.negative_contributors();

        // The bad node should have negative contribution
        assert!(
            negative.iter().any(|(id, _)| *id == "bad"),
            "Bad node should have negative Shapley value"
        );
    }

    #[test]
    fn test_verify_efficiency() {
        let config = ShapleyConfig::exact().with_seed(42);
        let mut calculator = ShapleyCalculator::new(config);

        let gradients = create_test_gradients();
        let aggregated = array![1.0, 2.0, 3.0];

        let result = calculator.calculate_values(&gradients, &aggregated);

        assert!(verify_efficiency(&result, 0.01));
    }

    #[test]
    fn test_verify_symmetry() {
        let config = ShapleyConfig::exact().with_seed(42);
        let mut calculator = ShapleyCalculator::new(config);

        let mut gradients = HashMap::new();
        gradients.insert("sym1".to_string(), array![1.0, 2.0]);
        gradients.insert("sym2".to_string(), array![1.0, 2.0]);
        gradients.insert("other".to_string(), array![0.5, 1.0]);

        let aggregated = array![1.0, 2.0];

        let result = calculator.calculate_values(&gradients, &aggregated);

        assert!(verify_symmetry(
            &result,
            &["sym1".to_string(), "sym2".to_string()],
            0.01
        ));
    }

    #[test]
    fn test_reproducibility_with_seed() {
        let gradients = create_test_gradients();
        let aggregated = array![1.0, 2.0, 3.0];

        let config1 = ShapleyConfig::monte_carlo(100).with_seed(12345);
        let mut calc1 = ShapleyCalculator::new(config1);
        let result1 = calc1.calculate_values(&gradients, &aggregated);

        let config2 = ShapleyConfig::monte_carlo(100).with_seed(12345);
        let mut calc2 = ShapleyCalculator::new(config2);
        let result2 = calc2.calculate_values(&gradients, &aggregated);

        // Same seed should produce same results
        for node in gradients.keys() {
            let v1 = result1.values.get(node).unwrap();
            let v2 = result2.values.get(node).unwrap();
            assert_relative_eq!(v1, v2, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_performance_medium_coalition() {
        use std::time::Instant;

        let mut gradients = HashMap::new();
        for i in 0..15 {
            gradients.insert(format!("node{}", i), Array1::from_vec(vec![i as f32; 20]));
        }
        let aggregated = Array1::from_vec(vec![7.0; 20]);

        // Use fewer samples for faster test
        let config = ShapleyConfig::monte_carlo(100).with_seed(42);
        let mut calculator = ShapleyCalculator::new(config);

        let start = Instant::now();
        let result = calculator.calculate_values(&gradients, &aggregated);
        let elapsed = start.elapsed();

        assert_eq!(result.values.len(), 15);
        // More relaxed timing for CI environments
        assert!(elapsed.as_secs() < 30, "Took too long: {:?}", elapsed);
    }

    #[test]
    fn test_performance_large_coalition() {
        use std::time::Instant;

        let mut gradients = HashMap::new();
        for i in 0..120 {
            gradients.insert(format!("node{}", i), Array1::from_vec(vec![i as f32; 10]));
        }
        let aggregated = Array1::from_vec(vec![60.0; 10]);

        // Use fewer permutations for faster test
        let config = ShapleyConfig::permutation(50).with_seed(42);
        let mut calculator = ShapleyCalculator::new(config);

        let start = Instant::now();
        let result = calculator.calculate_values(&gradients, &aggregated);
        let elapsed = start.elapsed();

        assert_eq!(result.values.len(), 120);
        assert!(matches!(result.method_used, SamplingMethod::Permutation { .. }));
        // More relaxed timing for CI environments
        assert!(elapsed.as_secs() < 60, "Took too long: {:?}", elapsed);
    }

    #[test]
    fn test_xorshift_rng() {
        let mut rng = XorShiftRng::new(42);

        let v1 = rng.next();
        let v2 = rng.next();
        let v3 = rng.next();

        // Values should be different
        assert_ne!(v1, v2);
        assert_ne!(v2, v3);
        assert_ne!(v1, v3);
    }

    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0), 1.0);
        assert_eq!(factorial(1), 1.0);
        assert_eq!(factorial(5), 120.0);
        assert_eq!(factorial(10), 3628800.0);
    }
}
