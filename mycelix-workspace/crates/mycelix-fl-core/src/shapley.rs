// Ported from fl-aggregator/src/shapley.rs
//! Shapley Value Attribution for Fair Contribution Scoring in Federated Learning.
//!
//! Implements Shapley values for measuring individual node contributions to the
//! aggregated gradient. Uses `&[f32]` slices instead of ndarray for WASM compat.
//!
//! ## Properties
//!
//! - **Efficiency**: Values sum to total coalition value
//! - **Symmetry**: Identical contributions get identical values
//! - **Null player**: Non-contributors get zero value
//!
//! ## Computational Methods
//!
//! - **Exact**: O(2^n) — suitable for n <= 10
//! - **MonteCarlo**: O(m*n) where m = samples — suitable for 10 < n <= 100
//! - **Permutation**: O(p*n) where p = permutations — suitable for n > 100

use crate::aggregation::{euclidean_distance, l2_norm};
use crate::byzantine::cosine_similarity;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Sampling method for Shapley value computation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SamplingMethod {
    /// Exact computation — enumerate all 2^n subsets. Only for n <= 10.
    Exact,
    /// Monte Carlo sampling — randomly sample subsets.
    MonteCarlo { samples: usize },
    /// Permutation sampling — sample random orderings.
    Permutation { permutations: usize },
}

impl Default for SamplingMethod {
    fn default() -> Self {
        SamplingMethod::MonteCarlo { samples: 1000 }
    }
}

impl SamplingMethod {
    pub fn monte_carlo(samples: usize) -> Self {
        SamplingMethod::MonteCarlo { samples }
    }

    pub fn permutation(permutations: usize) -> Self {
        SamplingMethod::Permutation { permutations }
    }
}

/// Baseline computation method for value function.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Baseline {
    #[default]
    ZeroGradient,
    MeanGradient,
    MedianGradient,
}

/// Value function for measuring contribution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValueFunction {
    #[default]
    CosineSimilarity,
    L2Distance,
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
    /// N <= exact_threshold uses Exact.
    pub exact_threshold: usize,
    /// N > monte_carlo_threshold uses Permutation.
    pub monte_carlo_threshold: usize,
    /// Random seed for reproducibility.
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
    pub fn exact() -> Self {
        Self {
            sampling_method: SamplingMethod::Exact,
            ..Default::default()
        }
    }

    pub fn monte_carlo(samples: usize) -> Self {
        Self {
            sampling_method: SamplingMethod::MonteCarlo { samples },
            ..Default::default()
        }
    }

    pub fn permutation(permutations: usize) -> Self {
        Self {
            sampling_method: SamplingMethod::Permutation { permutations },
            ..Default::default()
        }
    }

    pub fn with_baseline(mut self, baseline: Baseline) -> Self {
        self.baseline = baseline;
        self
    }

    pub fn with_value_function(mut self, value_function: ValueFunction) -> Self {
        self.value_function = value_function;
        self
    }

    pub fn with_normalization(mut self) -> Self {
        self.normalize_values = true;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Auto-select method based on N.
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
                SamplingMethod::Permutation { permutations } => SamplingMethod::Permutation {
                    permutations: *permutations,
                },
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
    /// Sampling method actually used.
    pub method_used: SamplingMethod,
    /// Number of samples/permutations used.
    pub samples_used: usize,
    /// Efficiency check: |sum - total_coalition_value|.
    pub efficiency_error: f32,
}

impl ShapleyResult {
    pub fn get(&self, node_id: &str) -> Option<f32> {
        self.values.get(node_id).copied()
    }

    /// Nodes sorted by Shapley value (highest first).
    pub fn sorted_by_value(&self) -> Vec<(&String, f32)> {
        let mut sorted: Vec<_> = self.values.iter().map(|(k, v)| (k, *v)).collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted
    }

    /// Top K contributors.
    pub fn top_k(&self, k: usize) -> Vec<(&String, f32)> {
        self.sorted_by_value().into_iter().take(k).collect()
    }

    /// Nodes with negative Shapley values (potentially Byzantine).
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
    /// Create a new Shapley value calculator.
    pub fn new(config: ShapleyConfig) -> Self {
        let seed = config.seed.unwrap_or(42);
        Self {
            config,
            rng: XorShiftRng::new(seed),
        }
    }

    /// Calculate Shapley values for all nodes.
    ///
    /// # Arguments
    /// * `gradients` - Map of node IDs to their gradient contributions (as `Vec<f32>`)
    /// * `aggregated` - The aggregated (target) gradient
    pub fn calculate_values(
        &mut self,
        gradients: &HashMap<String, Vec<f32>>,
        aggregated: &[f32],
    ) -> ShapleyResult {
        let n = gradients.len();

        if n == 0 {
            return ShapleyResult {
                values: HashMap::new(),
                total_value: 0.0,
                method_used: self.config.sampling_method.clone(),
                samples_used: 0,
                efficiency_error: 0.0,
            };
        }

        let node_ids: Vec<&String> = gradients.keys().collect();
        let gradient_list: Vec<&Vec<f32>> = node_ids.iter().map(|id| &gradients[*id]).collect();
        let gradient_slices: Vec<&[f32]> = gradient_list.iter().map(|g| g.as_slice()).collect();

        let baseline = self.compute_baseline(&gradient_slices, aggregated);

        let method = self.config.auto_select_method(n);

        let (values, samples_used) = match &method {
            SamplingMethod::Exact => (
                self.compute_exact(&node_ids, &gradient_slices, aggregated, &baseline),
                1usize << n,
            ),
            SamplingMethod::MonteCarlo { samples } => (
                self.compute_monte_carlo(
                    &node_ids,
                    &gradient_slices,
                    aggregated,
                    &baseline,
                    *samples,
                ),
                *samples,
            ),
            SamplingMethod::Permutation { permutations } => (
                self.compute_permutation(
                    &node_ids,
                    &gradient_slices,
                    aggregated,
                    &baseline,
                    *permutations,
                ),
                *permutations,
            ),
        };

        let total_value = values.values().sum::<f32>();
        let full_coalition_value = self.coalition_value(&gradient_slices, aggregated, &baseline);
        let efficiency_error = (total_value - full_coalition_value).abs();

        let final_values = if self.config.normalize_values && total_value.abs() > 1e-10 {
            values
                .into_iter()
                .map(|(k, v)| (k, v / total_value))
                .collect()
        } else {
            values
        };

        ShapleyResult {
            values: final_values,
            total_value,
            method_used: method,
            samples_used,
            efficiency_error,
        }
    }

    /// Calculate marginal contribution of a single node.
    pub fn calculate_marginal_contribution(
        &mut self,
        node_id: &str,
        gradients: &HashMap<String, Vec<f32>>,
        aggregated: &[f32],
    ) -> f32 {
        let node_gradient = match gradients.get(node_id) {
            Some(g) => g,
            None => return 0.0,
        };

        let others: Vec<&[f32]> = gradients
            .iter()
            .filter(|(id, _)| id.as_str() != node_id)
            .map(|(_, g)| g.as_slice())
            .collect();

        if others.is_empty() {
            let all_grads: Vec<&[f32]> = vec![node_gradient.as_slice()];
            let baseline = self.compute_baseline(&all_grads, aggregated);
            return self.coalition_value(&all_grads, aggregated, &baseline);
        }

        let all_grads: Vec<&[f32]> = gradients.values().map(|g| g.as_slice()).collect();
        let baseline = self.compute_baseline(&all_grads, aggregated);

        let value_with = self.coalition_value(&all_grads, aggregated, &baseline);
        let value_without = self.coalition_value(&others, aggregated, &baseline);

        value_with - value_without
    }

    fn compute_baseline(&self, gradients: &[&[f32]], aggregated: &[f32]) -> Vec<f32> {
        let dim = aggregated.len();
        match self.config.baseline {
            Baseline::ZeroGradient => vec![0.0; dim],
            Baseline::MeanGradient => {
                if gradients.is_empty() {
                    return vec![0.0; dim];
                }
                let mut sum = vec![0.0f32; dim];
                for g in gradients {
                    for (i, v) in g.iter().enumerate() {
                        sum[i] += v;
                    }
                }
                let n = gradients.len() as f32;
                sum.iter_mut().for_each(|v| *v /= n);
                sum
            }
            Baseline::MedianGradient => {
                if gradients.is_empty() {
                    return vec![0.0; dim];
                }
                let mut result = vec![0.0f32; dim];
                let mut values: Vec<f32> = Vec::with_capacity(gradients.len());
                for d in 0..dim {
                    values.clear();
                    values.extend(gradients.iter().map(|g| g[d]));
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

    fn coalition_value(&self, coalition: &[&[f32]], aggregated: &[f32], baseline: &[f32]) -> f32 {
        if coalition.is_empty() {
            return 0.0;
        }

        let dim = aggregated.len();
        let mut coalition_agg = vec![0.0f32; dim];
        for g in coalition {
            for (i, v) in g.iter().enumerate() {
                coalition_agg[i] += v;
            }
        }
        let n = coalition.len() as f32;
        coalition_agg.iter_mut().for_each(|v| *v /= n);

        match self.config.value_function {
            ValueFunction::CosineSimilarity => {
                let sim = cosine_similarity(&coalition_agg, aggregated);
                (sim + 1.0) / 2.0
            }
            ValueFunction::L2Distance => {
                let baseline_dist = euclidean_distance(baseline, aggregated);
                let coalition_dist = euclidean_distance(&coalition_agg, aggregated);
                baseline_dist - coalition_dist
            }
            ValueFunction::ModelImprovement => {
                let dist = euclidean_distance(&coalition_agg, aggregated);
                let agg_norm = l2_norm(aggregated);
                if agg_norm < 1e-10 {
                    0.0
                } else {
                    1.0 - (dist / agg_norm).min(2.0) / 2.0
                }
            }
        }
    }

    fn compute_exact(
        &self,
        node_ids: &[&String],
        gradients: &[&[f32]],
        aggregated: &[f32],
        baseline: &[f32],
    ) -> HashMap<String, f32> {
        let n = node_ids.len();
        let mut values: HashMap<String, f32> =
            node_ids.iter().map(|id| ((*id).clone(), 0.0)).collect();

        let factorials: Vec<f64> = (0..=n).map(factorial).collect();

        for i in 0..n {
            let mut shapley_value = 0.0f32;

            for mask in 0..(1usize << (n - 1)) {
                let coalition_indices: Vec<usize> = (0..n)
                    .filter(|&j| j != i)
                    .enumerate()
                    .filter(|(bit_pos, _)| (mask >> bit_pos) & 1 == 1)
                    .map(|(_, j)| j)
                    .collect();

                let s = coalition_indices.len();

                let coalition: Vec<&[f32]> =
                    coalition_indices.iter().map(|&j| gradients[j]).collect();
                let v_s = self.coalition_value(&coalition, aggregated, baseline);

                let mut coalition_with_i = coalition.clone();
                coalition_with_i.push(gradients[i]);
                let v_s_with_i = self.coalition_value(&coalition_with_i, aggregated, baseline);

                let marginal = v_s_with_i - v_s;
                let weight = (factorials[s] * factorials[n - s - 1]) / factorials[n];

                shapley_value += weight as f32 * marginal;
            }

            values.insert(node_ids[i].clone(), shapley_value);
        }

        values
    }

    fn compute_monte_carlo(
        &mut self,
        node_ids: &[&String],
        gradients: &[&[f32]],
        aggregated: &[f32],
        baseline: &[f32],
        samples: usize,
    ) -> HashMap<String, f32> {
        let n = node_ids.len();
        let mut value_sums: Vec<f32> = vec![0.0; n];
        let mut counts: Vec<usize> = vec![0; n];

        for _ in 0..samples {
            let subset_mask: u64 = self.rng.next();

            for i in 0..n {
                let coalition_without: Vec<&[f32]> = (0..n)
                    .filter(|&j| j != i && ((subset_mask >> (j % 64)) & 1 == 1))
                    .map(|j| gradients[j])
                    .collect();

                let mut coalition_with = coalition_without.clone();
                coalition_with.push(gradients[i]);

                let v_without = self.coalition_value(&coalition_without, aggregated, baseline);
                let v_with = self.coalition_value(&coalition_with, aggregated, baseline);

                value_sums[i] += v_with - v_without;
                counts[i] += 1;
            }
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

    fn compute_permutation(
        &mut self,
        node_ids: &[&String],
        gradients: &[&[f32]],
        aggregated: &[f32],
        baseline: &[f32],
        permutations: usize,
    ) -> HashMap<String, f32> {
        let n = node_ids.len();
        let mut value_sums: Vec<f32> = vec![0.0; n];

        for _ in 0..permutations {
            let mut perm: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = (self.rng.next() as usize) % (i + 1);
                perm.swap(i, j);
            }

            let mut current_coalition: Vec<&[f32]> = Vec::with_capacity(n);
            let mut prev_value = 0.0;

            for &player_idx in &perm {
                current_coalition.push(gradients[player_idx]);
                let new_value = self.coalition_value(&current_coalition, aggregated, baseline);
                let marginal = new_value - prev_value;
                value_sums[player_idx] += marginal;
                prev_value = new_value;
            }
        }

        node_ids
            .iter()
            .enumerate()
            .map(|(i, id)| ((*id).clone(), value_sums[i] / permutations as f32))
            .collect()
    }
}

/// Simple XorShift64 RNG for reproducible sampling (WASM-safe, no std::time).
struct XorShiftRng {
    state: u64,
}

impl XorShiftRng {
    fn new(seed: u64) -> Self {
        // Avoid seed-0 fixed point
        let state = if seed == 0 {
            seed ^ 0x9E3779B97F4A7C15
        } else {
            seed
        };
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

/// Verify Shapley value efficiency property.
pub fn verify_efficiency(result: &ShapleyResult, tolerance: f32) -> bool {
    result.efficiency_error <= tolerance
}

/// Check if symmetric players have equal values.
pub fn verify_symmetry(result: &ShapleyResult, symmetric_nodes: &[String], tolerance: f32) -> bool {
    if symmetric_nodes.len() < 2 {
        return true;
    }

    let first_value = match result.values.get(&symmetric_nodes[0]) {
        Some(v) => *v,
        None => return false,
    };

    symmetric_nodes
        .iter()
        .skip(1)
        .all(|node| match result.values.get(node) {
            Some(v) => (v - first_value).abs() <= tolerance,
            None => false,
        })
}

/// Check if null players get zero value.
pub fn verify_null_player(result: &ShapleyResult, null_nodes: &[String], tolerance: f32) -> bool {
    null_nodes.iter().all(|node| match result.values.get(node) {
        Some(v) => v.abs() <= tolerance,
        None => true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_gradients() -> HashMap<String, Vec<f32>> {
        let mut gradients = HashMap::new();
        gradients.insert("node1".to_string(), vec![1.0, 2.0, 3.0]);
        gradients.insert("node2".to_string(), vec![1.1, 2.1, 3.1]);
        gradients.insert("node3".to_string(), vec![0.9, 1.9, 2.9]);
        gradients
    }

    #[test]
    fn test_shapley_config_default() {
        let config = ShapleyConfig::default();
        assert!(matches!(
            config.sampling_method,
            SamplingMethod::MonteCarlo { .. }
        ));
        assert_eq!(config.baseline, Baseline::ZeroGradient);
        assert_eq!(config.value_function, ValueFunction::CosineSimilarity);
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
        let aggregated = vec![1.0, 2.0, 3.0];

        let result = calculator.calculate_values(&gradients, &aggregated);

        assert_eq!(result.values.len(), 3);
        assert_eq!(result.method_used, SamplingMethod::Exact);
    }

    #[test]
    fn test_efficiency_property() {
        let config = ShapleyConfig::exact().with_seed(42);
        let mut calculator = ShapleyCalculator::new(config);

        let gradients = create_test_gradients();
        let aggregated = vec![1.0, 2.0, 3.0];

        let result = calculator.calculate_values(&gradients, &aggregated);

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

        let mut gradients = HashMap::new();
        gradients.insert("node1".to_string(), vec![1.0, 0.0]);
        gradients.insert("node2".to_string(), vec![1.0, 0.0]);

        let aggregated = vec![1.0, 0.0];

        let result = calculator.calculate_values(&gradients, &aggregated);

        let v1 = result.values.get("node1").unwrap();
        let v2 = result.values.get("node2").unwrap();
        assert!(
            (v1 - v2).abs() < 0.01,
            "Symmetric nodes should get equal values"
        );
    }

    #[test]
    fn test_byzantine_gradient_low_value() {
        let config = ShapleyConfig::exact()
            .with_value_function(ValueFunction::L2Distance)
            .with_seed(42);
        let mut calculator = ShapleyCalculator::new(config);

        let mut gradients = HashMap::new();
        gradients.insert("honest1".to_string(), vec![1.0, 2.0, 3.0]);
        gradients.insert("honest2".to_string(), vec![1.1, 2.1, 3.1]);
        gradients.insert("honest3".to_string(), vec![0.9, 1.9, 2.9]);
        gradients.insert("byzantine".to_string(), vec![-10.0, -20.0, -30.0]);

        let aggregated = vec![1.0, 2.0, 3.0];

        let result = calculator.calculate_values(&gradients, &aggregated);

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
        let mut gradients = HashMap::new();
        for i in 0..15 {
            gradients.insert(
                format!("node{}", i),
                vec![i as f32, (i + 1) as f32, (i + 2) as f32],
            );
        }
        let aggregated = vec![7.0, 8.0, 9.0];

        let config = ShapleyConfig::monte_carlo(100).with_seed(42);
        let mut calculator = ShapleyCalculator::new(config);

        let result = calculator.calculate_values(&gradients, &aggregated);

        assert!(matches!(
            result.method_used,
            SamplingMethod::MonteCarlo { .. }
        ));
        assert_eq!(result.values.len(), 15);
    }

    #[test]
    fn test_permutation_approximation() {
        let mut gradients = HashMap::new();
        for i in 0..120 {
            gradients.insert(format!("node{}", i), vec![i as f32; 10]);
        }
        let aggregated = vec![60.0; 10];

        let config = ShapleyConfig::permutation(50).with_seed(42);
        let mut calculator = ShapleyCalculator::new(config);

        let result = calculator.calculate_values(&gradients, &aggregated);

        assert!(matches!(
            result.method_used,
            SamplingMethod::Permutation { .. }
        ));
        assert_eq!(result.values.len(), 120);
    }

    #[test]
    fn test_exact_vs_monte_carlo_consistency() {
        let gradients = create_test_gradients();
        let aggregated = vec![1.0, 2.0, 3.0];

        let mut exact_calc = ShapleyCalculator::new(ShapleyConfig::exact().with_seed(42));
        let exact_result = exact_calc.calculate_values(&gradients, &aggregated);

        let mut mc_calc = ShapleyCalculator::new(ShapleyConfig::monte_carlo(5000).with_seed(42));
        let mc_result = mc_calc.calculate_values(&gradients, &aggregated);

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
        let aggregated = vec![1.0, 2.0, 3.0];

        let mc = calculator.calculate_marginal_contribution("node1", &gradients, &aggregated);
        assert!(mc.is_finite());
    }

    #[test]
    fn test_empty_gradients() {
        let config = ShapleyConfig::exact();
        let mut calculator = ShapleyCalculator::new(config);

        let gradients: HashMap<String, Vec<f32>> = HashMap::new();
        let aggregated = vec![1.0, 2.0, 3.0];

        let result = calculator.calculate_values(&gradients, &aggregated);
        assert!(result.values.is_empty());
        assert_eq!(result.total_value, 0.0);
    }

    #[test]
    fn test_single_node() {
        let config = ShapleyConfig::exact().with_seed(42);
        let mut calculator = ShapleyCalculator::new(config);

        let mut gradients = HashMap::new();
        gradients.insert("only_node".to_string(), vec![1.0, 2.0, 3.0]);

        let aggregated = vec![1.0, 2.0, 3.0];

        let result = calculator.calculate_values(&gradients, &aggregated);

        assert_eq!(result.values.len(), 1);
        assert!(result.values.get("only_node").unwrap().abs() > 0.0);
    }

    #[test]
    fn test_value_function_l2() {
        let config = ShapleyConfig::exact()
            .with_value_function(ValueFunction::L2Distance)
            .with_seed(42);
        let mut calculator = ShapleyCalculator::new(config);

        let gradients = create_test_gradients();
        let aggregated = vec![1.0, 2.0, 3.0];

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
        let aggregated = vec![1.0, 2.0, 3.0];

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
        let aggregated = vec![1.0, 2.0, 3.0];

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
        let aggregated = vec![1.0, 2.0, 3.0];

        let result = calculator.calculate_values(&gradients, &aggregated);
        assert_eq!(result.values.len(), 3);
    }

    #[test]
    fn test_normalization() {
        let config = ShapleyConfig::exact().with_normalization().with_seed(42);
        let mut calculator = ShapleyCalculator::new(config);

        let gradients = create_test_gradients();
        let aggregated = vec![1.0, 2.0, 3.0];

        let result = calculator.calculate_values(&gradients, &aggregated);
        let sum: f32 = result.values.values().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Normalized values should sum to ~1.0, got {}",
            sum
        );
    }

    #[test]
    fn test_sorted_by_value() {
        let config = ShapleyConfig::exact().with_seed(42);
        let mut calculator = ShapleyCalculator::new(config);

        let gradients = create_test_gradients();
        let aggregated = vec![1.0, 2.0, 3.0];

        let result = calculator.calculate_values(&gradients, &aggregated);
        let sorted = result.sorted_by_value();

        assert_eq!(sorted.len(), 3);
        for i in 1..sorted.len() {
            assert!(sorted[i - 1].1 >= sorted[i].1);
        }
    }

    #[test]
    fn test_top_k() {
        let config = ShapleyConfig::exact().with_seed(42);
        let mut calculator = ShapleyCalculator::new(config);

        let gradients = create_test_gradients();
        let aggregated = vec![1.0, 2.0, 3.0];

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
        gradients.insert("honest".to_string(), vec![1.0, 2.0, 3.0]);
        gradients.insert("bad".to_string(), vec![-100.0, -200.0, -300.0]);

        let aggregated = vec![1.0, 2.0, 3.0];

        let result = calculator.calculate_values(&gradients, &aggregated);
        let negative = result.negative_contributors();
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
        let aggregated = vec![1.0, 2.0, 3.0];

        let result = calculator.calculate_values(&gradients, &aggregated);
        assert!(verify_efficiency(&result, 0.01));
    }

    #[test]
    fn test_verify_symmetry() {
        let config = ShapleyConfig::exact().with_seed(42);
        let mut calculator = ShapleyCalculator::new(config);

        let mut gradients = HashMap::new();
        gradients.insert("sym1".to_string(), vec![1.0, 2.0]);
        gradients.insert("sym2".to_string(), vec![1.0, 2.0]);
        gradients.insert("other".to_string(), vec![0.5, 1.0]);

        let aggregated = vec![1.0, 2.0];

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
        let aggregated = vec![1.0, 2.0, 3.0];

        let config1 = ShapleyConfig::monte_carlo(100).with_seed(12345);
        let mut calc1 = ShapleyCalculator::new(config1);
        let result1 = calc1.calculate_values(&gradients, &aggregated);

        let config2 = ShapleyConfig::monte_carlo(100).with_seed(12345);
        let mut calc2 = ShapleyCalculator::new(config2);
        let result2 = calc2.calculate_values(&gradients, &aggregated);

        for node in gradients.keys() {
            let v1 = result1.values.get(node).unwrap();
            let v2 = result2.values.get(node).unwrap();
            assert!(
                (v1 - v2).abs() < 1e-6,
                "Same seed should produce same results"
            );
        }
    }

    #[test]
    fn test_xorshift_rng() {
        let mut rng = XorShiftRng::new(42);
        let v1 = rng.next();
        let v2 = rng.next();
        let v3 = rng.next();
        assert_ne!(v1, v2);
        assert_ne!(v2, v3);
        assert_ne!(v1, v3);
    }

    #[test]
    fn test_xorshift_seed_zero() {
        // Seed 0 should not be a fixed point
        let mut rng = XorShiftRng::new(0);
        let v1 = rng.next();
        let v2 = rng.next();
        assert_ne!(v1, 0);
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0), 1.0);
        assert_eq!(factorial(1), 1.0);
        assert_eq!(factorial(5), 120.0);
        assert_eq!(factorial(10), 3628800.0);
    }
}
