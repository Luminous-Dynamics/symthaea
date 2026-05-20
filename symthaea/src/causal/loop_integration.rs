// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Causal Loop Integration
//!
//! Integrates causal discovery into the cognitive loop to enhance
//! attention weighting and guide exploration.
//!
//! ## Overview
//!
//! The `CausalLoopEnhancer` tracks recent (input, output) pairs from the
//! cognitive loop, periodically runs causal discovery on accumulated pairs,
//! and uses discovered causal structure to:
//!
//! - Weight attention: Causal parents get more attention weight
//! - Guide exploration: Intervene on discovered causes to explore effects
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::causal::CausalLoopEnhancer;
//! use symthaea_core::hdc::ContinuousHV;
//!
//! let mut enhancer = CausalLoopEnhancer::new(42);
//!
//! // Each cognitive cycle, record the (input, output) pair
//! enhancer.record_cycle(&input_hv, &output_hv);
//!
//! // Periodically discover causal structure (e.g., every 100 cycles)
//! if enhancer.should_discover() {
//!     let graph = enhancer.run_discovery();
//!     println!("Discovered {} edges", graph.edges.len());
//! }
//!
//! // Use causal structure to weight attention
//! let attention_weights = enhancer.causal_attention_weights(feature_idx);
//! ```

use crate::intelligence::{CausalAttention, CausalDirection, CausalDiscoveryEngine};
use std::collections::{HashMap, VecDeque};
use symthaea_core::hdc::ContinuousHV;
use symthaea_core::hdc::HDC_DIMENSION;

/// Configuration for the causal loop enhancer
#[derive(Debug, Clone)]
pub struct CausalEnhancerConfig {
    /// Number of dimensions to subsample from HV for causal analysis
    /// (full 16384 dims would be too slow)
    pub subsample_dims: usize,

    /// Maximum number of (input, output) pairs to store
    pub max_history: usize,

    /// Minimum pairs needed before running causal discovery
    pub min_pairs_for_discovery: usize,

    /// Number of cycles between automatic discovery runs
    pub discovery_interval: usize,

    /// Random seed for reproducibility
    pub seed: u64,

    /// Attention boost factor for causal parents (1.0 = no boost)
    pub causal_attention_boost: f32,

    /// Exploration probability for intervention-based exploration
    pub exploration_probability: f32,
}

impl Default for CausalEnhancerConfig {
    fn default() -> Self {
        Self {
            subsample_dims: 64, // Subsample to 64 dims for tractable causal discovery
            max_history: 1000,
            min_pairs_for_discovery: 50,
            discovery_interval: 100,
            seed: 42,
            causal_attention_boost: 1.5,
            exploration_probability: 0.1,
        }
    }
}

/// A recorded (input, output) pair from the cognitive loop
#[derive(Debug, Clone)]
struct CyclePair {
    /// Subsampled input hypervector
    input: Vec<f64>,
    /// Subsampled output hypervector
    output: Vec<f64>,
    /// Cycle number when recorded (reserved for future use)
    #[allow(dead_code)]
    cycle: usize,
}

/// An edge in the discovered causal graph
#[derive(Debug, Clone)]
pub struct CausalGraphEdge {
    /// Source feature index
    pub from: usize,
    /// Target feature index
    pub to: usize,
    /// Causal direction (Forward: from->to causes, Backward: to->from causes)
    pub direction: CausalDirection,
    /// Confidence in this edge (0.0-1.0)
    pub confidence: f64,
    /// Strength of causal relationship (0.0-1.0)
    pub strength: f64,
}

/// Discovered causal structure over subsampled HV dimensions
#[derive(Debug, Clone, Default)]
pub struct CausalGraph {
    /// Number of variables (subsampled dimensions)
    pub n_vars: usize,
    /// Directed edges in the graph
    pub edges: Vec<CausalGraphEdge>,
    /// Parents for each variable index
    pub parents: HashMap<usize, Vec<(usize, f64)>>,
    /// Children for each variable index
    pub children: HashMap<usize, Vec<(usize, f64)>>,
    /// Cycle at which this graph was discovered
    pub discovered_at_cycle: usize,
}

impl CausalGraph {
    /// Check if this graph is empty (no edges)
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    /// Get the strongest cause of a variable
    pub fn strongest_cause(&self, var: usize) -> Option<(usize, f64)> {
        self.parents
            .get(&var)
            .and_then(|parents| parents.first().cloned())
    }

    /// Get the strongest effect of a variable
    pub fn strongest_effect(&self, var: usize) -> Option<(usize, f64)> {
        self.children
            .get(&var)
            .and_then(|children| children.first().cloned())
    }

    /// Convert to human-readable format
    pub fn summarize(&self) -> String {
        let mut summary = format!(
            "CausalGraph: {} vars, {} edges\n",
            self.n_vars,
            self.edges.len()
        );
        for edge in &self.edges {
            let dir_str = match edge.direction {
                CausalDirection::Forward => "->",
                CausalDirection::Backward => "<-",
            };
            summary.push_str(&format!(
                "  dim{} {} dim{} (conf={:.2}, str={:.2})\n",
                edge.from, dir_str, edge.to, edge.confidence, edge.strength
            ));
        }
        summary
    }
}

/// A discovered causal relationship for logging
#[derive(Debug, Clone)]
pub struct DiscoveredRelationship {
    /// Cycle at which discovered
    pub cycle: usize,
    /// Source dimension
    pub cause_dim: usize,
    /// Target dimension
    pub effect_dim: usize,
    /// Confidence of relationship
    pub confidence: f64,
    /// Strength of relationship
    pub strength: f64,
}

/// Statistics about causal discovery
#[derive(Debug, Clone, Default)]
pub struct CausalLoopStats {
    /// Total cycles processed
    pub total_cycles: usize,
    /// Number of discovery runs
    pub discovery_runs: usize,
    /// Number of edges discovered total
    pub total_edges_discovered: usize,
    /// Average confidence of discovered edges
    pub avg_confidence: f64,
    /// Number of times causal attention was used
    pub attention_uses: usize,
    /// Number of exploration interventions
    pub interventions: usize,
}

/// The Causal Loop Enhancer
///
/// Integrates causal discovery into the cognitive loop by:
/// 1. Tracking recent (input, output) pairs
/// 2. Periodically running causal discovery
/// 3. Using discovered structure to weight attention and guide exploration
pub struct CausalLoopEnhancer {
    /// Configuration
    config: CausalEnhancerConfig,

    /// History of (input, output) pairs
    history: VecDeque<CyclePair>,

    /// Current cycle count
    current_cycle: usize,

    /// Causal discovery engine
    discovery_engine: CausalDiscoveryEngine,

    /// Causal attention mechanism (reserved for future graph-guided attention)
    #[allow(dead_code)]
    causal_attention: CausalAttention,

    /// Most recently discovered causal graph
    current_graph: CausalGraph,

    /// History of discovered relationships (for logging)
    discovered_relationships: Vec<DiscoveredRelationship>,

    /// Indices used for subsampling
    subsample_indices: Vec<usize>,

    /// Statistics
    stats: CausalLoopStats,
}

impl CausalLoopEnhancer {
    /// Create a new causal loop enhancer with default config
    pub fn new(seed: u64) -> Self {
        let config = CausalEnhancerConfig {
            seed,
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Create with custom configuration
    pub fn with_config(config: CausalEnhancerConfig) -> Self {
        let subsample_indices =
            Self::generate_subsample_indices(HDC_DIMENSION, config.subsample_dims, config.seed);

        Self {
            discovery_engine: CausalDiscoveryEngine::with_ensemble_size(config.seed, 5),
            causal_attention: CausalAttention::new(config.seed),
            history: VecDeque::with_capacity(config.max_history),
            current_cycle: 0,
            current_graph: CausalGraph::default(),
            discovered_relationships: Vec::new(),
            subsample_indices,
            stats: CausalLoopStats::default(),
            config,
        }
    }

    /// Generate deterministic subsample indices
    fn generate_subsample_indices(total_dim: usize, subsample_dim: usize, seed: u64) -> Vec<usize> {
        use rand::{SeedableRng, seq::SliceRandom};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut indices: Vec<usize> = (0..total_dim).collect();
        indices.shuffle(&mut rng);
        indices.truncate(subsample_dim);
        indices.sort();
        indices
    }

    /// Subsample an HV to the configured number of dimensions
    fn subsample_hv(&self, hv: &ContinuousHV) -> Vec<f64> {
        self.subsample_indices
            .iter()
            .map(|&i| hv.values.get(i).copied().unwrap_or(0.0) as f64)
            .collect()
    }

    /// Record a cognitive cycle (input, output) pair
    ///
    /// This is called every cognitive loop cycle to accumulate
    /// data for causal discovery.
    pub fn record_cycle(&mut self, input: &ContinuousHV, output: &ContinuousHV) {
        self.current_cycle += 1;
        self.stats.total_cycles += 1;

        let pair = CyclePair {
            input: self.subsample_hv(input),
            output: self.subsample_hv(output),
            cycle: self.current_cycle,
        };

        // Maintain max history
        if self.history.len() >= self.config.max_history {
            self.history.pop_front();
        }
        self.history.push_back(pair);
    }

    /// Record from f32 vectors (common in cognitive loop)
    pub fn record_cycle_from_f32(&mut self, input: &[f32], output: &[f32]) {
        self.current_cycle += 1;
        self.stats.total_cycles += 1;

        // Subsample from input/output
        let subsample_from = |src: &[f32], indices: &[usize]| -> Vec<f64> {
            indices
                .iter()
                .map(|&i| src.get(i).copied().unwrap_or(0.0) as f64)
                .collect()
        };

        let pair = CyclePair {
            input: subsample_from(input, &self.subsample_indices),
            output: subsample_from(output, &self.subsample_indices),
            cycle: self.current_cycle,
        };

        if self.history.len() >= self.config.max_history {
            self.history.pop_front();
        }
        self.history.push_back(pair);
    }

    /// Check if it's time to run causal discovery
    pub fn should_discover(&self) -> bool {
        self.history.len() >= self.config.min_pairs_for_discovery
            && self
                .current_cycle
                .is_multiple_of(self.config.discovery_interval)
    }

    /// Detect causal structure from accumulated history
    ///
    /// This is the core discovery function. It:
    /// 1. Extracts variables (each subsampled dimension as a time series)
    /// 2. Runs bivariate causal discovery between input and output dimensions
    /// 3. Builds a causal graph
    pub fn detect_causal_structure(
        &mut self,
        history: &[(ContinuousHV, ContinuousHV)],
    ) -> CausalGraph {
        // Convert to internal format
        let pairs: Vec<CyclePair> = history
            .iter()
            .enumerate()
            .map(|(i, (input, output))| CyclePair {
                input: self.subsample_hv(input),
                output: self.subsample_hv(output),
                cycle: i,
            })
            .collect();

        self.detect_causal_structure_internal(&pairs)
    }

    /// Run causal discovery on accumulated history
    pub fn run_discovery(&mut self) -> CausalGraph {
        let history: Vec<CyclePair> = self.history.iter().cloned().collect();
        let graph = self.detect_causal_structure_internal(&history);
        self.current_graph = graph.clone();
        graph
    }

    /// Internal causal structure detection
    fn detect_causal_structure_internal(&mut self, history: &[CyclePair]) -> CausalGraph {
        self.stats.discovery_runs += 1;

        let n_pairs = history.len();
        if n_pairs < self.config.min_pairs_for_discovery {
            return CausalGraph::default();
        }

        let n_vars = self.config.subsample_dims;
        let mut edges = Vec::new();
        let mut parents: HashMap<usize, Vec<(usize, f64)>> = HashMap::new();
        let mut children: HashMap<usize, Vec<(usize, f64)>> = HashMap::new();

        // Extract time series for each dimension
        // We look at input[t] -> output[t] relationships
        let mut input_series: Vec<Vec<f64>> = vec![Vec::with_capacity(n_pairs); n_vars];
        let mut output_series: Vec<Vec<f64>> = vec![Vec::with_capacity(n_pairs); n_vars];

        for pair in history {
            for (i, &val) in pair.input.iter().enumerate() {
                if i < n_vars {
                    input_series[i].push(val);
                }
            }
            for (i, &val) in pair.output.iter().enumerate() {
                if i < n_vars {
                    output_series[i].push(val);
                }
            }
        }

        // Discover causal relationships: input_i -> output_j
        // Focus on cross-dimensional relationships (input affects output)
        for i in 0..n_vars {
            for j in 0..n_vars {
                if input_series[i].len() < 20 || output_series[j].len() < 20 {
                    continue;
                }

                let (direction, confidence) = self
                    .discovery_engine
                    .predict_with_confidence(&input_series[i], &output_series[j]);

                // Only record significant causal relationships
                if confidence > 0.3 {
                    // Compute strength using correlation as proxy
                    let strength = self
                        .compute_correlation(&input_series[i], &output_series[j])
                        .abs()
                        .min(1.0);

                    if strength > 0.1 {
                        let edge = CausalGraphEdge {
                            from: i,
                            to: j,
                            direction,
                            confidence,
                            strength,
                        };

                        // Record relationship for logging
                        self.discovered_relationships.push(DiscoveredRelationship {
                            cycle: self.current_cycle,
                            cause_dim: i,
                            effect_dim: j,
                            confidence,
                            strength,
                        });

                        // Update parent/child maps based on direction
                        match direction {
                            CausalDirection::Forward => {
                                // i causes j
                                parents
                                    .entry(j)
                                    .or_default()
                                    .push((i, strength * confidence));
                                children
                                    .entry(i)
                                    .or_default()
                                    .push((j, strength * confidence));
                            }
                            CausalDirection::Backward => {
                                // j causes i (reverse)
                                parents
                                    .entry(i)
                                    .or_default()
                                    .push((j, strength * confidence));
                                children
                                    .entry(j)
                                    .or_default()
                                    .push((i, strength * confidence));
                            }
                        }

                        edges.push(edge);
                    }
                }
            }
        }

        // Sort parents/children by strength
        for parents_list in parents.values_mut() {
            parents_list.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }
        for children_list in children.values_mut() {
            children_list
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }

        self.stats.total_edges_discovered += edges.len();
        if !edges.is_empty() {
            self.stats.avg_confidence =
                edges.iter().map(|e| e.confidence).sum::<f64>() / edges.len() as f64;
        }

        CausalGraph {
            n_vars,
            edges,
            parents,
            children,
            discovered_at_cycle: self.current_cycle,
        }
    }

    /// Compute correlation between two series
    fn compute_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len().min(y.len());
        if n < 2 {
            return 0.0;
        }

        let mean_x: f64 = x.iter().sum::<f64>() / n as f64;
        let mean_y: f64 = y.iter().sum::<f64>() / n as f64;

        let var_x: f64 = x.iter().map(|&xi| (xi - mean_x).powi(2)).sum::<f64>() / (n - 1) as f64;
        let var_y: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum::<f64>() / (n - 1) as f64;

        if var_x < 1e-10 || var_y < 1e-10 {
            return 0.0;
        }

        let cov: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
            .sum::<f64>()
            / (n - 1) as f64;

        cov / (var_x.sqrt() * var_y.sqrt())
    }

    /// Get attention weights based on causal structure
    ///
    /// Returns weights for each subsampled dimension, with causal parents
    /// receiving higher weights.
    pub fn causal_attention_weights(&mut self, target_dim: usize) -> Vec<f32> {
        self.stats.attention_uses += 1;

        let n = self.config.subsample_dims;
        let mut weights = vec![1.0f32; n];

        // Boost weights for causal parents
        if let Some(parents) = self.current_graph.parents.get(&target_dim) {
            for &(parent_idx, strength) in parents {
                if parent_idx < n {
                    // Boost proportional to causal strength and config
                    weights[parent_idx] *=
                        self.config.causal_attention_boost * (1.0 + strength as f32);
                }
            }
        }

        // Normalize weights
        let sum: f32 = weights.iter().sum();
        if sum > 0.0 {
            for w in &mut weights {
                *w /= sum;
            }
        }

        weights
    }

    /// Suggest an intervention for exploration
    ///
    /// Based on discovered causal structure, suggests which dimension
    /// to intervene on to explore causal effects.
    pub fn suggest_intervention(&mut self) -> Option<(usize, f64)> {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(
            self.config.seed.wrapping_add(self.current_cycle as u64),
        );

        // Only intervene with configured probability
        if rng.r#gen::<f32>() > self.config.exploration_probability {
            return None;
        }

        // Find the variable with most causal influence (most children)
        let mut best_var = None;
        let mut max_influence = 0.0;

        for (&var, children) in &self.current_graph.children {
            let influence: f64 = children.iter().map(|(_, s)| s).sum();
            if influence > max_influence {
                max_influence = influence;
                best_var = Some(var);
            }
        }

        if let Some(var) = best_var {
            self.stats.interventions += 1;
            // Suggest a random intervention value
            let intervention_value: f64 = rng.gen_range(-1.0..1.0);
            Some((var, intervention_value))
        } else {
            None
        }
    }

    /// Apply causal weights to a full attention tensor
    ///
    /// Maps the subsampled causal attention back to full HDC dimensions.
    pub fn apply_to_attention(&self, attention: &mut [f32], target_dim: usize) {
        // Get causal weights for subsampled space
        let n = self.config.subsample_dims;
        let mut causal_weights = vec![1.0f32; n];

        if let Some(parents) = self.current_graph.parents.get(&target_dim) {
            for &(parent_idx, strength) in parents {
                if parent_idx < n {
                    causal_weights[parent_idx] *=
                        self.config.causal_attention_boost * (1.0 + strength as f32);
                }
            }
        }

        // Apply to full attention vector using subsample indices
        for (subsampled_idx, &full_idx) in self.subsample_indices.iter().enumerate() {
            if full_idx < attention.len() {
                attention[full_idx] *= causal_weights[subsampled_idx];
            }
        }
    }

    /// Get the current causal graph
    pub fn current_graph(&self) -> &CausalGraph {
        &self.current_graph
    }

    /// Get discovered relationships history
    pub fn discovered_relationships(&self) -> &[DiscoveredRelationship] {
        &self.discovered_relationships
    }

    /// Get statistics
    pub fn stats(&self) -> &CausalLoopStats {
        &self.stats
    }

    /// Get number of stored pairs
    pub fn history_size(&self) -> usize {
        self.history.len()
    }

    /// Clear history and reset
    pub fn reset(&mut self) {
        self.history.clear();
        self.current_cycle = 0;
        self.current_graph = CausalGraph::default();
        self.discovered_relationships.clear();
        self.stats = CausalLoopStats::default();
    }

    /// Check if causal enhancement is enabled and has discovered structure
    pub fn has_causal_structure(&self) -> bool {
        !self.current_graph.is_empty()
    }

    /// Log discovered causal relationships
    pub fn log_discoveries(&self) {
        if self.current_graph.is_empty() {
            tracing::debug!("Causal enhancer: no structure discovered yet");
            return;
        }

        tracing::info!(
            edges = self.current_graph.edges.len(),
            cycle = self.current_graph.discovered_at_cycle,
            "Causal structure discovered"
        );

        for edge in &self.current_graph.edges {
            let dir_str = match edge.direction {
                CausalDirection::Forward => "causes",
                CausalDirection::Backward => "is caused by",
            };
            tracing::debug!(
                from = edge.from,
                to = edge.to,
                confidence = format!("{:.2}", edge.confidence),
                strength = format!("{:.2}", edge.strength),
                "dim{} {} dim{}",
                edge.from,
                dir_str,
                edge.to
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_causal_structure_discovery_on_synthetic_data() {
        let mut enhancer = CausalLoopEnhancer::new(42);

        // Generate synthetic data with broad causal relationships.
        // The enhancer subsamples 64 random dims from HDC_DIMENSION (16384),
        // so we must place causal signal across MANY dimensions to ensure
        // the subsampled dims capture the relationships.
        let n_samples = 200;
        let mut rng = rand::rngs::StdRng::seed_from_u64(12345);

        for _ in 0..n_samples {
            let mut input = vec![0.0f32; HDC_DIMENSION];
            let mut output = vec![0.0f32; HDC_DIMENSION];

            // Place strong causal relationships across many dimensions
            // so that subsampled indices will inevitably catch some.
            // input[i] causes output[i] for all i in 0..HDC_DIMENSION
            let cause_value: f32 = rng.gen_range(-1.0..1.0);
            for i in 0..HDC_DIMENSION {
                input[i] = cause_value + rng.gen_range(-0.05..0.05);
                output[i] = 0.8 * input[i] + rng.gen_range(-0.1..0.1);
            }

            enhancer.record_cycle_from_f32(&input, &output);
        }

        // Run discovery
        assert!(enhancer.should_discover() || enhancer.history_size() >= 50);
        let graph = enhancer.run_discovery();

        // Verify some structure was discovered
        assert!(
            !graph.edges.is_empty(),
            "Should discover at least one edge with synthetic causal data"
        );

        println!("Discovered graph:\n{}", graph.summarize());
        println!("Stats: {:?}", enhancer.stats());
    }

    #[test]
    fn test_causal_attention_weights() {
        let mut enhancer = CausalLoopEnhancer::new(42);

        // Manually set up a causal graph for testing
        enhancer.current_graph = CausalGraph {
            n_vars: 64,
            edges: vec![CausalGraphEdge {
                from: 0,
                to: 1,
                direction: CausalDirection::Forward,
                confidence: 0.8,
                strength: 0.7,
            }],
            parents: {
                let mut p = HashMap::new();
                p.insert(1, vec![(0, 0.56)]); // 0 is parent of 1
                p
            },
            children: {
                let mut c = HashMap::new();
                c.insert(0, vec![(1, 0.56)]); // 1 is child of 0
                c
            },
            discovered_at_cycle: 100,
        };

        // Get attention weights for dim 1 (which has dim 0 as parent)
        let weights = enhancer.causal_attention_weights(1);

        // The parent (dim 0) should have higher weight
        assert!(
            weights[0] > 1.0 / 64.0,
            "Parent dimension should have boosted weight: {} > {}",
            weights[0],
            1.0 / 64.0
        );
    }

    #[test]
    fn test_causal_knowledge_improves_predictions() {
        // This test verifies that having causal knowledge changes attention
        let mut enhancer_with_causal = CausalLoopEnhancer::new(42);
        let mut enhancer_without_causal = CausalLoopEnhancer::new(42);

        // Set up causal structure
        enhancer_with_causal.current_graph = CausalGraph {
            n_vars: 64,
            edges: vec![CausalGraphEdge {
                from: 5,
                to: 10,
                direction: CausalDirection::Forward,
                confidence: 0.9,
                strength: 0.8,
            }],
            parents: {
                let mut p = HashMap::new();
                p.insert(10, vec![(5, 0.72)]);
                p
            },
            children: {
                let mut c = HashMap::new();
                c.insert(5, vec![(10, 0.72)]);
                c
            },
            discovered_at_cycle: 100,
        };

        // Get attention for target dim 10
        let weights_with = enhancer_with_causal.causal_attention_weights(10);
        let weights_without = enhancer_without_causal.causal_attention_weights(10);

        // With causal knowledge, the parent (dim 5) should be weighted higher
        assert!(
            weights_with[5] > weights_without[5],
            "Causal parent should get more attention: {} > {}",
            weights_with[5],
            weights_without[5]
        );

        // The attention should be normalized (sum to 1)
        let sum_with: f32 = weights_with.iter().sum();
        let sum_without: f32 = weights_without.iter().sum();
        assert!((sum_with - 1.0).abs() < 0.01, "Weights should sum to 1");
        assert!((sum_without - 1.0).abs() < 0.01, "Weights should sum to 1");
    }

    #[test]
    fn test_intervention_suggestion() {
        let mut enhancer = CausalLoopEnhancer::new(42);

        // Set up causal structure where dim 0 has the most influence
        enhancer.current_graph = CausalGraph {
            n_vars: 64,
            edges: vec![
                CausalGraphEdge {
                    from: 0,
                    to: 1,
                    direction: CausalDirection::Forward,
                    confidence: 0.9,
                    strength: 0.8,
                },
                CausalGraphEdge {
                    from: 0,
                    to: 2,
                    direction: CausalDirection::Forward,
                    confidence: 0.85,
                    strength: 0.7,
                },
            ],
            parents: HashMap::new(),
            children: {
                let mut c = HashMap::new();
                c.insert(0, vec![(1, 0.72), (2, 0.595)]);
                c
            },
            discovered_at_cycle: 100,
        };

        // With high exploration probability, we should get suggestions
        enhancer.config.exploration_probability = 1.0;

        let suggestion = enhancer.suggest_intervention();
        assert!(
            suggestion.is_some(),
            "Should suggest intervention when structure exists"
        );

        if let Some((var, value)) = suggestion {
            assert_eq!(var, 0, "Should suggest intervening on most influential var");
            assert!((-1.0..=1.0).contains(&value), "Value should be in range");
        }
    }

    #[test]
    fn test_record_and_history() {
        let mut enhancer = CausalLoopEnhancer::with_config(CausalEnhancerConfig {
            max_history: 10,
            ..Default::default()
        });

        let input = ContinuousHV::random(HDC_DIMENSION, 1);
        let output = ContinuousHV::random(HDC_DIMENSION, 2);

        // Record more than max_history
        for _ in 0..15 {
            enhancer.record_cycle(&input, &output);
        }

        // Should be capped at max_history
        assert_eq!(enhancer.history_size(), 10);
    }
}
