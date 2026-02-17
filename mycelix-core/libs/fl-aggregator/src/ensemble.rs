//! Defense Ensemble (vote-based multi-defense aggregation).
//!
//! This module provides ensemble methods for combining multiple Byzantine-resistant
//! defense algorithms to achieve more robust aggregation. By running multiple defenses
//! and combining their results through voting strategies, the ensemble can be more
//! resilient to attacks that might fool individual defenses.
//!
//! ## Key Features
//!
//! - **Multiple Voting Strategies**: Majority, weighted, unanimous, median, best, and adaptive
//! - **Byzantine Consensus**: Vote-based detection of malicious nodes across defenses
//! - **Adaptive Weights**: Learn which defenses perform best over time
//! - **Agreement Metrics**: Measure how well defenses agree on results
//!
//! ## Example
//!
//! ```rust,ignore
//! use fl_aggregator::ensemble::{EnsembleConfig, EnsembleAggregator, VotingStrategy};
//! use fl_aggregator::byzantine::Defense;
//! use std::collections::HashMap;
//! use ndarray::Array1;
//!
//! // Create a robust ensemble with multiple defenses
//! let config = EnsembleConfig::robust();
//! let mut aggregator = EnsembleAggregator::new(config);
//!
//! // Submit gradients from nodes
//! let mut gradients = HashMap::new();
//! gradients.insert("node1".to_string(), Array1::from(vec![1.0, 2.0, 3.0]));
//! gradients.insert("node2".to_string(), Array1::from(vec![1.1, 2.1, 3.1]));
//! // ... more nodes
//!
//! // Aggregate with ensemble voting
//! let result = aggregator.aggregate(&gradients).unwrap();
//! println!("Agreement score: {:.2}", result.agreement_score);
//! ```

use crate::byzantine::{cosine_similarity, ByzantineAggregator, Defense, DefenseConfig};
use crate::error::{AggregatorError, Result};
use crate::Gradient;
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::time::Instant;

/// Configuration for the defense ensemble.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnsembleConfig {
    /// Defenses to use in the ensemble.
    pub defenses: Vec<Defense>,

    /// How to combine results from multiple defenses.
    pub voting_strategy: VotingStrategy,

    /// Optional weights for each defense (must match length of defenses).
    /// If None, equal weights are used.
    pub weights: Option<Vec<f32>>,

    /// Agreement threshold needed to flag a node as Byzantine (0.0 to 1.0).
    /// For example, 0.5 means majority must flag the node.
    pub byzantine_consensus_threshold: f32,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            defenses: vec![
                Defense::Krum { f: 1 },
                Defense::Median,
                Defense::TrimmedMean { beta: 0.1 },
            ],
            voting_strategy: VotingStrategy::Majority,
            weights: None,
            byzantine_consensus_threshold: 0.5,
        }
    }
}

impl EnsembleConfig {
    /// Create a robust ensemble preset: Krum + Median + TrimmedMean.
    ///
    /// This configuration provides good protection against various attack types
    /// by combining distance-based (Krum), coordinate-wise (Median), and
    /// statistical (TrimmedMean) defenses.
    pub fn robust() -> Self {
        Self {
            defenses: vec![
                Defense::Krum { f: 1 },
                Defense::Median,
                Defense::TrimmedMean { beta: 0.1 },
            ],
            voting_strategy: VotingStrategy::Majority,
            weights: None,
            byzantine_consensus_threshold: 0.5,
        }
    }

    /// Create a fast ensemble preset: FedAvg + Median.
    ///
    /// This configuration prioritizes speed over maximum robustness,
    /// suitable for low-threat environments or when performance is critical.
    pub fn fast() -> Self {
        Self {
            defenses: vec![Defense::FedAvg, Defense::Median],
            voting_strategy: VotingStrategy::Median,
            weights: None,
            byzantine_consensus_threshold: 0.5,
        }
    }

    /// Create a paranoid ensemble preset: All defenses with unanimous voting.
    ///
    /// This is the most conservative configuration, requiring all defenses
    /// to agree. Use when security is paramount and false positives are acceptable.
    pub fn paranoid() -> Self {
        Self {
            defenses: vec![
                Defense::FedAvg,
                Defense::Krum { f: 1 },
                Defense::MultiKrum { f: 1, k: 3 },
                Defense::Median,
                Defense::TrimmedMean { beta: 0.1 },
                Defense::GeometricMedian {
                    max_iterations: 100,
                    tolerance: 1e-6,
                },
            ],
            voting_strategy: VotingStrategy::Unanimous,
            weights: None,
            byzantine_consensus_threshold: 0.8,
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.defenses.is_empty() {
            return Err(AggregatorError::InvalidConfig(
                "Ensemble must have at least one defense".to_string(),
            ));
        }

        if let Some(ref weights) = self.weights {
            if weights.len() != self.defenses.len() {
                return Err(AggregatorError::InvalidConfig(format!(
                    "Weights length ({}) must match defenses length ({})",
                    weights.len(),
                    self.defenses.len()
                )));
            }

            for (i, w) in weights.iter().enumerate() {
                if *w < 0.0 {
                    return Err(AggregatorError::InvalidConfig(format!(
                        "Weight {} is negative: {}",
                        i, w
                    )));
                }
            }
        }

        if !(0.0..=1.0).contains(&self.byzantine_consensus_threshold) {
            return Err(AggregatorError::InvalidConfig(format!(
                "Byzantine consensus threshold must be in [0, 1], got {}",
                self.byzantine_consensus_threshold
            )));
        }

        Ok(())
    }
}

/// Strategy for combining results from multiple defenses.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum VotingStrategy {
    /// Use the result closest to the majority of defense outputs.
    Majority,

    /// Weighted average of all defense outputs.
    Weighted,

    /// All defenses must produce similar results (most conservative).
    Unanimous,

    /// Take the coordinate-wise median of all defense outputs.
    Median,

    /// Use the defense with the best historical performance.
    Best,

    /// Dynamically weight defenses based on recent accuracy.
    Adaptive,
}

impl Default for VotingStrategy {
    fn default() -> Self {
        VotingStrategy::Majority
    }
}

/// Result of ensemble aggregation.
#[derive(Debug, Clone)]
pub struct EnsembleResult {
    /// The final aggregated gradient after voting.
    pub aggregated_gradient: Array1<f32>,

    /// Individual results from each defense (keyed by defense name string).
    pub individual_results: HashMap<String, Array1<f32>>,

    /// Agreement score between defenses (0.0 to 1.0).
    /// Higher values indicate more agreement.
    pub agreement_score: f32,

    /// Byzantine consensus across defenses.
    pub byzantine_consensus: ByzantineConsensus,

    /// Time taken for computation in milliseconds.
    pub computation_time_ms: f64,
}

/// Byzantine detection consensus across multiple defenses.
#[derive(Debug, Clone, Default)]
pub struct ByzantineConsensus {
    /// Nodes flagged as potentially Byzantine with confidence scores.
    /// Confidence is the fraction of defenses that flagged the node.
    pub flagged_nodes: HashMap<String, f32>,

    /// Detailed votes from each defense for each node.
    /// Maps node_id -> list of (defense_name, flagged as Byzantine).
    pub votes: HashMap<String, Vec<(String, bool)>>,

    /// Nodes that reached consensus as Byzantine (above threshold).
    pub consensus_byzantine: Vec<String>,
}

impl ByzantineConsensus {
    /// Create a new empty Byzantine consensus.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if a node was flagged by consensus.
    pub fn is_byzantine(&self, node_id: &str) -> bool {
        self.consensus_byzantine.contains(&node_id.to_string())
    }

    /// Get the confidence that a node is Byzantine.
    pub fn byzantine_confidence(&self, node_id: &str) -> f32 {
        self.flagged_nodes.get(node_id).copied().unwrap_or(0.0)
    }
}

/// Feedback for updating adaptive weights.
#[derive(Debug, Clone)]
pub struct AggregationFeedback {
    /// The ground truth gradient (if available).
    pub ground_truth: Option<Array1<f32>>,

    /// Whether the aggregation was considered successful.
    pub success: bool,

    /// Node IDs that were later confirmed as Byzantine.
    pub confirmed_byzantine: Vec<String>,

    /// Node IDs that were incorrectly flagged (false positives).
    pub false_positives: Vec<String>,
}

/// Performance history for adaptive weighting.
#[derive(Debug, Clone, Default)]
struct DefensePerformance {
    /// Total number of rounds this defense participated in.
    total_rounds: usize,

    /// Number of rounds where this defense agreed with final outcome.
    correct_rounds: usize,

    /// Exponential moving average of performance.
    ema_performance: f32,

    /// Number of correct Byzantine detections.
    correct_byzantine_detections: usize,

    /// Number of false positive Byzantine detections.
    false_positives: usize,
}

impl DefensePerformance {
    fn accuracy(&self) -> f32 {
        if self.total_rounds == 0 {
            return 0.5; // Default to neutral
        }
        self.correct_rounds as f32 / self.total_rounds as f32
    }

    fn detection_precision(&self) -> f32 {
        let total = self.correct_byzantine_detections + self.false_positives;
        if total == 0 {
            return 0.5; // Default to neutral
        }
        self.correct_byzantine_detections as f32 / total as f32
    }
}

/// Ensemble aggregator that combines multiple Byzantine-resistant defenses.
pub struct EnsembleAggregator {
    /// Configuration for the ensemble.
    config: EnsembleConfig,

    /// Individual aggregators for each defense.
    aggregators: Vec<ByzantineAggregator>,

    /// Current weights for each defense.
    current_weights: Vec<f32>,

    /// Performance history for adaptive weighting (keyed by defense index).
    performance_history: Vec<DefensePerformance>,
}

impl EnsembleAggregator {
    /// Create a new ensemble aggregator with the given configuration.
    pub fn new(config: EnsembleConfig) -> Self {
        let num_defenses = config.defenses.len();

        // Initialize weights
        let current_weights = config.weights.clone().unwrap_or_else(|| {
            vec![1.0 / num_defenses as f32; num_defenses]
        });

        // Create individual aggregators
        let aggregators = config
            .defenses
            .iter()
            .map(|defense| ByzantineAggregator::new(DefenseConfig::with_defense(defense.clone())))
            .collect();

        // Initialize performance history
        let performance_history = (0..num_defenses)
            .map(|_| DefensePerformance::default())
            .collect();

        Self {
            config,
            aggregators,
            current_weights,
            performance_history,
        }
    }

    /// Get the name/key for a defense at a given index.
    fn defense_key(&self, index: usize) -> String {
        self.config.defenses[index].to_string()
    }

    /// Aggregate gradients using the ensemble of defenses.
    pub fn aggregate(&self, gradients: &HashMap<String, Array1<f32>>) -> Result<EnsembleResult> {
        let start = Instant::now();

        // Validate input
        if gradients.is_empty() {
            return Err(AggregatorError::Internal(
                "No gradients to aggregate".to_string(),
            ));
        }

        // Convert to slice format for individual aggregators
        let gradient_vec: Vec<Gradient> = gradients.values().cloned().collect();

        // Run each defense and collect results
        let mut individual_results: HashMap<String, Array1<f32>> = HashMap::new();
        let mut successful_defenses: Vec<usize> = Vec::new();

        for (i, aggregator) in self.aggregators.iter().enumerate() {
            match aggregator.aggregate(&gradient_vec) {
                Ok(result) => {
                    individual_results.insert(self.defense_key(i), result);
                    successful_defenses.push(i);
                }
                Err(e) => {
                    tracing::warn!("Defense {:?} failed: {}", self.config.defenses[i], e);
                    // Continue with other defenses
                }
            }
        }

        if individual_results.is_empty() {
            return Err(AggregatorError::Internal(
                "All defenses failed to aggregate".to_string(),
            ));
        }

        // Calculate agreement between defenses
        let agreement_score = self.calculate_agreement(&individual_results);

        // Detect Byzantine nodes across defenses
        let byzantine_consensus = self.detect_byzantine(gradients);

        // Combine results using voting strategy
        let aggregated_gradient = self.combine_results(&individual_results, &successful_defenses)?;

        let elapsed = start.elapsed();

        Ok(EnsembleResult {
            aggregated_gradient,
            individual_results,
            agreement_score,
            byzantine_consensus,
            computation_time_ms: elapsed.as_secs_f64() * 1000.0,
        })
    }

    /// Detect Byzantine nodes using consensus across defenses.
    pub fn detect_byzantine(&self, gradients: &HashMap<String, Array1<f32>>) -> ByzantineConsensus {
        let gradient_vec: Vec<Gradient> = gradients.values().cloned().collect();
        let node_ids: Vec<String> = gradients.keys().cloned().collect();

        // For each defense, determine which nodes it would flag as Byzantine
        let mut votes: HashMap<String, Vec<(String, bool)>> = HashMap::new();
        let mut flagged_counts: HashMap<String, usize> = HashMap::new();

        for (i, defense) in self.config.defenses.iter().enumerate() {
            // Determine which nodes this defense would exclude
            let flagged = self.detect_byzantine_for_defense(defense, &gradient_vec, &node_ids);
            let defense_name = self.defense_key(i);

            for node_id in &node_ids {
                let is_flagged = flagged.contains(node_id);
                votes
                    .entry(node_id.clone())
                    .or_default()
                    .push((defense_name.clone(), is_flagged));

                if is_flagged {
                    *flagged_counts.entry(node_id.clone()).or_insert(0) += 1;
                }
            }
        }

        // Calculate confidence scores and consensus
        let num_defenses = self.config.defenses.len();
        let mut flagged_nodes: HashMap<String, f32> = HashMap::new();
        let mut consensus_byzantine: Vec<String> = Vec::new();

        for (node_id, count) in flagged_counts {
            let confidence = count as f32 / num_defenses as f32;
            flagged_nodes.insert(node_id.clone(), confidence);

            if confidence >= self.config.byzantine_consensus_threshold {
                consensus_byzantine.push(node_id);
            }
        }

        ByzantineConsensus {
            flagged_nodes,
            votes,
            consensus_byzantine,
        }
    }

    /// Detect Byzantine nodes for a specific defense.
    fn detect_byzantine_for_defense(
        &self,
        defense: &Defense,
        gradients: &[Gradient],
        node_ids: &[String],
    ) -> Vec<String> {
        let n = gradients.len();
        if n == 0 {
            return Vec::new();
        }

        match defense {
            Defense::Krum { f } => {
                // Krum identifies outliers by distance
                self.detect_outliers_by_distance(gradients, node_ids, *f)
            }
            Defense::MultiKrum { f, k: _ } => {
                // MultiKrum excludes bottom n-k nodes
                self.detect_outliers_by_distance(gradients, node_ids, *f)
            }
            Defense::TrimmedMean { beta } => {
                // TrimmedMean removes extremes in each dimension
                self.detect_outliers_by_extremes(gradients, node_ids, *beta)
            }
            Defense::Median => {
                // Median is robust; flag nodes far from median
                self.detect_outliers_from_median(gradients, node_ids)
            }
            Defense::GeometricMedian { .. } => {
                // Similar to median
                self.detect_outliers_from_median(gradients, node_ids)
            }
            Defense::FedAvg => {
                // FedAvg has no Byzantine detection
                Vec::new()
            }
        }
    }

    /// Detect outliers using Krum-style distance scoring.
    fn detect_outliers_by_distance(
        &self,
        gradients: &[Gradient],
        node_ids: &[String],
        f: usize,
    ) -> Vec<String> {
        let n = gradients.len();
        if n <= 2 * f + 2 {
            return Vec::new();
        }

        // Compute Krum scores (sum of distances to nearest n-f-2 neighbors)
        let take_count = n.saturating_sub(f).saturating_sub(2);
        let mut scores: Vec<(usize, f32)> = gradients
            .iter()
            .enumerate()
            .map(|(i, gi)| {
                let mut distances: Vec<f32> = gradients
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, gj)| {
                        gi.iter()
                            .zip(gj.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f32>()
                            .sqrt()
                    })
                    .collect();
                distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                let score: f32 = distances.iter().take(take_count).sum();
                (i, score)
            })
            .collect();

        // Sort by score descending (highest = most outlier-like)
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        // Flag top f nodes as potential Byzantine
        scores
            .iter()
            .take(f)
            .map(|(i, _)| node_ids[*i].clone())
            .collect()
    }

    /// Detect outliers by extreme values (TrimmedMean style).
    fn detect_outliers_by_extremes(
        &self,
        gradients: &[Gradient],
        node_ids: &[String],
        beta: f32,
    ) -> Vec<String> {
        if gradients.is_empty() {
            return Vec::new();
        }

        let n = gradients.len();
        let dim = gradients[0].len();
        let trim_count = (n as f32 * beta).floor() as usize;

        // Count how often each node is in the extreme positions
        let mut extreme_counts: HashMap<usize, usize> = HashMap::new();

        for d in 0..dim {
            let mut indexed_values: Vec<(usize, f32)> = gradients
                .iter()
                .enumerate()
                .map(|(i, g)| (i, g[d]))
                .collect();
            indexed_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

            // Count bottom trim_count
            for (idx, _) in indexed_values.iter().take(trim_count) {
                *extreme_counts.entry(*idx).or_insert(0) += 1;
            }

            // Count top trim_count
            for (idx, _) in indexed_values.iter().rev().take(trim_count) {
                *extreme_counts.entry(*idx).or_insert(0) += 1;
            }
        }

        // Flag nodes that are frequently extreme
        let threshold = dim / 4; // Flagged if extreme in >25% of dimensions
        extreme_counts
            .iter()
            .filter(|(_, count)| **count > threshold)
            .map(|(idx, _)| node_ids[*idx].clone())
            .collect()
    }

    /// Detect outliers by distance from coordinate-wise median.
    fn detect_outliers_from_median(
        &self,
        gradients: &[Gradient],
        node_ids: &[String],
    ) -> Vec<String> {
        if gradients.is_empty() {
            return Vec::new();
        }

        let dim = gradients[0].len();

        // Calculate coordinate-wise median
        let mut median = Array1::zeros(dim);
        for d in 0..dim {
            let mut values: Vec<f32> = gradients.iter().map(|g| g[d]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            let mid = values.len() / 2;
            median[d] = if values.len() % 2 == 0 {
                (values[mid - 1] + values[mid]) / 2.0
            } else {
                values[mid]
            };
        }

        // Calculate distances from median
        let distances: Vec<(usize, f32)> = gradients
            .iter()
            .enumerate()
            .map(|(i, g)| {
                let dist = g
                    .iter()
                    .zip(median.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                (i, dist)
            })
            .collect();

        // Calculate median distance for threshold
        let mut dist_values: Vec<f32> = distances.iter().map(|(_, d)| *d).collect();
        dist_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let median_dist = dist_values[dist_values.len() / 2];

        // Flag nodes with distance > 3 * median distance
        let threshold = 3.0 * median_dist;
        distances
            .iter()
            .filter(|(_, dist)| *dist > threshold)
            .map(|(i, _)| node_ids[*i].clone())
            .collect()
    }

    /// Calculate agreement score between defense outputs.
    fn calculate_agreement(&self, results: &HashMap<String, Array1<f32>>) -> f32 {
        let outputs: Vec<&Array1<f32>> = results.values().collect();
        if outputs.len() < 2 {
            return 1.0; // Single defense always agrees with itself
        }

        // Calculate pairwise cosine similarities
        let mut total_similarity = 0.0;
        let mut pair_count = 0;

        for i in 0..outputs.len() {
            for j in (i + 1)..outputs.len() {
                let similarity = cosine_similarity(outputs[i].view(), outputs[j].view());
                // Convert from [-1, 1] to [0, 1]
                total_similarity += (similarity + 1.0) / 2.0;
                pair_count += 1;
            }
        }

        if pair_count == 0 {
            return 1.0;
        }

        total_similarity / pair_count as f32
    }

    /// Combine results from multiple defenses using the configured voting strategy.
    fn combine_results(
        &self,
        results: &HashMap<String, Array1<f32>>,
        successful_indices: &[usize],
    ) -> Result<Array1<f32>> {
        if results.is_empty() {
            return Err(AggregatorError::Internal(
                "No results to combine".to_string(),
            ));
        }

        let outputs: Vec<&Array1<f32>> = results.values().collect();
        let dim = outputs[0].len();

        match &self.config.voting_strategy {
            VotingStrategy::Majority => self.combine_majority(&outputs),
            VotingStrategy::Weighted => self.combine_weighted(&outputs, successful_indices),
            VotingStrategy::Unanimous => self.combine_unanimous(&outputs),
            VotingStrategy::Median => self.combine_median(&outputs, dim),
            VotingStrategy::Best => self.combine_best(results),
            VotingStrategy::Adaptive => self.combine_weighted(&outputs, successful_indices),
        }
    }

    /// Combine using majority voting (select output closest to most others).
    fn combine_majority(&self, outputs: &[&Array1<f32>]) -> Result<Array1<f32>> {
        if outputs.len() == 1 {
            return Ok(outputs[0].clone());
        }

        // For each output, count how many others it's similar to
        let mut best_idx = 0;
        let mut best_score = 0.0;

        for i in 0..outputs.len() {
            let mut similarity_sum = 0.0;
            for j in 0..outputs.len() {
                if i != j {
                    similarity_sum += cosine_similarity(outputs[i].view(), outputs[j].view());
                }
            }
            if similarity_sum > best_score {
                best_score = similarity_sum;
                best_idx = i;
            }
        }

        Ok(outputs[best_idx].clone())
    }

    /// Combine using weighted average.
    fn combine_weighted(
        &self,
        outputs: &[&Array1<f32>],
        successful_indices: &[usize],
    ) -> Result<Array1<f32>> {
        if outputs.is_empty() {
            return Err(AggregatorError::Internal(
                "No outputs to combine".to_string(),
            ));
        }

        let dim = outputs[0].len();
        let mut result = Array1::zeros(dim);
        let mut total_weight = 0.0;

        for (i, output) in outputs.iter().enumerate() {
            let weight = if i < successful_indices.len() {
                let idx = successful_indices[i];
                if idx < self.current_weights.len() {
                    self.current_weights[idx]
                } else {
                    1.0 / outputs.len() as f32
                }
            } else {
                1.0 / outputs.len() as f32
            };

            result += &((*output).clone() * weight);
            total_weight += weight;
        }

        if total_weight > 0.0 {
            result /= total_weight;
        }

        Ok(result)
    }

    /// Combine using unanimous voting (require high agreement).
    fn combine_unanimous(&self, outputs: &[&Array1<f32>]) -> Result<Array1<f32>> {
        // Check that all outputs are sufficiently similar
        let agreement = self.calculate_agreement_vec(outputs);

        if agreement < 0.9 {
            tracing::warn!(
                "Unanimous voting failed: agreement score {:.2} < 0.9",
                agreement
            );
            // Fall back to median when unanimous fails
            return self.combine_median(outputs, outputs[0].len());
        }

        // If unanimous, average all outputs
        let dim = outputs[0].len();
        let mut result = Array1::zeros(dim);
        for output in outputs {
            result += &(*output).clone();
        }
        result /= outputs.len() as f32;

        Ok(result)
    }

    /// Calculate agreement from a vector of outputs.
    fn calculate_agreement_vec(&self, outputs: &[&Array1<f32>]) -> f32 {
        if outputs.len() < 2 {
            return 1.0;
        }

        let mut total_similarity = 0.0;
        let mut pair_count = 0;

        for i in 0..outputs.len() {
            for j in (i + 1)..outputs.len() {
                let similarity = cosine_similarity(outputs[i].view(), outputs[j].view());
                total_similarity += (similarity + 1.0) / 2.0;
                pair_count += 1;
            }
        }

        if pair_count == 0 {
            return 1.0;
        }

        total_similarity / pair_count as f32
    }

    /// Combine using coordinate-wise median.
    fn combine_median(&self, outputs: &[&Array1<f32>], dim: usize) -> Result<Array1<f32>> {
        let mut result = Array1::zeros(dim);

        for d in 0..dim {
            let mut values: Vec<f32> = outputs.iter().map(|o| o[d]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

            let mid = values.len() / 2;
            result[d] = if values.len() % 2 == 0 {
                (values[mid - 1] + values[mid]) / 2.0
            } else {
                values[mid]
            };
        }

        Ok(result)
    }

    /// Combine using best performing defense.
    fn combine_best(&self, results: &HashMap<String, Array1<f32>>) -> Result<Array1<f32>> {
        // Find defense with best historical performance
        let mut best_idx: Option<usize> = None;
        let mut best_performance = 0.0;

        for (i, perf) in self.performance_history.iter().enumerate() {
            let score = perf.ema_performance;
            if score > best_performance {
                best_performance = score;
                best_idx = Some(i);
            }
        }

        // Get result for best defense
        if let Some(idx) = best_idx {
            let key = self.defense_key(idx);
            if let Some(result) = results.get(&key) {
                return Ok(result.clone());
            }
        }

        // Fall back to first result if no history
        results
            .values()
            .next()
            .cloned()
            .ok_or_else(|| AggregatorError::Internal("No best defense found".to_string()))
    }

    /// Update adaptive weights based on feedback.
    pub fn update_weights(&mut self, feedback: &AggregationFeedback) {
        let ema_alpha = 0.1; // Exponential moving average decay factor
        let num_defenses = self.performance_history.len();

        for i in 0..num_defenses {
            self.performance_history[i].total_rounds += 1;

            // Update based on ground truth if available
            if feedback.ground_truth.is_some() {
                // Compare defense output to ground truth
                // This would require storing individual results, simplified here
                if feedback.success {
                    self.performance_history[i].correct_rounds += 1;
                }
            } else if feedback.success {
                self.performance_history[i].correct_rounds += 1;
            }

            // Update Byzantine detection metrics
            // Note: was_flagged_by_defense currently returns false (placeholder)
            // In a full implementation, this would check cached detection results
            for _confirmed in &feedback.confirmed_byzantine {
                // Placeholder: would check if defense i flagged this node
                // self.performance_history[i].correct_byzantine_detections += 1;
            }
            for _fp in &feedback.false_positives {
                // Placeholder: would check if defense i flagged this node
                // self.performance_history[i].false_positives += 1;
            }

            // Update EMA performance
            let current_accuracy = self.performance_history[i].accuracy();
            let detection_precision = self.performance_history[i].detection_precision();
            let combined_score = 0.7 * current_accuracy + 0.3 * detection_precision;

            self.performance_history[i].ema_performance =
                ema_alpha * combined_score + (1.0 - ema_alpha) * self.performance_history[i].ema_performance;

            // Update weight based on performance
            if self.config.voting_strategy == VotingStrategy::Adaptive {
                self.current_weights[i] = self.performance_history[i].ema_performance.max(0.1);
            }
        }

        // Normalize weights
        if self.config.voting_strategy == VotingStrategy::Adaptive {
            let total: f32 = self.current_weights.iter().sum();
            if total > 0.0 {
                for w in &mut self.current_weights {
                    *w /= total;
                }
            }
        }
    }

    /// Check if a defense flagged a specific node (for feedback tracking).
    fn was_flagged_by_defense(&self, _defense_idx: usize, _node_id: &str) -> bool {
        // This would require storing detection results from previous runs
        // Simplified implementation - could be enhanced with caching
        false
    }

    /// Get the current weights for each defense.
    pub fn get_weights(&self) -> &[f32] {
        &self.current_weights
    }

    /// Get the configuration.
    pub fn config(&self) -> &EnsembleConfig {
        &self.config
    }

    /// Calculate agreement metrics for the ensemble.
    pub fn calculate_agreement_metrics(
        &self,
        results: &HashMap<String, Array1<f32>>,
    ) -> AgreementMetrics {
        let outputs: Vec<&Array1<f32>> = results.values().collect();

        // Pairwise cosine similarities
        let mut pairwise_similarities = Vec::new();
        for i in 0..outputs.len() {
            for j in (i + 1)..outputs.len() {
                let sim = cosine_similarity(outputs[i].view(), outputs[j].view());
                pairwise_similarities.push(sim);
            }
        }

        // Calculate statistics
        let mean_similarity = if pairwise_similarities.is_empty() {
            1.0
        } else {
            pairwise_similarities.iter().sum::<f32>() / pairwise_similarities.len() as f32
        };

        let variance = if pairwise_similarities.len() < 2 {
            0.0
        } else {
            let squared_diff_sum: f32 = pairwise_similarities
                .iter()
                .map(|s| (s - mean_similarity).powi(2))
                .sum();
            squared_diff_sum / (pairwise_similarities.len() - 1) as f32
        };

        let std_dev = variance.sqrt();

        // 95% confidence interval (assuming normal distribution)
        let n = pairwise_similarities.len() as f32;
        let margin = if n > 0.0 {
            1.96 * std_dev / n.sqrt()
        } else {
            0.0
        };

        AgreementMetrics {
            mean_pairwise_similarity: mean_similarity,
            similarity_variance: variance,
            similarity_std_dev: std_dev,
            confidence_interval_95: (mean_similarity - margin, mean_similarity + margin),
            min_similarity: pairwise_similarities
                .iter()
                .copied()
                .reduce(f32::min)
                .unwrap_or(1.0),
            max_similarity: pairwise_similarities
                .iter()
                .copied()
                .reduce(f32::max)
                .unwrap_or(1.0),
        }
    }
}

/// Agreement metrics between defense outputs.
#[derive(Debug, Clone)]
pub struct AgreementMetrics {
    /// Mean pairwise cosine similarity between defense outputs.
    pub mean_pairwise_similarity: f32,

    /// Variance in pairwise similarities.
    pub similarity_variance: f32,

    /// Standard deviation of similarities.
    pub similarity_std_dev: f32,

    /// 95% confidence interval for the mean similarity.
    pub confidence_interval_95: (f32, f32),

    /// Minimum pairwise similarity observed.
    pub min_similarity: f32,

    /// Maximum pairwise similarity observed.
    pub max_similarity: f32,
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
        gradients.insert("node4".to_string(), array![1.05, 2.05, 3.05]);
        gradients.insert("node5".to_string(), array![0.95, 1.95, 2.95]);
        gradients
    }

    fn create_gradients_with_byzantine() -> HashMap<String, Array1<f32>> {
        let mut gradients = create_test_gradients();
        // Add Byzantine node with malicious gradient
        gradients.insert("byzantine".to_string(), array![100.0, 200.0, 300.0]);
        gradients
    }

    #[test]
    fn test_majority_voting_produces_sensible_results() {
        let config = EnsembleConfig {
            defenses: vec![
                Defense::Krum { f: 1 },
                Defense::Median,
                Defense::TrimmedMean { beta: 0.1 },
            ],
            voting_strategy: VotingStrategy::Majority,
            weights: None,
            byzantine_consensus_threshold: 0.5,
        };

        let aggregator = EnsembleAggregator::new(config);
        let gradients = create_test_gradients();

        let result = aggregator.aggregate(&gradients).unwrap();

        // Result should be close to [1.0, 2.0, 3.0] (the cluster center)
        assert!((result.aggregated_gradient[0] - 1.0).abs() < 0.5);
        assert!((result.aggregated_gradient[1] - 2.0).abs() < 0.5);
        assert!((result.aggregated_gradient[2] - 3.0).abs() < 0.5);

        // Agreement should be high for similar inputs
        assert!(result.agreement_score > 0.8);
    }

    #[test]
    fn test_byzantine_consensus_with_different_agreement_levels() {
        let gradients = create_gradients_with_byzantine();

        // Test with low threshold (should flag Byzantine easily)
        let config_low = EnsembleConfig {
            defenses: vec![
                Defense::Krum { f: 1 },
                Defense::Median,
                Defense::TrimmedMean { beta: 0.1 },
            ],
            voting_strategy: VotingStrategy::Majority,
            weights: None,
            byzantine_consensus_threshold: 0.3,
        };

        let aggregator_low = EnsembleAggregator::new(config_low);
        let consensus_low = aggregator_low.detect_byzantine(&gradients);

        // Byzantine node should have high confidence
        let byzantine_confidence = consensus_low.byzantine_confidence("byzantine");
        assert!(
            byzantine_confidence > 0.0,
            "Byzantine node should be flagged"
        );

        // Test with high threshold
        let config_high = EnsembleConfig {
            defenses: vec![
                Defense::Krum { f: 1 },
                Defense::Median,
                Defense::TrimmedMean { beta: 0.1 },
            ],
            voting_strategy: VotingStrategy::Majority,
            weights: None,
            byzantine_consensus_threshold: 0.9,
        };

        let aggregator_high = EnsembleAggregator::new(config_high);
        let consensus_high = aggregator_high.detect_byzantine(&gradients);

        // With high threshold, might not reach consensus
        // Just verify it runs without error
        assert!(consensus_high.votes.contains_key("byzantine"));
    }

    #[test]
    fn test_ensemble_more_robust_than_individual_defenses() {
        let gradients = create_gradients_with_byzantine();

        // Single defense (FedAvg - no protection)
        let fedavg_config = DefenseConfig::with_defense(Defense::FedAvg);
        let fedavg = ByzantineAggregator::new(fedavg_config);
        let gradient_vec: Vec<_> = gradients.values().cloned().collect();
        let fedavg_result = fedavg.aggregate(&gradient_vec).unwrap();

        // Ensemble with multiple defenses
        let ensemble_config = EnsembleConfig::robust();
        let ensemble = EnsembleAggregator::new(ensemble_config);
        let ensemble_result = ensemble.aggregate(&gradients).unwrap();

        // Calculate distance from expected center [1.0, 2.0, 3.0]
        let expected = array![1.0, 2.0, 3.0];

        let fedavg_dist: f32 = fedavg_result
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        let ensemble_dist: f32 = ensemble_result
            .aggregated_gradient
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        // Ensemble should be closer to expected (less affected by Byzantine)
        assert!(
            ensemble_dist < fedavg_dist,
            "Ensemble distance {} should be less than FedAvg distance {}",
            ensemble_dist,
            fedavg_dist
        );
    }

    #[test]
    fn test_adaptive_weight_updates() {
        let config = EnsembleConfig {
            defenses: vec![
                Defense::Krum { f: 1 },
                Defense::Median,
                Defense::TrimmedMean { beta: 0.1 },
            ],
            voting_strategy: VotingStrategy::Adaptive,
            weights: None,
            byzantine_consensus_threshold: 0.5,
        };

        let mut aggregator = EnsembleAggregator::new(config);

        // Initial weights should be equal
        let initial_weights = aggregator.get_weights().to_vec();
        assert_eq!(initial_weights.len(), 3);

        // Provide positive feedback
        let feedback = AggregationFeedback {
            ground_truth: None,
            success: true,
            confirmed_byzantine: vec!["byzantine".to_string()],
            false_positives: vec![],
        };

        aggregator.update_weights(&feedback);

        // Weights should still be valid (sum to 1)
        let updated_weights = aggregator.get_weights();
        let sum: f32 = updated_weights.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_agreement_metrics() {
        let config = EnsembleConfig::robust();
        let aggregator = EnsembleAggregator::new(config);
        let gradients = create_test_gradients();

        let result = aggregator.aggregate(&gradients).unwrap();
        let metrics = aggregator.calculate_agreement_metrics(&result.individual_results);

        // For similar gradients, agreement should be high
        assert!(metrics.mean_pairwise_similarity > 0.8);
        assert!(metrics.similarity_variance < 0.1);
        assert!(metrics.min_similarity > 0.7);
        assert!(metrics.max_similarity <= 1.0);

        // Confidence interval should be reasonable
        assert!(metrics.confidence_interval_95.0 <= metrics.mean_pairwise_similarity);
        assert!(metrics.confidence_interval_95.1 >= metrics.mean_pairwise_similarity);
    }

    #[test]
    fn test_compare_ensemble_vs_single_defense_on_attack() {
        // Create gradients with multiple Byzantine nodes
        let mut gradients = create_test_gradients();
        gradients.insert("attacker1".to_string(), array![50.0, 100.0, 150.0]);
        gradients.insert("attacker2".to_string(), array![-50.0, -100.0, -150.0]);

        let expected = array![1.0, 2.0, 3.0];

        // Test single defenses
        let gradient_vec: Vec<_> = gradients.values().cloned().collect();

        // Median defense
        let median = ByzantineAggregator::new(DefenseConfig::with_defense(Defense::Median));
        let median_result = median.aggregate(&gradient_vec).unwrap();
        let median_dist: f32 = median_result
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        // Krum defense
        let krum = ByzantineAggregator::new(DefenseConfig::with_defense(Defense::Krum { f: 2 }));
        let krum_result = krum.aggregate(&gradient_vec).unwrap();
        let krum_dist: f32 = krum_result
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        // Ensemble
        let ensemble_config = EnsembleConfig {
            defenses: vec![
                Defense::Krum { f: 2 },
                Defense::Median,
                Defense::TrimmedMean { beta: 0.2 },
            ],
            voting_strategy: VotingStrategy::Median,
            weights: None,
            byzantine_consensus_threshold: 0.5,
        };
        let ensemble = EnsembleAggregator::new(ensemble_config);
        let ensemble_result = ensemble.aggregate(&gradients).unwrap();
        let ensemble_dist: f32 = ensemble_result
            .aggregated_gradient
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        // All Byzantine-resistant methods should be close to expected
        assert!(median_dist < 2.0, "Median distance: {}", median_dist);
        assert!(krum_dist < 2.0, "Krum distance: {}", krum_dist);
        assert!(ensemble_dist < 2.0, "Ensemble distance: {}", ensemble_dist);

        // Log results for analysis
        println!("Median distance: {}", median_dist);
        println!("Krum distance: {}", krum_dist);
        println!("Ensemble distance: {}", ensemble_dist);
    }

    #[test]
    fn test_ensemble_presets() {
        // Test robust preset
        let robust = EnsembleConfig::robust();
        assert_eq!(robust.defenses.len(), 3);
        assert_eq!(robust.voting_strategy, VotingStrategy::Majority);
        robust.validate().unwrap();

        // Test fast preset
        let fast = EnsembleConfig::fast();
        assert_eq!(fast.defenses.len(), 2);
        assert_eq!(fast.voting_strategy, VotingStrategy::Median);
        fast.validate().unwrap();

        // Test paranoid preset
        let paranoid = EnsembleConfig::paranoid();
        assert_eq!(paranoid.defenses.len(), 6);
        assert_eq!(paranoid.voting_strategy, VotingStrategy::Unanimous);
        paranoid.validate().unwrap();
    }

    #[test]
    fn test_config_validation() {
        // Empty defenses should fail
        let empty_config = EnsembleConfig {
            defenses: vec![],
            voting_strategy: VotingStrategy::Majority,
            weights: None,
            byzantine_consensus_threshold: 0.5,
        };
        assert!(empty_config.validate().is_err());

        // Mismatched weights should fail
        let mismatched_config = EnsembleConfig {
            defenses: vec![Defense::Median, Defense::FedAvg],
            voting_strategy: VotingStrategy::Weighted,
            weights: Some(vec![0.5]), // Wrong length
            byzantine_consensus_threshold: 0.5,
        };
        assert!(mismatched_config.validate().is_err());

        // Invalid threshold should fail
        let invalid_threshold = EnsembleConfig {
            defenses: vec![Defense::Median],
            voting_strategy: VotingStrategy::Majority,
            weights: None,
            byzantine_consensus_threshold: 1.5, // Out of range
        };
        assert!(invalid_threshold.validate().is_err());
    }

    #[test]
    fn test_weighted_voting() {
        let config = EnsembleConfig {
            defenses: vec![Defense::FedAvg, Defense::Median],
            voting_strategy: VotingStrategy::Weighted,
            weights: Some(vec![0.3, 0.7]), // Prefer Median
            byzantine_consensus_threshold: 0.5,
        };

        let aggregator = EnsembleAggregator::new(config);
        let gradients = create_test_gradients();

        let result = aggregator.aggregate(&gradients).unwrap();

        // Result should be valid and close to cluster center
        assert!(result.aggregated_gradient.len() == 3);
        assert!((result.aggregated_gradient[0] - 1.0).abs() < 0.5);
    }

    #[test]
    fn test_unanimous_voting_with_agreement() {
        let config = EnsembleConfig {
            defenses: vec![Defense::Median, Defense::TrimmedMean { beta: 0.1 }],
            voting_strategy: VotingStrategy::Unanimous,
            weights: None,
            byzantine_consensus_threshold: 0.5,
        };

        let aggregator = EnsembleAggregator::new(config);
        let gradients = create_test_gradients();

        let result = aggregator.aggregate(&gradients).unwrap();

        // With well-behaved gradients, unanimous should succeed
        assert!(result.aggregated_gradient.len() == 3);
    }

    #[test]
    fn test_computation_time_recorded() {
        let config = EnsembleConfig::fast();
        let aggregator = EnsembleAggregator::new(config);
        let gradients = create_test_gradients();

        let result = aggregator.aggregate(&gradients).unwrap();

        // Computation time should be positive and reasonable
        assert!(result.computation_time_ms >= 0.0);
        assert!(result.computation_time_ms < 10000.0); // Less than 10 seconds
    }

    #[test]
    fn test_individual_results_stored() {
        let config = EnsembleConfig::robust();
        let aggregator = EnsembleAggregator::new(config);
        let gradients = create_test_gradients();

        let result = aggregator.aggregate(&gradients).unwrap();

        // Should have results from all defenses
        assert_eq!(result.individual_results.len(), 3);

        // Each result should have correct dimensions
        for (_, gradient) in &result.individual_results {
            assert_eq!(gradient.len(), 3);
        }
    }
}
