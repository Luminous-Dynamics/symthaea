// Ported from fl-aggregator/src/ensemble.rs
//! Ensemble Defense: Multi-defense voting for robust Byzantine detection.
//!
//! Runs multiple aggregation algorithms (Krum, Median, TrimmedMean, etc.)
//! and combines their results through configurable voting strategies.
//! Implements the [`ByzantinePlugin`] trait so it can be plugged directly
//! into the `UnifiedPipeline`.
//!
//! # Voting Strategies
//!
//! - **Majority**: Select the result closest to the most other results.
//! - **Weighted**: Weighted average using per-defense weights.
//! - **Unanimous**: Require high agreement; fall back to median otherwise.
//! - **Median**: Coordinate-wise median of all defense outputs.
//! - **Best**: Use the defense with best historical performance.
//! - **Adaptive**: Dynamically weight defenses based on recent accuracy.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::aggregation::{self, AggregationError};
use crate::pipeline::{ExternalWeightMap, ParticipantWeightAdjustment};
use crate::plugins::ByzantinePlugin;
use crate::types::{AggregationMethod, GradientUpdate};

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for the defense ensemble.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnsembleConfig {
    /// Aggregation methods to use in the ensemble.
    pub methods: Vec<AggregationMethod>,

    /// How to combine results from multiple defenses.
    pub voting_strategy: VotingStrategy,

    /// Optional weights for each method (must match length of methods).
    /// If None, equal weights are used.
    pub weights: Option<Vec<f32>>,

    /// Agreement threshold needed to flag a node as Byzantine (0.0 to 1.0).
    /// For example, 0.5 means majority must flag the node.
    pub byzantine_consensus_threshold: f32,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            methods: vec![
                AggregationMethod::Krum,
                AggregationMethod::Median,
                AggregationMethod::TrimmedMean,
            ],
            voting_strategy: VotingStrategy::Majority,
            weights: None,
            byzantine_consensus_threshold: 0.5,
        }
    }
}

impl EnsembleConfig {
    /// Create a robust ensemble preset: Krum + Median + TrimmedMean.
    pub fn robust() -> Self {
        Self::default()
    }

    /// Create a fast ensemble preset: FedAvg + Median.
    pub fn fast() -> Self {
        Self {
            methods: vec![AggregationMethod::FedAvg, AggregationMethod::Median],
            voting_strategy: VotingStrategy::Median,
            weights: None,
            byzantine_consensus_threshold: 0.5,
        }
    }

    /// Create a paranoid ensemble preset: All defenses with unanimous voting.
    pub fn paranoid() -> Self {
        Self {
            methods: vec![
                AggregationMethod::FedAvg,
                AggregationMethod::Krum,
                AggregationMethod::MultiKrum,
                AggregationMethod::Median,
                AggregationMethod::TrimmedMean,
                AggregationMethod::GeometricMedian,
            ],
            voting_strategy: VotingStrategy::Unanimous,
            weights: None,
            byzantine_consensus_threshold: 0.8,
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), EnsembleError> {
        if self.methods.is_empty() {
            return Err(EnsembleError::InvalidConfig(
                "Ensemble must have at least one defense".into(),
            ));
        }

        if let Some(ref weights) = self.weights {
            if weights.len() != self.methods.len() {
                return Err(EnsembleError::InvalidConfig(format!(
                    "Weights length ({}) must match methods length ({})",
                    weights.len(),
                    self.methods.len()
                )));
            }
            for (i, w) in weights.iter().enumerate() {
                if *w < 0.0 {
                    return Err(EnsembleError::InvalidConfig(format!(
                        "Weight {} is negative: {}",
                        i, w
                    )));
                }
            }
        }

        if !(0.0..=1.0).contains(&self.byzantine_consensus_threshold) {
            return Err(EnsembleError::InvalidConfig(format!(
                "Byzantine consensus threshold must be in [0, 1], got {}",
                self.byzantine_consensus_threshold
            )));
        }

        Ok(())
    }
}

// =============================================================================
// Voting Strategy
// =============================================================================

/// Strategy for combining results from multiple defenses.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum VotingStrategy {
    /// Select the result closest to the majority of defense outputs.
    Majority,
    /// Weighted average of all defense outputs.
    Weighted,
    /// Require high agreement; fall back to median if agreement < 0.9.
    Unanimous,
    /// Coordinate-wise median of all defense outputs.
    Median,
    /// Use the defense with the best historical performance.
    Best,
    /// Dynamically weight defenses based on recent accuracy (same as Weighted).
    Adaptive,
}

impl Default for VotingStrategy {
    fn default() -> Self {
        VotingStrategy::Majority
    }
}

// =============================================================================
// Error & Result Types
// =============================================================================

/// Errors specific to ensemble aggregation.
#[derive(Debug, thiserror::Error)]
pub enum EnsembleError {
    #[error("Invalid ensemble config: {0}")]
    InvalidConfig(String),
    #[error("All defenses failed")]
    AllDefensesFailed,
    #[error("No gradient updates provided")]
    NoUpdates,
    #[error("Aggregation error: {0}")]
    Aggregation(#[from] AggregationError),
}

/// Ensemble aggregation result.
#[derive(Debug, Clone)]
pub struct EnsembleResult {
    /// The final aggregated gradient after voting.
    pub aggregated: Vec<f32>,
    /// Individual results from each defense (keyed by method name).
    pub individual_results: HashMap<String, Vec<f32>>,
    /// Agreement score between defenses (0.0 to 1.0).
    pub agreement_score: f32,
    /// Byzantine consensus: node IDs flagged with confidence.
    pub flagged_nodes: HashMap<String, f32>,
    /// Nodes that reached consensus threshold.
    pub consensus_byzantine: Vec<String>,
}

// =============================================================================
// Per-defense performance tracking
// =============================================================================

#[derive(Debug, Clone, Default)]
struct DefensePerformance {
    total_rounds: usize,
    correct_rounds: usize,
    ema_performance: f32,
}

impl DefensePerformance {
    fn accuracy(&self) -> f32 {
        if self.total_rounds == 0 {
            return 0.5;
        }
        self.correct_rounds as f32 / self.total_rounds as f32
    }
}

// =============================================================================
// Ensemble Defense
// =============================================================================

/// Ensemble defense that combines multiple aggregation methods.
///
/// Can be used standalone or as a `ByzantinePlugin` in the unified pipeline.
pub struct EnsembleDefense {
    config: EnsembleConfig,
    current_weights: Vec<f32>,
    performance: Vec<DefensePerformance>,
}

impl EnsembleDefense {
    /// Create a new ensemble defense.
    pub fn new(config: EnsembleConfig) -> Self {
        let n = config.methods.len();
        let weights = config
            .weights
            .clone()
            .unwrap_or_else(|| vec![1.0 / n as f32; n]);
        let performance = vec![DefensePerformance::default(); n];

        Self {
            config,
            current_weights: weights,
            performance,
        }
    }

    /// Run the ensemble aggregation.
    pub fn aggregate(&self, updates: &[GradientUpdate]) -> Result<EnsembleResult, EnsembleError> {
        if updates.is_empty() {
            return Err(EnsembleError::NoUpdates);
        }

        // Run each aggregation method
        let mut results: HashMap<String, Vec<f32>> = HashMap::new();
        let mut successful_indices: Vec<usize> = Vec::new();

        for (i, method) in self.config.methods.iter().enumerate() {
            match self.run_method(method, updates) {
                Ok(result) => {
                    results.insert(format!("{:?}", method), result);
                    successful_indices.push(i);
                }
                Err(_) => {
                    // Continue with other defenses
                }
            }
        }

        if results.is_empty() {
            return Err(EnsembleError::AllDefensesFailed);
        }

        // Agreement score
        let agreement = self.calculate_agreement(&results);

        // Byzantine detection by comparing each update against each defense's result
        let (flagged_nodes, consensus_byzantine) = self.detect_byzantine(updates, &results);

        // Combine results
        let aggregated = self.combine_results(&results, &successful_indices)?;

        Ok(EnsembleResult {
            aggregated,
            individual_results: results,
            agreement_score: agreement,
            flagged_nodes,
            consensus_byzantine,
        })
    }

    /// Run a single aggregation method.
    fn run_method(
        &self,
        method: &AggregationMethod,
        updates: &[GradientUpdate],
    ) -> Result<Vec<f32>, AggregationError> {
        match method {
            AggregationMethod::FedAvg => aggregation::fedavg(updates),
            AggregationMethod::TrimmedMean => aggregation::trimmed_mean(updates, 0.1),
            AggregationMethod::Median => aggregation::coordinate_median(updates),
            AggregationMethod::Krum => aggregation::krum(updates, 1),
            AggregationMethod::MultiKrum => aggregation::multi_krum(updates, 1, 3),
            AggregationMethod::GeometricMedian => aggregation::geometric_median(updates, 100, 1e-6),
            AggregationMethod::TrustWeighted => aggregation::fedavg(updates),
        }
    }

    /// Calculate agreement score (pairwise cosine similarity mapped to [0,1]).
    fn calculate_agreement(&self, results: &HashMap<String, Vec<f32>>) -> f32 {
        let outputs: Vec<&Vec<f32>> = results.values().collect();
        if outputs.len() < 2 {
            return 1.0;
        }

        let mut total = 0.0_f32;
        let mut count = 0_usize;

        for i in 0..outputs.len() {
            for j in (i + 1)..outputs.len() {
                let sim = cosine_similarity(outputs[i], outputs[j]);
                total += (sim + 1.0) / 2.0;
                count += 1;
            }
        }

        if count == 0 {
            1.0
        } else {
            total / count as f32
        }
    }

    /// Detect Byzantine nodes using outlier distance from each defense result.
    fn detect_byzantine(
        &self,
        updates: &[GradientUpdate],
        results: &HashMap<String, Vec<f32>>,
    ) -> (HashMap<String, f32>, Vec<String>) {
        let n_defenses = results.len();
        if n_defenses == 0 || updates.is_empty() {
            return (HashMap::new(), Vec::new());
        }

        // For each defense result, detect outliers by distance from result
        let mut flag_counts: HashMap<String, usize> = HashMap::new();

        for result in results.values() {
            let distances: Vec<(usize, f32)> = updates
                .iter()
                .enumerate()
                .map(|(i, u)| {
                    let dist = euclidean_distance(&u.gradients, result);
                    (i, dist)
                })
                .collect();

            // Calculate median distance for threshold
            let mut dist_vals: Vec<f32> = distances.iter().map(|(_, d)| *d).collect();
            dist_vals.sort_by(|a, b| a.total_cmp(b));
            let median_dist = dist_vals[dist_vals.len() / 2];

            // Flag nodes > 3x median distance
            let threshold = 3.0 * median_dist;
            for (i, dist) in &distances {
                if *dist > threshold {
                    *flag_counts
                        .entry(updates[*i].participant_id.clone())
                        .or_insert(0) += 1;
                }
            }
        }

        // Calculate confidence and consensus
        let mut flagged_nodes = HashMap::new();
        let mut consensus = Vec::new();

        for (node_id, count) in &flag_counts {
            let confidence = *count as f32 / n_defenses as f32;
            flagged_nodes.insert(node_id.clone(), confidence);
            if confidence >= self.config.byzantine_consensus_threshold {
                consensus.push(node_id.clone());
            }
        }

        (flagged_nodes, consensus)
    }

    /// Combine results using the configured voting strategy.
    fn combine_results(
        &self,
        results: &HashMap<String, Vec<f32>>,
        successful_indices: &[usize],
    ) -> Result<Vec<f32>, EnsembleError> {
        let outputs: Vec<&Vec<f32>> = results.values().collect();
        if outputs.is_empty() {
            return Err(EnsembleError::AllDefensesFailed);
        }
        let dim = outputs[0].len();

        match &self.config.voting_strategy {
            VotingStrategy::Majority => Ok(self.combine_majority(&outputs)),
            VotingStrategy::Weighted | VotingStrategy::Adaptive => {
                Ok(self.combine_weighted(&outputs, successful_indices))
            }
            VotingStrategy::Unanimous => Ok(self.combine_unanimous(&outputs, dim)),
            VotingStrategy::Median => Ok(self.combine_median(&outputs, dim)),
            VotingStrategy::Best => Ok(self.combine_best(&outputs)),
        }
    }

    /// Select output most similar to all others.
    fn combine_majority(&self, outputs: &[&Vec<f32>]) -> Vec<f32> {
        if outputs.len() == 1 {
            return outputs[0].clone();
        }

        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;

        for i in 0..outputs.len() {
            let mut sim_sum = 0.0_f32;
            for j in 0..outputs.len() {
                if i != j {
                    sim_sum += cosine_similarity(outputs[i], outputs[j]);
                }
            }
            if sim_sum > best_score {
                best_score = sim_sum;
                best_idx = i;
            }
        }

        outputs[best_idx].clone()
    }

    /// Weighted average.
    fn combine_weighted(&self, outputs: &[&Vec<f32>], successful_indices: &[usize]) -> Vec<f32> {
        let dim = outputs[0].len();
        let mut result = vec![0.0_f32; dim];
        let mut total_weight = 0.0_f32;

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

            for (d, val) in output.iter().enumerate() {
                result[d] += val * weight;
            }
            total_weight += weight;
        }

        if total_weight > 0.0 {
            for v in &mut result {
                *v /= total_weight;
            }
        }

        result
    }

    /// Unanimous: require high agreement, else fall back to median.
    fn combine_unanimous(&self, outputs: &[&Vec<f32>], dim: usize) -> Vec<f32> {
        let agreement = {
            let mut total = 0.0_f32;
            let mut count = 0_usize;
            for i in 0..outputs.len() {
                for j in (i + 1)..outputs.len() {
                    let sim = cosine_similarity(outputs[i], outputs[j]);
                    total += (sim + 1.0) / 2.0;
                    count += 1;
                }
            }
            if count > 0 {
                total / count as f32
            } else {
                1.0
            }
        };

        if agreement < 0.9 {
            return self.combine_median(outputs, dim);
        }

        // Average all outputs
        let n = outputs.len() as f32;
        let mut result = vec![0.0_f32; dim];
        for output in outputs {
            for (d, val) in output.iter().enumerate() {
                result[d] += val;
            }
        }
        for v in &mut result {
            *v /= n;
        }
        result
    }

    /// Coordinate-wise median.
    fn combine_median(&self, outputs: &[&Vec<f32>], dim: usize) -> Vec<f32> {
        let mut result = vec![0.0_f32; dim];
        for d in 0..dim {
            let mut values: Vec<f32> = outputs.iter().map(|o| o[d]).collect();
            values.sort_by(|a, b| a.total_cmp(b));
            let mid = values.len() / 2;
            result[d] = if values.len() % 2 == 0 && values.len() > 1 {
                (values[mid - 1] + values[mid]) / 2.0
            } else {
                values[mid]
            };
        }
        result
    }

    /// Best performing defense (by historical EMA).
    fn combine_best(&self, outputs: &[&Vec<f32>]) -> Vec<f32> {
        let mut best_idx = 0;
        let mut best_perf = 0.0_f32;

        for (i, perf) in self.performance.iter().enumerate() {
            if perf.ema_performance > best_perf && i < outputs.len() {
                best_perf = perf.ema_performance;
                best_idx = i;
            }
        }

        if best_idx < outputs.len() {
            outputs[best_idx].clone()
        } else {
            outputs[0].clone()
        }
    }

    /// Update adaptive weights after a round.
    pub fn record_feedback(&mut self, success: bool) {
        let ema_alpha = 0.1_f32;

        for perf in self.performance.iter_mut() {
            perf.total_rounds += 1;
            if success {
                perf.correct_rounds += 1;
            }

            let accuracy = perf.accuracy();
            perf.ema_performance = ema_alpha * accuracy + (1.0 - ema_alpha) * perf.ema_performance;
        }

        if self.config.voting_strategy == VotingStrategy::Adaptive {
            for (i, perf) in self.performance.iter().enumerate() {
                self.current_weights[i] = perf.ema_performance.max(0.1);
            }
            let total: f32 = self.current_weights.iter().sum();
            if total > 0.0 {
                for w in &mut self.current_weights {
                    *w /= total;
                }
            }
        }
    }

    /// Get current weights.
    pub fn weights(&self) -> &[f32] {
        &self.current_weights
    }

    /// Get config reference.
    pub fn config(&self) -> &EnsembleConfig {
        &self.config
    }
}

/// Implement `ByzantinePlugin` so ensemble can be used in the pipeline.
impl ByzantinePlugin for EnsembleDefense {
    fn analyze(&mut self, updates: &[GradientUpdate]) -> ExternalWeightMap {
        let mut weights = ExternalWeightMap::new();

        if let Ok(result) = self.aggregate(updates) {
            for node_id in &result.consensus_byzantine {
                weights.insert(
                    node_id.clone(),
                    vec![ParticipantWeightAdjustment {
                        weight_multiplier: 0.0,
                        veto: true,
                        source: "ensemble_defense".into(),
                    }],
                );
            }
        }

        weights
    }

    fn name(&self) -> &str {
        "ensemble_defense"
    }

    fn record_outcome(&mut self, _round: u64, _excluded_ids: &[String]) {
        self.record_feedback(true);
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    let dot: f32 = a[..len].iter().zip(&b[..len]).map(|(x, y)| x * y).sum();
    let na = a[..len].iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b[..len].iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-10 || nb < 1e-10 {
        return 0.0;
    }
    dot / (na * nb)
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    a[..len]
        .iter()
        .zip(&b[..len])
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::GradientUpdate;

    fn make_update(id: &str, grads: Vec<f32>) -> GradientUpdate {
        GradientUpdate::new(id.into(), 1, grads, 32, 0.5)
    }

    fn test_updates() -> Vec<GradientUpdate> {
        vec![
            make_update("n1", vec![1.0, 2.0, 3.0]),
            make_update("n2", vec![1.1, 2.1, 3.1]),
            make_update("n3", vec![0.9, 1.9, 2.9]),
            make_update("n4", vec![1.05, 2.05, 3.05]),
            make_update("n5", vec![0.95, 1.95, 2.95]),
        ]
    }

    fn updates_with_byzantine() -> Vec<GradientUpdate> {
        let mut u = test_updates();
        u.push(make_update("byzantine", vec![100.0, 200.0, 300.0]));
        u
    }

    #[test]
    fn test_config_default() {
        let c = EnsembleConfig::default();
        assert_eq!(c.methods.len(), 3);
        assert_eq!(c.voting_strategy, VotingStrategy::Majority);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_config_presets() {
        let robust = EnsembleConfig::robust();
        assert_eq!(robust.methods.len(), 3);
        assert!(robust.validate().is_ok());

        let fast = EnsembleConfig::fast();
        assert_eq!(fast.methods.len(), 2);
        assert!(fast.validate().is_ok());

        let paranoid = EnsembleConfig::paranoid();
        assert_eq!(paranoid.methods.len(), 6);
        assert!(paranoid.validate().is_ok());
    }

    #[test]
    fn test_config_validation_errors() {
        // Empty methods
        let empty = EnsembleConfig {
            methods: vec![],
            ..Default::default()
        };
        assert!(empty.validate().is_err());

        // Mismatched weights
        let mismatch = EnsembleConfig {
            methods: vec![AggregationMethod::Median, AggregationMethod::Krum],
            weights: Some(vec![0.5]),
            ..Default::default()
        };
        assert!(mismatch.validate().is_err());

        // Invalid threshold
        let bad_thresh = EnsembleConfig {
            byzantine_consensus_threshold: 1.5,
            ..Default::default()
        };
        assert!(bad_thresh.validate().is_err());
    }

    #[test]
    fn test_majority_voting() {
        let ensemble = EnsembleDefense::new(EnsembleConfig::robust());
        let updates = test_updates();

        let result = ensemble.aggregate(&updates).unwrap();
        assert!((result.aggregated[0] - 1.0).abs() < 0.5);
        assert!((result.aggregated[1] - 2.0).abs() < 0.5);
        assert!((result.aggregated[2] - 3.0).abs() < 0.5);
        assert!(result.agreement_score > 0.8);
    }

    #[test]
    fn test_median_voting() {
        let config = EnsembleConfig {
            voting_strategy: VotingStrategy::Median,
            ..EnsembleConfig::robust()
        };
        let ensemble = EnsembleDefense::new(config);
        let updates = test_updates();

        let result = ensemble.aggregate(&updates).unwrap();
        assert!((result.aggregated[0] - 1.0).abs() < 0.5);
    }

    #[test]
    fn test_weighted_voting() {
        let config = EnsembleConfig {
            methods: vec![AggregationMethod::FedAvg, AggregationMethod::Median],
            voting_strategy: VotingStrategy::Weighted,
            weights: Some(vec![0.3, 0.7]),
            byzantine_consensus_threshold: 0.5,
        };
        let ensemble = EnsembleDefense::new(config);
        let updates = test_updates();

        let result = ensemble.aggregate(&updates).unwrap();
        assert_eq!(result.aggregated.len(), 3);
    }

    #[test]
    fn test_unanimous_voting() {
        let config = EnsembleConfig {
            methods: vec![AggregationMethod::Median, AggregationMethod::TrimmedMean],
            voting_strategy: VotingStrategy::Unanimous,
            ..Default::default()
        };
        let ensemble = EnsembleDefense::new(config);
        let updates = test_updates();

        let result = ensemble.aggregate(&updates).unwrap();
        assert_eq!(result.aggregated.len(), 3);
    }

    #[test]
    fn test_byzantine_detection() {
        let config = EnsembleConfig {
            byzantine_consensus_threshold: 0.3,
            ..EnsembleConfig::robust()
        };
        let ensemble = EnsembleDefense::new(config);
        let updates = updates_with_byzantine();

        let result = ensemble.aggregate(&updates).unwrap();
        let confidence = result
            .flagged_nodes
            .get("byzantine")
            .copied()
            .unwrap_or(0.0);
        assert!(
            confidence > 0.0,
            "Byzantine node should be flagged, got confidence {}",
            confidence
        );
    }

    #[test]
    fn test_ensemble_more_robust_than_fedavg() {
        let updates = updates_with_byzantine();

        let fedavg_result = aggregation::fedavg(&updates).unwrap();

        let ensemble = EnsembleDefense::new(EnsembleConfig::robust());
        let ensemble_result = ensemble.aggregate(&updates).unwrap();

        let expected = vec![1.0_f32, 2.0, 3.0];
        let fedavg_dist = euclidean_distance(&fedavg_result, &expected);
        let ensemble_dist = euclidean_distance(&ensemble_result.aggregated, &expected);

        assert!(
            ensemble_dist < fedavg_dist,
            "Ensemble dist {} should be < FedAvg dist {}",
            ensemble_dist,
            fedavg_dist
        );
    }

    #[test]
    fn test_individual_results_populated() {
        let ensemble = EnsembleDefense::new(EnsembleConfig::robust());
        let updates = test_updates();

        let result = ensemble.aggregate(&updates).unwrap();
        assert_eq!(result.individual_results.len(), 3);
        for (_, grad) in &result.individual_results {
            assert_eq!(grad.len(), 3);
        }
    }

    #[test]
    fn test_adaptive_weight_update() {
        let config = EnsembleConfig {
            voting_strategy: VotingStrategy::Adaptive,
            ..EnsembleConfig::robust()
        };
        let mut ensemble = EnsembleDefense::new(config);

        let initial: Vec<f32> = ensemble.weights().to_vec();
        assert_eq!(initial.len(), 3);

        ensemble.record_feedback(true);

        let sum: f32 = ensemble.weights().iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Weights should sum to ~1.0, got {}",
            sum
        );
    }

    #[test]
    fn test_empty_updates_error() {
        let ensemble = EnsembleDefense::new(EnsembleConfig::default());
        assert!(ensemble.aggregate(&[]).is_err());
    }

    #[test]
    fn test_byzantine_plugin_trait() {
        let mut ensemble = EnsembleDefense::new(EnsembleConfig {
            byzantine_consensus_threshold: 0.3,
            ..EnsembleConfig::robust()
        });

        let updates = updates_with_byzantine();
        let weight_map = ensemble.analyze(&updates);

        assert_eq!(ensemble.name(), "ensemble_defense");
        if let Some(adjustments) = weight_map.get("byzantine") {
            assert!(adjustments.iter().any(|a| a.veto));
        }
    }

    #[test]
    fn test_cosine_similarity_helper() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-5);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 1e-5);

        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) + 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_euclidean_distance_helper() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];
        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 1e-5);
    }
}
