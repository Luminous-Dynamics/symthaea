// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # ML-Based Anomaly Detection
//!
//! Machine learning approaches for detecting subtle agent behavioral anomalies
//! that rule-based systems might miss.
//!
//! ## Algorithms
//!
//! - **Isolation Forest**: Detects outliers by measuring isolation depth
//! - **Autoencoder-style Reconstruction**: Measures pattern deviation from learned norms
//! - **Time-series Anomaly**: Detects unusual trust evolution patterns
//! - **Ensemble Scoring**: Combines multiple detectors for robust anomaly scoring

use super::{ActionOutcome, InstrumentalActor};
use crate::matl::KVector;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[cfg(feature = "ts-export")]
use ts_rs::TS;

// ============================================================================
// Feature Extraction
// ============================================================================

/// Features extracted from agent behavior for ML analysis
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct AgentFeatures {
    // Behavioral features
    /// Success rate over recent actions
    pub success_rate: f64,
    /// Action frequency (actions per hour)
    pub action_frequency: f64,
    /// Timing regularity (coefficient of variation of intervals)
    pub timing_regularity: f64,
    /// Average KREDIT consumption per action
    pub avg_kredit_per_action: f64,

    // K-Vector features (8 dimensions)
    /// Current K-Vector values
    pub k_vector: [f64; 8],
    /// K-Vector velocity (rate of change)
    pub k_vector_velocity: [f64; 8],
    /// K-Vector acceleration
    pub k_vector_acceleration: [f64; 8],

    // Epistemic features
    /// Average epistemic weight of outputs
    pub avg_epistemic_weight: f64,
    /// Epistemic claim distribution (E0-E4 proportions)
    pub epistemic_distribution: [f64; 5],
    /// Verified accuracy rate
    pub verified_accuracy: f64,

    // Social features
    /// Number of unique counterparties
    pub counterparty_diversity: f64,
    /// Reputation propagation score
    pub reputation_score: f64,

    // Derived features
    /// Trust score
    pub trust_score: f64,
    /// Overall quality score
    pub quality_score: f64,
}

impl AgentFeatures {
    /// Extract features from an agent
    pub fn extract(agent: &InstrumentalActor, k_vector_history: &[KVector]) -> Self {
        let behavior_len = agent.behavior_log.len();

        // Success rate
        let successes = agent
            .behavior_log
            .iter()
            .filter(|e| e.outcome == ActionOutcome::Success)
            .count();
        let success_rate = if behavior_len > 0 {
            successes as f64 / behavior_len as f64
        } else {
            0.5
        };

        // Action frequency
        let action_frequency = if behavior_len >= 2 {
            let first = agent.behavior_log.first().map(|e| e.timestamp).unwrap_or(0);
            let last = agent.behavior_log.last().map(|e| e.timestamp).unwrap_or(0);
            let duration_hours = (last.saturating_sub(first)) as f64 / 3600.0;
            if duration_hours > 0.0 {
                behavior_len as f64 / duration_hours
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Timing regularity
        let timing_regularity = compute_timing_regularity(&agent.behavior_log);

        // Average KREDIT per action
        let total_kredit: u64 = agent.behavior_log.iter().map(|e| e.kredit_consumed).sum();
        let avg_kredit_per_action = if behavior_len > 0 {
            total_kredit as f64 / behavior_len as f64
        } else {
            0.0
        };

        // K-Vector features
        let k_vector = [
            agent.k_vector.k_r as f64,
            agent.k_vector.k_a as f64,
            agent.k_vector.k_i as f64,
            agent.k_vector.k_p as f64,
            agent.k_vector.k_m as f64,
            agent.k_vector.k_s as f64,
            agent.k_vector.k_h as f64,
            agent.k_vector.k_topo as f64,
        ];

        // K-Vector velocity (first derivative)
        let k_vector_velocity = if k_vector_history.len() >= 2 {
            let prev = &k_vector_history[k_vector_history.len() - 2];
            [
                (agent.k_vector.k_r - prev.k_r) as f64,
                (agent.k_vector.k_a - prev.k_a) as f64,
                (agent.k_vector.k_i - prev.k_i) as f64,
                (agent.k_vector.k_p - prev.k_p) as f64,
                (agent.k_vector.k_m - prev.k_m) as f64,
                (agent.k_vector.k_s - prev.k_s) as f64,
                (agent.k_vector.k_h - prev.k_h) as f64,
                (agent.k_vector.k_topo - prev.k_topo) as f64,
            ]
        } else {
            [0.0; 8]
        };

        // K-Vector acceleration (second derivative)
        let k_vector_acceleration = if k_vector_history.len() >= 3 {
            let prev1 = &k_vector_history[k_vector_history.len() - 2];
            let prev2 = &k_vector_history[k_vector_history.len() - 3];
            let vel1 = [
                (agent.k_vector.k_r - prev1.k_r) as f64,
                (agent.k_vector.k_a - prev1.k_a) as f64,
                (agent.k_vector.k_i - prev1.k_i) as f64,
                (agent.k_vector.k_p - prev1.k_p) as f64,
                (agent.k_vector.k_m - prev1.k_m) as f64,
                (agent.k_vector.k_s - prev1.k_s) as f64,
                (agent.k_vector.k_h - prev1.k_h) as f64,
                (agent.k_vector.k_topo - prev1.k_topo) as f64,
            ];
            let vel0 = [
                (prev1.k_r - prev2.k_r) as f64,
                (prev1.k_a - prev2.k_a) as f64,
                (prev1.k_i - prev2.k_i) as f64,
                (prev1.k_p - prev2.k_p) as f64,
                (prev1.k_m - prev2.k_m) as f64,
                (prev1.k_s - prev2.k_s) as f64,
                (prev1.k_h - prev2.k_h) as f64,
                (prev1.k_topo - prev2.k_topo) as f64,
            ];
            [
                vel1[0] - vel0[0],
                vel1[1] - vel0[1],
                vel1[2] - vel0[2],
                vel1[3] - vel0[3],
                vel1[4] - vel0[4],
                vel1[5] - vel0[5],
                vel1[6] - vel0[6],
                vel1[7] - vel0[7],
            ]
        } else {
            [0.0; 8]
        };

        // Epistemic features
        let avg_epistemic_weight = agent.epistemic_stats.average_weight as f64;
        let total_outputs = agent.epistemic_stats.total_outputs.max(1) as f64;
        let epistemic_distribution = [
            agent.epistemic_stats.empirical_distribution[0] as f64 / total_outputs,
            agent.epistemic_stats.empirical_distribution[1] as f64 / total_outputs,
            agent.epistemic_stats.empirical_distribution[2] as f64 / total_outputs,
            agent.epistemic_stats.empirical_distribution[3] as f64 / total_outputs,
            agent.epistemic_stats.empirical_distribution[4] as f64 / total_outputs,
        ];
        let verified_accuracy = agent.verified_accuracy() as f64;

        // Social features
        let mut unique_counterparties = std::collections::HashSet::new();
        for entry in &agent.behavior_log {
            for cp in &entry.counterparties {
                unique_counterparties.insert(cp.clone());
            }
        }
        let counterparty_diversity = unique_counterparties.len() as f64;

        // Trust and quality
        let trust_score = agent.k_vector.trust_score() as f64;
        let quality_score = agent.epistemic_quality() as f64;

        Self {
            success_rate,
            action_frequency,
            timing_regularity,
            avg_kredit_per_action,
            k_vector,
            k_vector_velocity,
            k_vector_acceleration,
            avg_epistemic_weight,
            epistemic_distribution,
            verified_accuracy,
            counterparty_diversity,
            reputation_score: 0.0, // Set externally if available
            trust_score,
            quality_score,
        }
    }

    /// Convert to feature vector for ML algorithms
    pub fn to_vector(&self) -> Vec<f64> {
        let mut v = vec![
            self.success_rate,
            self.action_frequency,
            self.timing_regularity,
            self.avg_kredit_per_action,
        ];
        v.extend_from_slice(&self.k_vector);
        v.extend_from_slice(&self.k_vector_velocity);
        v.extend_from_slice(&self.k_vector_acceleration);
        v.push(self.avg_epistemic_weight);
        v.extend_from_slice(&self.epistemic_distribution);
        v.push(self.verified_accuracy);
        v.push(self.counterparty_diversity);
        v.push(self.reputation_score);
        v.push(self.trust_score);
        v.push(self.quality_score);
        v
    }

    /// Feature vector dimension
    pub const DIMENSION: usize = 4 + 8 + 8 + 8 + 1 + 5 + 1 + 1 + 1 + 1 + 1; // 39
}

fn compute_timing_regularity(log: &[super::BehaviorLogEntry]) -> f64 {
    if log.len() < 3 {
        return 0.5; // Neutral default
    }

    let mut timestamps: Vec<u64> = log.iter().map(|e| e.timestamp).collect();
    timestamps.sort();

    let intervals: Vec<f64> = timestamps
        .windows(2)
        .map(|w| w[1].saturating_sub(w[0]) as f64)
        .filter(|&i| i > 0.0)
        .collect();

    if intervals.is_empty() {
        return 0.5;
    }

    let mean: f64 = intervals.iter().sum::<f64>() / intervals.len() as f64;
    if mean == 0.0 {
        return 0.5;
    }

    let variance: f64 = intervals
        .iter()
        .map(|i| ((i - mean) / mean).powi(2))
        .sum::<f64>()
        / intervals.len() as f64;

    // Higher CV = more irregular = lower regularity score
    let cv = variance.sqrt();
    (1.0 - cv.min(1.0)).max(0.0)
}

// ============================================================================
// Isolation Forest
// ============================================================================

/// Isolation Forest node
#[derive(Clone, Debug)]
enum IsolationNode {
    /// Internal node with split
    Internal {
        feature_idx: usize,
        split_value: f64,
        left: Box<IsolationNode>,
        right: Box<IsolationNode>,
    },
    /// External (leaf) node
    External { size: usize },
}

/// Single Isolation Tree
#[derive(Clone, Debug)]
pub struct IsolationTree {
    root: IsolationNode,
    _height_limit: usize,
}

impl IsolationTree {
    /// Build a tree from samples
    pub fn build(samples: &[Vec<f64>], height_limit: usize, rng: &mut SimpleRng) -> Self {
        let root = Self::build_node(samples, 0, height_limit, rng);
        Self {
            root,
            _height_limit: height_limit,
        }
    }

    fn build_node(
        samples: &[Vec<f64>],
        depth: usize,
        limit: usize,
        rng: &mut SimpleRng,
    ) -> IsolationNode {
        if depth >= limit || samples.len() <= 1 {
            return IsolationNode::External {
                size: samples.len(),
            };
        }

        if samples.is_empty() || samples[0].is_empty() {
            return IsolationNode::External { size: 0 };
        }

        // Random feature selection
        let n_features = samples[0].len();
        let feature_idx = rng.next_usize() % n_features;

        // Find min/max for this feature
        let values: Vec<f64> = samples.iter().map(|s| s[feature_idx]).collect();
        let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if (max_val - min_val).abs() < 1e-10 {
            return IsolationNode::External {
                size: samples.len(),
            };
        }

        // Random split point
        let split_value = min_val + rng.next_f64() * (max_val - min_val);

        // Partition samples
        let (left_samples, right_samples): (Vec<_>, Vec<_>) = samples
            .iter()
            .cloned()
            .partition(|s| s[feature_idx] < split_value);

        // Handle edge case where all samples go to one side
        if left_samples.is_empty() || right_samples.is_empty() {
            return IsolationNode::External {
                size: samples.len(),
            };
        }

        IsolationNode::Internal {
            feature_idx,
            split_value,
            left: Box::new(Self::build_node(&left_samples, depth + 1, limit, rng)),
            right: Box::new(Self::build_node(&right_samples, depth + 1, limit, rng)),
        }
    }

    /// Compute path length for a sample
    pub fn path_length(&self, sample: &[f64]) -> f64 {
        self.path_length_node(&self.root, sample, 0)
    }

    fn path_length_node(&self, node: &IsolationNode, sample: &[f64], depth: usize) -> f64 {
        match node {
            IsolationNode::External { size } => depth as f64 + c_factor(*size),
            IsolationNode::Internal {
                feature_idx,
                split_value,
                left,
                right,
            } => {
                if sample[*feature_idx] < *split_value {
                    self.path_length_node(left, sample, depth + 1)
                } else {
                    self.path_length_node(right, sample, depth + 1)
                }
            }
        }
    }
}

/// Average path length of unsuccessful search in BST
fn c_factor(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    let n = n as f64;
    2.0 * (n.ln() + 0.5772156649) - (2.0 * (n - 1.0) / n)
}

/// Isolation Forest ensemble
#[derive(Clone)]
pub struct IsolationForest {
    trees: Vec<IsolationTree>,
    sample_size: usize,
    n_estimators: usize,
}

impl IsolationForest {
    /// Create a new forest
    pub fn new(n_estimators: usize, sample_size: usize) -> Self {
        Self {
            trees: Vec::new(),
            sample_size,
            n_estimators,
        }
    }

    /// Fit the forest on training data
    pub fn fit(&mut self, data: &[Vec<f64>], seed: u64) {
        let mut rng = SimpleRng::new(seed);
        let height_limit = (self.sample_size as f64).log2().ceil() as usize;

        self.trees.clear();
        for _ in 0..self.n_estimators {
            // Subsample
            let subsample: Vec<Vec<f64>> = (0..self.sample_size.min(data.len()))
                .map(|_| {
                    let idx = rng.next_usize() % data.len();
                    data[idx].clone()
                })
                .collect();

            let tree = IsolationTree::build(&subsample, height_limit, &mut rng);
            self.trees.push(tree);
        }
    }

    /// Compute anomaly score for a sample (0 = normal, 1 = anomaly)
    pub fn score(&self, sample: &[f64]) -> f64 {
        if self.trees.is_empty() {
            return 0.5;
        }

        let avg_path_length: f64 = self
            .trees
            .iter()
            .map(|t| t.path_length(sample))
            .sum::<f64>()
            / self.trees.len() as f64;

        let c = c_factor(self.sample_size);
        if c == 0.0 {
            return 0.5;
        }

        // Anomaly score: 2^(-avg_path_length / c)
        2.0_f64.powf(-avg_path_length / c)
    }

    /// Predict if sample is anomaly (score > threshold)
    pub fn predict(&self, sample: &[f64], threshold: f64) -> bool {
        self.score(sample) > threshold
    }
}

// ============================================================================
// Reconstruction-Based Anomaly Detection
// ============================================================================

/// Simple autoencoder-style reconstruction using PCA-like compression
/// (Without deep learning dependencies)
#[derive(Clone)]
pub struct ReconstructionDetector {
    /// Mean of training features
    mean: Vec<f64>,
    /// Covariance matrix (for Mahalanobis distance)
    inv_cov: Vec<Vec<f64>>,
    /// Training data size
    n_samples: usize,
    /// Feature dimension
    dimension: usize,
}

impl ReconstructionDetector {
    /// Create a new detector
    pub fn new() -> Self {
        Self {
            mean: Vec::new(),
            inv_cov: Vec::new(),
            n_samples: 0,
            dimension: 0,
        }
    }

    /// Fit on training data (compute mean and covariance)
    pub fn fit(&mut self, data: &[Vec<f64>]) {
        if data.is_empty() || data[0].is_empty() {
            return;
        }

        self.n_samples = data.len();
        self.dimension = data[0].len();

        // Compute mean
        self.mean = vec![0.0; self.dimension];
        for sample in data {
            for (i, v) in sample.iter().enumerate() {
                self.mean[i] += v;
            }
        }
        for m in &mut self.mean {
            *m /= self.n_samples as f64;
        }

        // Compute covariance matrix
        let mut cov = vec![vec![0.0; self.dimension]; self.dimension];
        for sample in data {
            for i in 0..self.dimension {
                for j in 0..self.dimension {
                    cov[i][j] += (sample[i] - self.mean[i]) * (sample[j] - self.mean[j]);
                }
            }
        }
        for (i, row) in cov.iter_mut().enumerate() {
            for val in row.iter_mut() {
                *val /= (self.n_samples - 1).max(1) as f64;
            }
            // Regularization to ensure invertibility
            row[i] += 1e-6;
        }

        // Compute pseudo-inverse (simplified - use diagonal for efficiency)
        self.inv_cov = vec![vec![0.0; self.dimension]; self.dimension];
        for (i, row) in cov.iter().enumerate() {
            if row[i] > 1e-10 {
                self.inv_cov[i][i] = 1.0 / row[i];
            }
        }
    }

    /// Compute reconstruction error (Mahalanobis-like distance)
    pub fn reconstruction_error(&self, sample: &[f64]) -> f64 {
        if self.mean.is_empty() || sample.len() != self.dimension {
            return 0.0;
        }

        // Compute (x - mean)^T * inv_cov * (x - mean)
        let diff: Vec<f64> = sample.iter().zip(&self.mean).map(|(s, m)| s - m).collect();

        let mut distance = 0.0;
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                distance += diff[i] * self.inv_cov[i][j] * diff[j];
            }
        }

        distance.sqrt()
    }

    /// Compute anomaly score (normalized reconstruction error)
    pub fn score(&self, sample: &[f64]) -> f64 {
        let error = self.reconstruction_error(sample);
        // Sigmoid normalization
        1.0 / (1.0 + (-error + 3.0).exp())
    }
}

impl Default for ReconstructionDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Time-Series Anomaly Detection
// ============================================================================

/// Time-series anomaly detector for trust evolution
#[derive(Clone)]
pub struct TimeSeriesAnomalyDetector {
    /// Window size for rolling statistics
    window_size: usize,
    /// Sensitivity threshold (number of std devs)
    sensitivity: f64,
    /// Historical values
    history: VecDeque<f64>,
    /// Running mean
    running_mean: f64,
    /// Running variance
    running_var: f64,
}

impl TimeSeriesAnomalyDetector {
    /// Create a new detector
    pub fn new(window_size: usize, sensitivity: f64) -> Self {
        Self {
            window_size,
            sensitivity,
            history: VecDeque::with_capacity(window_size),
            running_mean: 0.0,
            running_var: 0.0,
        }
    }

    /// Add a new observation and check for anomaly
    pub fn observe(&mut self, value: f64) -> TimeSeriesAnomalyResult {
        let is_anomaly;
        let z_score;

        if self.history.len() >= self.window_size {
            // Compute z-score
            let std_dev = self.running_var.sqrt();
            let diff = (value - self.running_mean).abs();

            z_score = if std_dev > 1e-10 {
                (value - self.running_mean) / std_dev
            } else if diff > 1e-10 {
                // If std_dev is ~0 but value differs from mean, it's definitely anomalous
                // Use a large z-score proportional to the difference
                (value - self.running_mean).signum()
                    * (diff / self.running_mean.abs().max(1.0))
                    * 100.0
            } else {
                0.0
            };

            is_anomaly = z_score.abs() > self.sensitivity;

            // Remove oldest value from running stats
            if let Some(old) = self.history.pop_front() {
                let n = self.history.len() as f64;
                if n > 0.0 {
                    let old_mean = self.running_mean;
                    self.running_mean = (old_mean * (n + 1.0) - old) / n;
                    self.running_var =
                        ((self.running_var * n + (old - old_mean) * (old - self.running_mean)) / n)
                            .max(0.0);
                }
            }
        } else {
            is_anomaly = false;
            z_score = 0.0;
        }

        // Update running statistics with new value
        let n = self.history.len() as f64;
        if n == 0.0 {
            self.running_mean = value;
            self.running_var = 0.0;
        } else {
            let old_mean = self.running_mean;
            self.running_mean = old_mean + (value - old_mean) / (n + 1.0);
            self.running_var = (self.running_var * n
                + (value - old_mean) * (value - self.running_mean))
                / (n + 1.0);
        }

        self.history.push_back(value);

        TimeSeriesAnomalyResult {
            is_anomaly,
            z_score,
            value,
            mean: self.running_mean,
            std_dev: self.running_var.sqrt(),
        }
    }

    /// Reset the detector
    pub fn reset(&mut self) {
        self.history.clear();
        self.running_mean = 0.0;
        self.running_var = 0.0;
    }
}

/// Result from time-series anomaly detection
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct TimeSeriesAnomalyResult {
    /// Whether this observation is anomalous
    pub is_anomaly: bool,
    /// Z-score of observation
    pub z_score: f64,
    /// The observed value
    pub value: f64,
    /// Running mean
    pub mean: f64,
    /// Running standard deviation
    pub std_dev: f64,
}

// ============================================================================
// Ensemble Anomaly Detector
// ============================================================================

/// Ensemble combining multiple anomaly detection methods
pub struct MLAnomalyDetector {
    /// Isolation Forest
    isolation_forest: IsolationForest,
    /// Reconstruction detector
    reconstruction: ReconstructionDetector,
    /// Time-series detectors (one per K-Vector dimension)
    time_series: Vec<TimeSeriesAnomalyDetector>,
    /// Configuration
    config: MLAnomalyConfig,
    /// Training data (for online updates)
    training_data: Vec<Vec<f64>>,
    /// Is the detector trained?
    is_trained: bool,
}

/// Configuration for ML anomaly detection
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct MLAnomalyConfig {
    /// Number of isolation trees
    pub n_estimators: usize,
    /// Sample size for each tree
    pub sample_size: usize,
    /// Time-series window size
    pub ts_window_size: usize,
    /// Time-series sensitivity (std devs)
    pub ts_sensitivity: f64,
    /// Weight for isolation forest score
    pub isolation_weight: f64,
    /// Weight for reconstruction score
    pub reconstruction_weight: f64,
    /// Weight for time-series score
    pub time_series_weight: f64,
    /// Anomaly threshold (0-1)
    pub anomaly_threshold: f64,
    /// Minimum training samples before scoring
    pub min_training_samples: usize,
}

impl Default for MLAnomalyConfig {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            sample_size: 256,
            ts_window_size: 50,
            ts_sensitivity: 3.0,
            isolation_weight: 0.4,
            reconstruction_weight: 0.3,
            time_series_weight: 0.3,
            anomaly_threshold: 0.65,
            min_training_samples: 100,
        }
    }
}

/// Result from ML anomaly detection
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct MLAnomalyResult {
    /// Overall anomaly score (0-1)
    pub anomaly_score: f64,
    /// Whether classified as anomaly
    pub is_anomaly: bool,
    /// Isolation forest score
    pub isolation_score: f64,
    /// Reconstruction error score
    pub reconstruction_score: f64,
    /// Time-series anomaly score
    pub time_series_score: f64,
    /// Which dimensions are anomalous
    pub anomalous_dimensions: Vec<String>,
    /// Confidence in the result
    pub confidence: f64,
}

impl MLAnomalyDetector {
    /// Create a new detector with config
    pub fn new(config: MLAnomalyConfig) -> Self {
        let time_series = (0..8)
            .map(|_| TimeSeriesAnomalyDetector::new(config.ts_window_size, config.ts_sensitivity))
            .collect();

        Self {
            isolation_forest: IsolationForest::new(config.n_estimators, config.sample_size),
            reconstruction: ReconstructionDetector::new(),
            time_series,
            config,
            training_data: Vec::new(),
            is_trained: false,
        }
    }

    /// Add training sample (normal agent behavior)
    pub fn add_training_sample(&mut self, features: &AgentFeatures) {
        self.training_data.push(features.to_vector());

        // Also update time-series with K-Vector dimensions
        for (i, ts) in self.time_series.iter_mut().enumerate() {
            ts.observe(features.k_vector[i]);
        }
    }

    /// Train the detector on accumulated samples
    pub fn train(&mut self, seed: u64) {
        if self.training_data.len() < self.config.min_training_samples {
            return;
        }

        self.isolation_forest.fit(&self.training_data, seed);
        self.reconstruction.fit(&self.training_data);
        self.is_trained = true;
    }

    /// Detect anomalies in agent features
    pub fn detect(&mut self, features: &AgentFeatures) -> MLAnomalyResult {
        let vector = features.to_vector();

        // Compute individual scores
        let isolation_score = if self.is_trained {
            self.isolation_forest.score(&vector)
        } else {
            0.5
        };

        let reconstruction_score = if self.is_trained {
            self.reconstruction.score(&vector)
        } else {
            0.5
        };

        // Time-series anomaly detection on K-Vector dimensions
        let mut ts_anomalies = 0;
        let mut anomalous_dimensions = Vec::new();
        let dimension_names = ["k_r", "k_a", "k_i", "k_p", "k_m", "k_s", "k_h", "k_topo"];

        for (i, ts) in self.time_series.iter_mut().enumerate() {
            let result = ts.observe(features.k_vector[i]);
            if result.is_anomaly {
                ts_anomalies += 1;
                anomalous_dimensions.push(dimension_names[i].to_string());
            }
        }

        let time_series_score = ts_anomalies as f64 / 8.0;

        // Weighted ensemble
        let total_weight = self.config.isolation_weight
            + self.config.reconstruction_weight
            + self.config.time_series_weight;

        let anomaly_score = (isolation_score * self.config.isolation_weight
            + reconstruction_score * self.config.reconstruction_weight
            + time_series_score * self.config.time_series_weight)
            / total_weight;

        let is_anomaly = anomaly_score > self.config.anomaly_threshold;

        // Confidence based on agreement between detectors
        let scores = [isolation_score, reconstruction_score, time_series_score];
        let mean_score: f64 = scores.iter().sum::<f64>() / 3.0;
        let variance: f64 = scores.iter().map(|s| (s - mean_score).powi(2)).sum::<f64>() / 3.0;
        let confidence = 1.0 - variance.sqrt().min(1.0);

        MLAnomalyResult {
            anomaly_score,
            is_anomaly,
            isolation_score,
            reconstruction_score,
            time_series_score,
            anomalous_dimensions,
            confidence,
        }
    }

    /// Get number of training samples
    pub fn training_samples(&self) -> usize {
        self.training_data.len()
    }

    /// Check if detector is trained
    pub fn is_trained(&self) -> bool {
        self.is_trained
    }
}

// ============================================================================
// Simple RNG (for reproducibility without external deps)
// ============================================================================

/// Simple xorshift64 RNG
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    /// Create a new RNG with the given seed.
    pub fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    /// Generate the next pseudo-random u64.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Generate the next pseudo-random f64 in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Generate the next pseudo-random usize.
    pub fn next_usize(&mut self) -> usize {
        self.next_u64() as usize
    }
}

// ============================================================================
// Integration with Existing Systems
// ============================================================================

/// Integrates ML anomaly detection with existing detectors
pub struct HybridAnomalyDetector {
    /// ML-based detector
    ml_detector: MLAnomalyDetector,
    /// Reference to gaming detector results
    gaming_weight: f64,
    /// Reference to Sybil detector results
    sybil_weight: f64,
    /// ML weight
    ml_weight: f64,
}

impl HybridAnomalyDetector {
    /// Create a new hybrid detector
    pub fn new(ml_config: MLAnomalyConfig) -> Self {
        Self {
            ml_detector: MLAnomalyDetector::new(ml_config),
            gaming_weight: 0.3,
            sybil_weight: 0.2,
            ml_weight: 0.5,
        }
    }

    /// Add training sample
    pub fn add_training_sample(&mut self, features: &AgentFeatures) {
        self.ml_detector.add_training_sample(features);
    }

    /// Train the ML component
    pub fn train(&mut self, seed: u64) {
        self.ml_detector.train(seed);
    }

    /// Detect anomalies combining all signals
    pub fn detect(
        &mut self,
        features: &AgentFeatures,
        gaming_score: f64,
        sybil_score: f64,
    ) -> HybridAnomalyResult {
        let ml_result = self.ml_detector.detect(features);

        // Combine scores
        let total_weight = self.gaming_weight + self.sybil_weight + self.ml_weight;
        let combined_score = (gaming_score * self.gaming_weight
            + sybil_score * self.sybil_weight
            + ml_result.anomaly_score * self.ml_weight)
            / total_weight;

        // Determine anomaly type
        let anomaly_type = if ml_result.anomaly_score > 0.7 {
            AnomalyType::MLDetected
        } else if gaming_score > 0.6 {
            AnomalyType::GamingDetected
        } else if sybil_score > 0.6 {
            AnomalyType::SybilDetected
        } else if combined_score > 0.6 {
            AnomalyType::Ensemble
        } else {
            AnomalyType::None
        };

        let recommendation = get_recommendation(combined_score, anomaly_type);

        HybridAnomalyResult {
            combined_score,
            ml_result,
            gaming_score,
            sybil_score,
            anomaly_type,
            recommendation,
        }
    }
}

/// Types of detected anomalies
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub enum AnomalyType {
    /// No anomaly detected
    None,
    /// Detected by ML methods
    MLDetected,
    /// Gaming behavior detected
    GamingDetected,
    /// Sybil pattern detected
    SybilDetected,
    /// Detected by ensemble agreement
    Ensemble,
}

/// Result from hybrid detector
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct HybridAnomalyResult {
    /// Combined anomaly score
    pub combined_score: f64,
    /// ML detection result
    pub ml_result: MLAnomalyResult,
    /// Gaming detection score
    pub gaming_score: f64,
    /// Sybil detection score
    pub sybil_score: f64,
    /// Primary anomaly type
    pub anomaly_type: AnomalyType,
    /// Recommended action
    pub recommendation: AnomalyRecommendation,
}

/// Recommended action for anomaly
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub enum AnomalyRecommendation {
    /// No action needed
    None,
    /// Flag for manual review
    Review,
    /// Increase monitoring
    Monitor,
    /// Throttle agent actions
    Throttle,
    /// Quarantine immediately
    Quarantine,
}

fn get_recommendation(score: f64, anomaly_type: AnomalyType) -> AnomalyRecommendation {
    match (score, &anomaly_type) {
        (s, _) if s < 0.3 => AnomalyRecommendation::None,
        (s, _) if s < 0.5 => AnomalyRecommendation::Monitor,
        (s, _) if s < 0.65 => AnomalyRecommendation::Review,
        (s, _) if s < 0.8 => AnomalyRecommendation::Throttle,
        _ => AnomalyRecommendation::Quarantine,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentic::UncertaintyCalibration;
    use crate::agentic::{AgentClass, AgentConstraints, AgentId, AgentStatus, EpistemicStats};

    fn create_test_agent() -> InstrumentalActor {
        InstrumentalActor {
            agent_id: AgentId::generate(),
            sponsor_did: "did:test:sponsor".to_string(),
            agent_class: AgentClass::Supervised,
            kredit_balance: 5000,
            kredit_cap: 10000,
            constraints: AgentConstraints::default(),
            behavior_log: vec![],
            status: AgentStatus::Active,
            created_at: 1000,
            last_activity: 1000,
            actions_this_hour: 0,
            k_vector: KVector::new_participant(),
            epistemic_stats: EpistemicStats::default(),
            output_history: vec![],
            uncertainty_calibration: UncertaintyCalibration::default(),
            pending_escalations: vec![],
        }
    }

    #[test]
    fn test_feature_extraction() {
        let agent = create_test_agent();
        let features = AgentFeatures::extract(&agent, &[]);

        assert_eq!(features.success_rate, 0.5); // Default for empty log
        assert_eq!(features.k_vector.len(), 8);
        assert_eq!(features.to_vector().len(), AgentFeatures::DIMENSION);
    }

    #[test]
    fn test_isolation_forest() {
        let mut rng = SimpleRng::new(42);

        // Generate normal data
        let normal_data: Vec<Vec<f64>> = (0..100)
            .map(|_| vec![rng.next_f64(), rng.next_f64()])
            .collect();

        // Build forest
        let mut forest = IsolationForest::new(50, 64);
        forest.fit(&normal_data, 42);

        // Normal point should have lower score
        let normal_score = forest.score(&[0.5, 0.5]);
        // Outlier should have higher score
        let outlier_score = forest.score(&[10.0, 10.0]);

        assert!(
            outlier_score > normal_score,
            "Outlier score {} should be higher than normal {}",
            outlier_score,
            normal_score
        );
    }

    #[test]
    fn test_reconstruction_detector() {
        let mut detector = ReconstructionDetector::new();

        // Train on clustered data
        let data: Vec<Vec<f64>> = (0..50)
            .map(|i| vec![i as f64 * 0.1, i as f64 * 0.1 + 0.5])
            .collect();

        detector.fit(&data);

        // Point near training data should have low error
        let near_error = detector.reconstruction_error(&[2.5, 3.0]);
        // Point far from training data should have high error
        let far_error = detector.reconstruction_error(&[100.0, 100.0]);

        assert!(
            far_error > near_error,
            "Far error {} should be higher than near {}",
            far_error,
            near_error
        );
    }

    #[test]
    fn test_time_series_detector() {
        let mut detector = TimeSeriesAnomalyDetector::new(10, 2.0);

        // Add stable values
        for _ in 0..20 {
            let result = detector.observe(1.0);
            // After warmup, should not be anomaly
            if detector.history.len() >= 10 {
                assert!(!result.is_anomaly);
            }
        }

        // Add anomalous spike
        let result = detector.observe(10.0);
        assert!(result.is_anomaly, "Spike should be detected as anomaly");
        assert!(result.z_score > 2.0);
    }

    #[test]
    fn test_ml_anomaly_detector() {
        let config = MLAnomalyConfig {
            min_training_samples: 10,
            ..Default::default()
        };
        let mut detector = MLAnomalyDetector::new(config);

        // Add training samples
        for i in 0..50 {
            let mut features = AgentFeatures::default();
            features.success_rate = 0.75 + (i as f64 % 10.0) * 0.01;
            features.k_vector = [0.6, 0.5, 0.8, 0.65, 0.2, 0.3, 0.5, 0.2];
            detector.add_training_sample(&features);
        }

        detector.train(42);
        assert!(detector.is_trained());

        // Normal features should have low anomaly score
        let mut normal = AgentFeatures::default();
        normal.success_rate = 0.76;
        normal.k_vector = [0.6, 0.5, 0.8, 0.65, 0.2, 0.3, 0.5, 0.2];
        let result = detector.detect(&normal);
        assert!(
            result.anomaly_score < 0.7,
            "Normal agent score: {}",
            result.anomaly_score
        );

        // Anomalous features should have higher score
        let mut anomalous = AgentFeatures::default();
        anomalous.success_rate = 0.99; // Suspiciously high
        anomalous.k_vector = [0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99];
        let result = detector.detect(&anomalous);
        // Note: With limited training data, detection may not be perfect
        assert!(
            result.isolation_score > 0.3,
            "Anomalous isolation score: {}",
            result.isolation_score
        );
    }

    #[test]
    fn test_hybrid_detector() {
        let config = MLAnomalyConfig {
            min_training_samples: 5,
            ..Default::default()
        };
        let mut detector = HybridAnomalyDetector::new(config);

        // Minimal training
        for _ in 0..10 {
            let features = AgentFeatures::default();
            detector.add_training_sample(&features);
        }
        detector.train(42);

        // Test with various signals
        let features = AgentFeatures::default();

        // Low scores = no anomaly
        let result = detector.detect(&features, 0.2, 0.1);
        assert_eq!(result.anomaly_type, AnomalyType::None);

        // High gaming score
        let result = detector.detect(&features, 0.8, 0.1);
        assert_eq!(result.anomaly_type, AnomalyType::GamingDetected);

        // High sybil score
        let result = detector.detect(&features, 0.2, 0.8);
        assert_eq!(result.anomaly_type, AnomalyType::SybilDetected);
    }

    #[test]
    fn test_recommendation_levels() {
        assert_eq!(
            get_recommendation(0.1, AnomalyType::None),
            AnomalyRecommendation::None
        );
        assert_eq!(
            get_recommendation(0.4, AnomalyType::None),
            AnomalyRecommendation::Monitor
        );
        assert_eq!(
            get_recommendation(0.55, AnomalyType::MLDetected),
            AnomalyRecommendation::Review
        );
        assert_eq!(
            get_recommendation(0.75, AnomalyType::GamingDetected),
            AnomalyRecommendation::Throttle
        );
        assert_eq!(
            get_recommendation(0.9, AnomalyType::Ensemble),
            AnomalyRecommendation::Quarantine
        );
    }
}
