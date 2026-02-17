//! Byzantine-resistant aggregation algorithms.
//!
//! Provides multiple defense strategies against Byzantine (malicious) participants:
//!
//! - **Krum**: Selects the gradient closest to others (most "central")
//! - **MultiKrum**: Selects top-k closest gradients and averages them
//! - **Median**: Coordinate-wise median (robust to outliers)
//! - **TrimmedMean**: Removes extreme values before averaging
//! - **FedAvg**: Simple averaging (no Byzantine protection)
//!
//! ## Architecture
//!
//! Core algorithms (FedAvg, Krum, Median, TrimmedMean) are delegated to
//! `mycelix-fl-core` — the single source of truth for canonical FL algorithms.
//! MultiKrum and GeometricMedian are implemented locally as they are unique
//! to the fl-aggregator tier.
//!
//! ## Byzantine Tolerance
//!
//! For `n` total nodes and `f` Byzantine nodes:
//! - Krum/MultiKrum require `n >= 2f + 3`
//! - Validated maximum: 34% (see `mycelix_fl_core::types::MAX_BYZANTINE_TOLERANCE`)

use crate::error::{AggregatorError, Result};
use crate::Gradient;
use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// Safe float comparison that logs a warning if NaN is encountered.
///
/// Since gradients are validated on submission, NaN values should never reach
/// this point. If they do, it indicates a bug elsewhere that bypassed validation.
fn safe_f32_cmp(a: &f32, b: &f32) -> Ordering {
    match a.partial_cmp(b) {
        Some(ord) => ord,
        None => {
            // This should never happen with validated input
            tracing::warn!(
                "NaN value encountered in Byzantine comparison (a={}, b={}). \
                 This indicates a validation bypass bug.",
                a, b
            );
            Ordering::Equal
        }
    }
}

/// Byzantine defense algorithm selection.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Defense {
    /// Simple federated averaging (no Byzantine protection).
    FedAvg,

    /// Krum algorithm - selects single most representative gradient.
    /// `f` is the maximum number of Byzantine nodes to tolerate.
    Krum { f: usize },

    /// Multi-Krum - selects and averages top-k most representative gradients.
    /// `f` is Byzantine tolerance, `k` is number of gradients to average.
    MultiKrum { f: usize, k: usize },

    /// Coordinate-wise median aggregation.
    Median,

    /// Trimmed mean - removes β fraction from each tail before averaging.
    TrimmedMean { beta: f32 },

    /// Geometric median (iterative approximation).
    GeometricMedian { max_iterations: usize, tolerance: f32 },
}

impl PartialEq for Defense {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Defense::FedAvg, Defense::FedAvg) => true,
            (Defense::Krum { f: f1 }, Defense::Krum { f: f2 }) => f1 == f2,
            (Defense::MultiKrum { f: f1, k: k1 }, Defense::MultiKrum { f: f2, k: k2 }) => {
                f1 == f2 && k1 == k2
            }
            (Defense::Median, Defense::Median) => true,
            (Defense::TrimmedMean { beta: b1 }, Defense::TrimmedMean { beta: b2 }) => {
                b1.to_bits() == b2.to_bits()
            }
            (
                Defense::GeometricMedian { max_iterations: m1, tolerance: t1 },
                Defense::GeometricMedian { max_iterations: m2, tolerance: t2 },
            ) => m1 == m2 && t1.to_bits() == t2.to_bits(),
            _ => false,
        }
    }
}

impl Eq for Defense {}

impl std::hash::Hash for Defense {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Defense::FedAvg => {}
            Defense::Krum { f } => f.hash(state),
            Defense::MultiKrum { f, k } => {
                f.hash(state);
                k.hash(state);
            }
            Defense::Median => {}
            Defense::TrimmedMean { beta } => beta.to_bits().hash(state),
            Defense::GeometricMedian { max_iterations, tolerance } => {
                max_iterations.hash(state);
                tolerance.to_bits().hash(state);
            }
        }
    }
}

impl Default for Defense {
    fn default() -> Self {
        Defense::Krum { f: 1 }
    }
}

impl std::fmt::Display for Defense {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Defense::FedAvg => write!(f, "FedAvg"),
            Defense::Krum { f: byzantine } => write!(f, "Krum(f={})", byzantine),
            Defense::MultiKrum { f: byzantine, k } => {
                write!(f, "MultiKrum(f={}, k={})", byzantine, k)
            }
            Defense::Median => write!(f, "Median"),
            Defense::TrimmedMean { beta } => write!(f, "TrimmedMean(β={})", beta),
            Defense::GeometricMedian { .. } => write!(f, "GeometricMedian"),
        }
    }
}

/// Configuration for defense algorithms.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DefenseConfig {
    /// The defense algorithm to use.
    pub defense: Defense,

    /// Whether to normalize gradients before aggregation.
    pub normalize: bool,

    /// Clip gradient norms to this value (0 = no clipping).
    pub clip_norm: f32,

    /// Minimum number of gradients required (overrides algorithm default).
    pub min_gradients: Option<usize>,
}

impl Default for DefenseConfig {
    fn default() -> Self {
        Self {
            defense: Defense::default(),
            normalize: false,
            clip_norm: 0.0,
            min_gradients: None,
        }
    }
}

impl DefenseConfig {
    /// Create config with specific defense.
    pub fn with_defense(defense: Defense) -> Self {
        Self {
            defense,
            ..Default::default()
        }
    }

    /// Enable gradient normalization.
    pub fn with_normalization(mut self) -> Self {
        self.normalize = true;
        self
    }

    /// Set gradient clipping threshold.
    pub fn with_clip_norm(mut self, clip: f32) -> Self {
        self.clip_norm = clip;
        self
    }

    /// Get minimum gradients required for this defense.
    pub fn min_required(&self) -> usize {
        if let Some(min) = self.min_gradients {
            return min;
        }

        match &self.defense {
            Defense::FedAvg => 1,
            Defense::Krum { f } => 2 * f + 3,
            Defense::MultiKrum { f, .. } => 2 * f + 3,
            Defense::Median => 1,
            Defense::TrimmedMean { beta } => {
                // Need at least enough to have something after trimming
                let min = (2.0 / (1.0 - 2.0 * beta)).ceil() as usize;
                min.max(3)
            }
            Defense::GeometricMedian { .. } => 1,
        }
    }
}

/// Byzantine-resistant aggregator implementation.
pub struct ByzantineAggregator {
    config: DefenseConfig,
}

impl ByzantineAggregator {
    /// Create new aggregator with given defense configuration.
    pub fn new(config: DefenseConfig) -> Self {
        Self { config }
    }

    /// Aggregate gradients using configured defense algorithm.
    pub fn aggregate(&self, gradients: &[Gradient]) -> Result<Gradient> {
        let n = gradients.len();
        let min_required = self.config.min_required();

        if n < min_required {
            return Err(AggregatorError::InsufficientGradients {
                have: n,
                need: min_required,
                defense: self.config.defense.to_string(),
            });
        }

        // Pre-process: normalize and/or clip if configured
        let processed = self.preprocess(gradients);
        let processed_refs: Vec<_> = processed.iter().collect();

        // Apply defense algorithm
        match &self.config.defense {
            Defense::FedAvg => self.fedavg(&processed_refs),
            Defense::Krum { f } => self.krum(&processed_refs, *f),
            Defense::MultiKrum { f, k } => self.multi_krum(&processed_refs, *f, *k),
            Defense::Median => self.median(&processed_refs),
            Defense::TrimmedMean { beta } => self.trimmed_mean(&processed_refs, *beta),
            Defense::GeometricMedian {
                max_iterations,
                tolerance,
            } => self.geometric_median(&processed_refs, *max_iterations, *tolerance),
        }
    }

    /// Convert ndarray gradients to fl-core GradientUpdate format for delegation.
    fn to_core_updates(gradients: &[&Gradient]) -> Vec<mycelix_fl_core::types::GradientUpdate> {
        gradients
            .iter()
            .enumerate()
            .map(|(i, g)| {
                mycelix_fl_core::types::GradientUpdate::new(
                    format!("node-{}", i),
                    1,          // round
                    g.to_vec(), // Array1<f32> → Vec<f32>
                    1,          // batch_size
                    0.0,        // loss (not tracked at this layer)
                )
            })
            .collect()
    }

    /// Preprocess gradients (normalize, clip).
    fn preprocess(&self, gradients: &[Gradient]) -> Vec<Gradient> {
        gradients
            .iter()
            .map(|g| {
                let mut processed = g.clone();

                // Clip norm if configured
                if self.config.clip_norm > 0.0 {
                    let norm = l2_norm(processed.view());
                    if norm > self.config.clip_norm {
                        processed *= self.config.clip_norm / norm;
                    }
                }

                // Normalize if configured
                if self.config.normalize {
                    let norm = l2_norm(processed.view());
                    if norm > 1e-10 {
                        processed /= norm;
                    }
                }

                processed
            })
            .collect()
    }

    /// Simple federated averaging.
    ///
    /// Delegates to `mycelix_fl_core::aggregation::fedavg`.
    fn fedavg(&self, gradients: &[&Gradient]) -> Result<Gradient> {
        let updates = Self::to_core_updates(gradients);
        let result = mycelix_fl_core::aggregation::fedavg(&updates)
            .map_err(|e| AggregatorError::Internal(format!("Core FedAvg error: {}", e)))?;
        Ok(Array1::from(result))
    }

    /// Krum algorithm - select gradient closest to others.
    ///
    /// Delegates to `mycelix_fl_core::aggregation::krum`.
    fn krum(&self, gradients: &[&Gradient], f: usize) -> Result<Gradient> {
        let n = gradients.len();

        if n <= 2 * f + 2 {
            return Err(AggregatorError::InsufficientGradients {
                have: n,
                need: 2 * f + 3,
                defense: format!("Krum(f={})", f),
            });
        }

        // fl-core's krum takes num_select (1 for standard Krum)
        let updates = Self::to_core_updates(gradients);
        let result = mycelix_fl_core::aggregation::krum(&updates, 1)
            .map_err(|e| AggregatorError::Internal(format!("Core Krum error: {}", e)))?;

        tracing::debug!("Krum selected gradient via fl-core delegation");

        Ok(Array1::from(result))
    }

    /// Multi-Krum - select and average top-k gradients by Krum score.
    fn multi_krum(&self, gradients: &[&Gradient], f: usize, k: usize) -> Result<Gradient> {
        let n = gradients.len();

        if k > n - f {
            return Err(AggregatorError::InvalidConfig(format!(
                "MultiKrum k={} > n-f={}",
                k,
                n - f
            )));
        }

        let scores = self.compute_krum_scores(gradients, f);

        // Sort indices by score
        let mut indexed_scores: Vec<_> = scores.iter().enumerate().collect();
        indexed_scores.sort_by(|(_, a), (_, b)| safe_f32_cmp(a, b));

        // Select top k
        let selected: Vec<usize> = indexed_scores.iter().take(k).map(|(i, _)| *i).collect();

        tracing::debug!(
            "MultiKrum selected {} gradients: {:?}",
            k,
            selected
                .iter()
                .map(|&i| format!("{}({:.2})", i, scores[i]))
                .collect::<Vec<_>>()
        );

        // Average selected gradients
        self.average_indices(gradients, &selected)
    }

    /// Compute Krum scores for all gradients.
    fn compute_krum_scores(&self, gradients: &[&Gradient], f: usize) -> Vec<f32> {
        let n = gradients.len();
        let take_count = n.saturating_sub(f).saturating_sub(2);

        gradients
            .iter()
            .enumerate()
            .map(|(i, gi)| {
                // Compute distances to all other gradients
                let mut distances: Vec<f32> = gradients
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, gj)| euclidean_distance(gi.view(), gj.view()))
                    .collect();

                // Sort and sum smallest n-f-2 distances
                distances.sort_by(|a, b| safe_f32_cmp(a, b));
                distances.iter().take(take_count).sum()
            })
            .collect()
    }

    /// Coordinate-wise median aggregation.
    ///
    /// Delegates to `mycelix_fl_core::aggregation::coordinate_median`.
    fn median(&self, gradients: &[&Gradient]) -> Result<Gradient> {
        let updates = Self::to_core_updates(gradients);
        let result = mycelix_fl_core::aggregation::coordinate_median(&updates)
            .map_err(|e| AggregatorError::Internal(format!("Core Median error: {}", e)))?;
        Ok(Array1::from(result))
    }

    /// Trimmed mean - remove β fraction from each tail.
    ///
    /// Delegates to `mycelix_fl_core::aggregation::trimmed_mean`.
    fn trimmed_mean(&self, gradients: &[&Gradient], beta: f32) -> Result<Gradient> {
        if !(0.0..0.5).contains(&beta) {
            return Err(AggregatorError::InvalidConfig(format!(
                "TrimmedMean beta must be in [0, 0.5), got {}",
                beta
            )));
        }

        let updates = Self::to_core_updates(gradients);
        let result = mycelix_fl_core::aggregation::trimmed_mean(&updates, beta)
            .map_err(|e| AggregatorError::Internal(format!("Core TrimmedMean error: {}", e)))?;

        let n = gradients.len();
        let trim_count = (n as f32 * beta).floor() as usize;
        tracing::debug!(
            "TrimmedMean: trimmed {} values from each end ({} remaining)",
            trim_count,
            n - 2 * trim_count
        );

        Ok(Array1::from(result))
    }

    /// Geometric median via Weiszfeld algorithm.
    fn geometric_median(
        &self,
        gradients: &[&Gradient],
        max_iterations: usize,
        tolerance: f32,
    ) -> Result<Gradient> {
        let n = gradients.len();
        let dim = gradients[0].len();

        // Initialize with arithmetic mean
        let mut median = Array1::zeros(dim);
        for g in gradients {
            median += &g.view();
        }
        median /= n as f32;

        // Weiszfeld iteration
        for iter in 0..max_iterations {
            let mut numerator = Array1::zeros(dim);
            let mut denominator = 0.0f32;

            for g in gradients {
                let dist = euclidean_distance(median.view(), g.view());
                if dist > 1e-10 {
                    let weight = 1.0 / dist;
                    numerator += &(g.view().to_owned() * weight);
                    denominator += weight;
                }
            }

            if denominator < 1e-10 {
                break;
            }

            let new_median = numerator / denominator;
            let change = euclidean_distance(new_median.view(), median.view());

            median = new_median;

            if change < tolerance {
                tracing::debug!("GeometricMedian converged after {} iterations", iter + 1);
                break;
            }
        }

        Ok(median)
    }

    /// Average gradients at specified indices.
    fn average_indices(&self, gradients: &[&Gradient], indices: &[usize]) -> Result<Gradient> {
        if indices.is_empty() {
            return Err(AggregatorError::Internal(
                "No indices to average".to_string(),
            ));
        }

        let dim = gradients[0].len();
        let mut sum = Array1::zeros(dim);

        for &idx in indices {
            sum += &gradients[idx].view();
        }

        Ok(sum / indices.len() as f32)
    }
}

/// Calculate Euclidean (L2) distance between two vectors.
#[inline]
pub fn euclidean_distance(v1: ArrayView1<f32>, v2: ArrayView1<f32>) -> f32 {
    v1.iter()
        .zip(v2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Calculate L2 norm of a vector.
#[inline]
pub fn l2_norm(v: ArrayView1<f32>) -> f32 {
    v.iter().map(|x| x.powi(2)).sum::<f32>().sqrt()
}

/// Calculate cosine similarity between two vectors.
#[inline]
pub fn cosine_similarity(v1: ArrayView1<f32>, v2: ArrayView1<f32>) -> f32 {
    let dot: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    let norm1 = l2_norm(v1);
    let norm2 = l2_norm(v2);

    if norm1 < 1e-10 || norm2 < 1e-10 {
        return 0.0;
    }

    dot / (norm1 * norm2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_fedavg() {
        let aggregator = ByzantineAggregator::new(DefenseConfig::with_defense(Defense::FedAvg));

        let gradients = vec![
            array![1.0, 2.0, 3.0],
            array![4.0, 5.0, 6.0],
            array![7.0, 8.0, 9.0],
        ];
        let refs: Vec<_> = gradients.iter().collect();

        let result = aggregator.aggregate(&gradients).unwrap();

        assert_relative_eq!(result[0], 4.0, epsilon = 1e-6);
        assert_relative_eq!(result[1], 5.0, epsilon = 1e-6);
        assert_relative_eq!(result[2], 6.0, epsilon = 1e-6);
    }

    #[test]
    fn test_krum_rejects_byzantine() {
        let aggregator =
            ByzantineAggregator::new(DefenseConfig::with_defense(Defense::Krum { f: 1 }));

        // Normal gradients cluster around [1, 2, 3]
        // Byzantine gradient is far away at [100, 200, 300]
        let gradients = vec![
            array![1.0, 2.0, 3.0],
            array![1.1, 2.1, 3.1],
            array![1.2, 2.2, 3.2],
            array![0.9, 1.9, 2.9],
            array![100.0, 200.0, 300.0], // Byzantine
        ];

        let result = aggregator.aggregate(&gradients).unwrap();

        // Should not select the Byzantine gradient
        assert!(result[0] < 5.0);
        assert!(result[1] < 10.0);
        assert!(result[2] < 15.0);
    }

    #[test]
    fn test_multi_krum() {
        let aggregator =
            ByzantineAggregator::new(DefenseConfig::with_defense(Defense::MultiKrum { f: 1, k: 3 }));

        let gradients = vec![
            array![1.0, 2.0],
            array![1.1, 2.1],
            array![1.2, 2.2],
            array![0.9, 1.9],
            array![100.0, 200.0], // Byzantine
        ];

        let result = aggregator.aggregate(&gradients).unwrap();

        // Average of 3 best should be near [1, 2]
        assert!(result[0] < 5.0);
        assert!(result[1] < 5.0);
    }

    #[test]
    fn test_median() {
        let aggregator = ByzantineAggregator::new(DefenseConfig::with_defense(Defense::Median));

        let gradients = vec![
            array![1.0, 2.0],
            array![2.0, 3.0],
            array![3.0, 4.0],
            array![100.0, 200.0], // Byzantine
            array![4.0, 5.0],
        ];

        let result = aggregator.aggregate(&gradients).unwrap();

        // Median should be [3.0, 4.0]
        assert_relative_eq!(result[0], 3.0, epsilon = 1e-6);
        assert_relative_eq!(result[1], 4.0, epsilon = 1e-6);
    }

    #[test]
    fn test_trimmed_mean() {
        let aggregator =
            ByzantineAggregator::new(DefenseConfig::with_defense(Defense::TrimmedMean {
                beta: 0.2,
            }));

        // With 5 values and beta=0.2, we trim 1 from each end
        let gradients = vec![
            array![1.0],   // trimmed (lowest)
            array![2.0],   // kept
            array![3.0],   // kept
            array![4.0],   // kept
            array![100.0], // trimmed (highest)
        ];

        let result = aggregator.aggregate(&gradients).unwrap();

        // Mean of [2, 3, 4] = 3.0
        assert_relative_eq!(result[0], 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_geometric_median() {
        let aggregator =
            ByzantineAggregator::new(DefenseConfig::with_defense(Defense::GeometricMedian {
                max_iterations: 100,
                tolerance: 1e-6,
            }));

        let gradients = vec![
            array![0.0, 0.0],
            array![1.0, 0.0],
            array![0.0, 1.0],
            array![1.0, 1.0],
        ];

        let result = aggregator.aggregate(&gradients).unwrap();

        // Geometric median of unit square vertices is approximately (0.5, 0.5)
        assert_relative_eq!(result[0], 0.5, epsilon = 0.1);
        assert_relative_eq!(result[1], 0.5, epsilon = 0.1);
    }

    #[test]
    fn test_gradient_clipping() {
        let config = DefenseConfig::with_defense(Defense::FedAvg).with_clip_norm(1.0);
        let aggregator = ByzantineAggregator::new(config);

        // Large gradient that should be clipped
        let gradients = vec![array![10.0, 0.0], array![0.0, 10.0]];

        let result = aggregator.aggregate(&gradients).unwrap();

        // Both inputs clipped to norm=1, so result should have small norm
        let result_norm = l2_norm(result.view());
        assert!(result_norm <= 1.5); // Some tolerance for averaging
    }

    #[test]
    fn test_insufficient_gradients_error() {
        let aggregator =
            ByzantineAggregator::new(DefenseConfig::with_defense(Defense::Krum { f: 2 }));

        // Krum with f=2 needs at least 7 gradients
        let gradients = vec![array![1.0], array![2.0], array![3.0]];

        let result = aggregator.aggregate(&gradients);
        assert!(matches!(
            result,
            Err(AggregatorError::InsufficientGradients { .. })
        ));
    }

    #[test]
    fn test_euclidean_distance() {
        let v1 = array![0.0, 0.0, 0.0];
        let v2 = array![3.0, 4.0, 0.0];

        let dist = euclidean_distance(v1.view(), v2.view());
        assert_relative_eq!(dist, 5.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cosine_similarity() {
        let v1 = array![1.0, 0.0];
        let v2 = array![1.0, 0.0];
        assert_relative_eq!(cosine_similarity(v1.view(), v2.view()), 1.0, epsilon = 1e-6);

        let v3 = array![0.0, 1.0];
        assert_relative_eq!(cosine_similarity(v1.view(), v3.view()), 0.0, epsilon = 1e-6);

        let v4 = array![-1.0, 0.0];
        assert_relative_eq!(
            cosine_similarity(v1.view(), v4.view()),
            -1.0,
            epsilon = 1e-6
        );
    }
}
