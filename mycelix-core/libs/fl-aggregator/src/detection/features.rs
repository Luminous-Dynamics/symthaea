//! Feature extraction for Byzantine detection.
//!
//! Computes composite feature vectors from gradients:
//! - PoGQ (Proof of Gradient Quality): Cosine similarity to median
//! - TCDM (Temporal Consistency): Correlation with historical gradients
//! - Z-Score: Statistical outlier detection
//! - Entropy: Shannon entropy of gradient distribution

use crate::Gradient;
use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Composite features for a single gradient.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompositeFeatures {
    /// PoGQ score [0, 1] - Higher is more honest (cosine similarity to median)
    pub pogq_score: f32,

    /// TCDM score [0, 1] - Temporal consistency with history
    pub tcdm_score: f32,

    /// Z-score magnitude - Distance from mean (higher = outlier)
    pub zscore_magnitude: f32,

    /// Shannon entropy of gradient distribution
    pub entropy_score: f32,

    /// L2 norm of gradient
    pub gradient_norm: f32,

    /// Node identifier
    pub node_id: String,

    /// Round number
    pub round_num: u64,
}

impl CompositeFeatures {
    /// Convert to feature vector for ML classifier.
    pub fn to_array(&self) -> Array1<f32> {
        Array1::from(vec![
            self.pogq_score,
            self.tcdm_score,
            self.zscore_magnitude,
            self.entropy_score,
            self.gradient_norm,
        ])
    }

    /// Feature names for interpretability.
    pub fn feature_names() -> Vec<&'static str> {
        vec![
            "pogq_score",
            "tcdm_score",
            "zscore_magnitude",
            "entropy_score",
            "gradient_norm",
        ]
    }

    /// Number of features.
    pub const NUM_FEATURES: usize = 5;
}

/// Feature extractor maintaining per-node history for TCDM.
pub struct FeatureExtractor {
    /// History window size for TCDM computation.
    history_window: usize,

    /// Per-node gradient history: node_id -> recent gradients.
    gradient_history: HashMap<String, VecDeque<Gradient>>,

    /// Global statistics (updated each round).
    global_mean: Option<Gradient>,
    global_std: Option<Gradient>,
    global_median: Option<Gradient>,
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new(5)
    }
}

impl FeatureExtractor {
    /// Create a new feature extractor.
    ///
    /// # Arguments
    /// * `history_window` - Number of past gradients to retain for TCDM.
    pub fn new(history_window: usize) -> Self {
        Self {
            history_window,
            gradient_history: HashMap::new(),
            global_mean: None,
            global_std: None,
            global_median: None,
        }
    }

    /// Update global gradient statistics for z-score and PoGQ computation.
    pub fn update_global_statistics(&mut self, gradients: &[Gradient]) {
        if gradients.is_empty() {
            return;
        }

        let n = gradients.len();
        let dim = gradients[0].len();

        // Compute mean
        let mut mean = Array1::zeros(dim);
        for g in gradients {
            mean += &g.view();
        }
        mean /= n as f32;

        // Compute std
        let mut variance = Array1::zeros(dim);
        for g in gradients {
            let diff = &g.view() - &mean.view();
            variance += &(&diff * &diff);
        }
        variance /= n as f32;
        let std = variance.mapv(|v| (v + 1e-8).sqrt());

        // Compute median (element-wise)
        let mut median = Array1::zeros(dim);
        for d in 0..dim {
            let mut values: Vec<f32> = gradients.iter().map(|g| g[d]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = values.len() / 2;
            median[d] = if values.len() % 2 == 0 {
                (values[mid - 1] + values[mid]) / 2.0
            } else {
                values[mid]
            };
        }

        self.global_mean = Some(mean);
        self.global_std = Some(std);
        self.global_median = Some(median);
    }

    /// Extract all features for a single gradient.
    pub fn extract_features(
        &mut self,
        gradient: &Gradient,
        node_id: &str,
        round_num: u64,
        all_gradients: Option<&[Gradient]>,
    ) -> CompositeFeatures {
        // Update global stats if provided
        if let Some(grads) = all_gradients {
            self.update_global_statistics(grads);
        }

        // 1. Compute PoGQ (cosine similarity to median)
        let pogq_score = self.compute_pogq(gradient);

        // 2. Compute TCDM (temporal consistency)
        let tcdm_score = self.compute_tcdm(gradient, node_id);

        // 3. Compute Z-score magnitude
        let zscore_magnitude = self.compute_zscore(gradient);

        // 4. Compute entropy
        let entropy_score = self.compute_entropy(gradient);

        // 5. Compute gradient norm
        let gradient_norm = l2_norm(gradient.view());

        // Update history
        self.update_history(gradient.clone(), node_id);

        CompositeFeatures {
            pogq_score,
            tcdm_score,
            zscore_magnitude,
            entropy_score,
            gradient_norm,
            node_id: node_id.to_string(),
            round_num,
        }
    }

    /// Compute Proof of Gradient Quality (cosine similarity to median).
    fn compute_pogq(&self, gradient: &Gradient) -> f32 {
        let median = match &self.global_median {
            Some(m) => m,
            None => return 0.5, // Neutral score if no stats available
        };

        let dot_product: f32 = gradient.iter().zip(median.iter()).map(|(a, b)| a * b).sum();
        let norm_g = l2_norm(gradient.view());
        let norm_m = l2_norm(median.view());

        if norm_g < 1e-10 || norm_m < 1e-10 {
            return 0.0;
        }

        let similarity = dot_product / (norm_g * norm_m);

        // Convert from [-1, 1] to [0, 1]
        (similarity + 1.0) / 2.0
    }

    /// Compute Temporal Consistency Detection Metric.
    fn compute_tcdm(&self, gradient: &Gradient, node_id: &str) -> f32 {
        let history = match self.gradient_history.get(node_id) {
            Some(h) if !h.is_empty() => h,
            _ => return 0.5, // Neutral score for first gradient
        };

        // Compute average cosine similarity with historical gradients
        let mut similarities = Vec::new();
        for past_gradient in history.iter() {
            let dot_prod: f32 = gradient
                .iter()
                .zip(past_gradient.iter())
                .map(|(a, b)| a * b)
                .sum();
            let norm_curr = l2_norm(gradient.view());
            let norm_past = l2_norm(past_gradient.view());

            if norm_curr < 1e-10 || norm_past < 1e-10 {
                continue;
            }

            let sim = dot_prod / (norm_curr * norm_past);
            similarities.push((sim + 1.0) / 2.0); // Convert to [0, 1]
        }

        if similarities.is_empty() {
            return 0.5;
        }

        similarities.iter().sum::<f32>() / similarities.len() as f32
    }

    /// Compute Z-score magnitude (statistical outlier detection).
    fn compute_zscore(&self, gradient: &Gradient) -> f32 {
        let (mean, std) = match (&self.global_mean, &self.global_std) {
            (Some(m), Some(s)) => (m, s),
            _ => return 0.0,
        };

        // Element-wise z-score
        let z_scores: Array1<f32> = gradient
            .iter()
            .zip(mean.iter())
            .zip(std.iter())
            .map(|((g, m), s)| (g - m) / s)
            .collect();

        // Return L2 norm of z-score vector
        l2_norm(z_scores.view())
    }

    /// Compute Shannon entropy of gradient distribution.
    fn compute_entropy(&self, gradient: &Gradient) -> f32 {
        // Handle empty gradient
        if gradient.is_empty() {
            return 0.0;
        }

        let num_bins = (gradient.len() / 10).clamp(2, 50);

        // Find min/max for binning
        let min_val = gradient.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = gradient.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Check for invalid min/max (all NaN or degenerate)
        let range = max_val - min_val;
        if !range.is_finite() || range.abs() < 1e-10 {
            return 0.0;  // All values identical or invalid
        }

        // Create histogram
        let bin_width = range / num_bins as f32;

        // Safety check: avoid division by denormalized bin_width
        if !bin_width.is_normal() {
            return 0.0;
        }

        let mut hist = vec![0usize; num_bins];

        for &val in gradient.iter() {
            if val.is_finite() {
                let bin = ((val - min_val) / bin_width) as usize;
                let bin = bin.min(num_bins - 1); // Handle edge case
                hist[bin] += 1;
            }
        }

        // Convert to probabilities and compute entropy
        let total = gradient.len() as f32;
        if total == 0.0 {
            return 0.0;
        }

        let mut entropy = 0.0f32;

        for &count in &hist {
            if count > 0 {
                let p = count as f32 / total;
                // Skip denormalized probabilities to avoid log2 issues
                if p.is_normal() && p > 1e-15 {
                    entropy -= p * p.log2();
                }
            }
        }

        // Final safety check
        if entropy.is_finite() { entropy } else { 0.0 }
    }

    /// Update gradient history for TCDM computation.
    fn update_history(&mut self, gradient: Gradient, node_id: &str) {
        let history = self
            .gradient_history
            .entry(node_id.to_string())
            .or_insert_with(|| VecDeque::with_capacity(self.history_window));

        if history.len() >= self.history_window {
            history.pop_front();
        }
        history.push_back(gradient);
    }

    /// Clear all history and statistics.
    pub fn reset(&mut self) {
        self.gradient_history.clear();
        self.global_mean = None;
        self.global_std = None;
        self.global_median = None;
    }

    /// Get the number of nodes with history.
    pub fn nodes_tracked(&self) -> usize {
        self.gradient_history.len()
    }
}

/// Calculate L2 norm of a vector.
#[inline]
fn l2_norm(v: ArrayView1<f32>) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Batch feature extraction for efficiency.
pub fn extract_features_batch(
    gradients: &[Gradient],
    node_ids: &[String],
    round_num: u64,
    extractor: &mut FeatureExtractor,
) -> Vec<CompositeFeatures> {
    // Update global stats once for entire batch
    extractor.update_global_statistics(gradients);

    // Extract features for each gradient
    gradients
        .iter()
        .zip(node_ids.iter())
        .map(|(gradient, node_id)| {
            extractor.extract_features(gradient, node_id, round_num, None)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_pogq_identical_gradients() {
        let mut extractor = FeatureExtractor::new(5);
        let gradient = array![1.0, 2.0, 3.0];
        let all_gradients = vec![gradient.clone(), gradient.clone(), gradient.clone()];

        extractor.update_global_statistics(&all_gradients);
        let pogq = extractor.compute_pogq(&gradient);

        // Identical gradients should have PoGQ = 1.0
        assert!((pogq - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_pogq_opposite_gradients() {
        let mut extractor = FeatureExtractor::new(5);
        let gradient = array![1.0, 2.0, 3.0];
        let opposite = array![-1.0, -2.0, -3.0];
        let all_gradients = vec![opposite.clone()];

        extractor.update_global_statistics(&all_gradients);
        let pogq = extractor.compute_pogq(&gradient);

        // Opposite direction should have PoGQ = 0.0
        assert!(pogq < 0.1);
    }

    #[test]
    fn test_tcdm_consistent_node() {
        let mut extractor = FeatureExtractor::new(5);

        // Submit similar gradients from same node
        for i in 0..5 {
            let gradient = array![1.0 + i as f32 * 0.01, 2.0, 3.0];
            extractor.extract_features(&gradient, "node_1", i as u64, None);
        }

        // New similar gradient should have high TCDM
        let new_gradient = array![1.05, 2.0, 3.0];
        let tcdm = extractor.compute_tcdm(&new_gradient, "node_1");

        assert!(tcdm > 0.9);
    }

    #[test]
    fn test_zscore_outlier() {
        let mut extractor = FeatureExtractor::new(5);

        // Normal gradients around [1, 2, 3]
        let normal_gradients = vec![
            array![1.0, 2.0, 3.0],
            array![1.1, 2.1, 3.1],
            array![0.9, 1.9, 2.9],
            array![1.0, 2.0, 3.0],
        ];
        extractor.update_global_statistics(&normal_gradients);

        // Outlier gradient
        let outlier = array![100.0, 200.0, 300.0];
        let zscore = extractor.compute_zscore(&outlier);

        // Should have high z-score
        assert!(zscore > 10.0);
    }

    #[test]
    fn test_entropy_uniform() {
        let extractor = FeatureExtractor::new(5);

        // Gradient with uniform distribution
        let gradient: Gradient = (0..100).map(|i| i as f32 / 100.0).collect();
        let entropy = extractor.compute_entropy(&gradient);

        // Uniform distribution should have relatively high entropy
        assert!(entropy > 2.0);
    }

    #[test]
    fn test_composite_features() {
        let mut extractor = FeatureExtractor::new(5);
        let gradient = array![1.0, 2.0, 3.0];
        let all_gradients = vec![gradient.clone()];

        let features =
            extractor.extract_features(&gradient, "node_1", 0, Some(&all_gradients));

        assert_eq!(features.node_id, "node_1");
        assert_eq!(features.round_num, 0);
        assert!(features.gradient_norm > 0.0);

        // Test conversion to array
        let arr = features.to_array();
        assert_eq!(arr.len(), CompositeFeatures::NUM_FEATURES);
    }
}
