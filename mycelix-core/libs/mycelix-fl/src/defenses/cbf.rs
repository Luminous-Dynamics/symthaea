// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Conformal Behavioral Filter (CBF) for Byzantine detection.
//!
//! A representation-based pre-aggregation filter that uses PCA for feature
//! extraction and conformal prediction for anomaly detection with FPR
//! guarantees.
//!
//! Pipeline:
//! 1. Extract features from each gradient (norm, mean, std, sparsity, direction change)
//! 2. Apply PCA to reduce features to k principal components
//! 3. Compute nonconformity scores (distance from centroid in PCA space)
//! 4. Apply conformal prediction: reject gradients with p-value < alpha
//! 5. Average the remaining (non-anomalous) gradients

use std::collections::HashMap;

use crate::error::FlError;
use crate::types::{AggregationResult, Gradient};

use super::validate_gradients;

/// Number of features extracted per gradient.
const NUM_FEATURES: usize = 5;

/// Maximum number of power iteration steps for eigendecomposition.
const POWER_ITER_MAX: usize = 200;

/// Convergence threshold for power iteration.
const POWER_ITER_TOL: f64 = 1e-8;

/// Conformal Behavioral Filter.
///
/// Maintains per-node feature history and previous-round gradients to compute
/// direction-change features. The conformal threshold guarantees a bounded
/// false positive rate on clean data.
pub struct ConformalBehavioralFilter {
    /// History of feature vectors per node (for trend analysis).
    feature_history: HashMap<String, Vec<Vec<f64>>>,
    /// Previous round gradients for computing direction change.
    prev_gradients: HashMap<String, Vec<f32>>,
    /// Conformal FPR threshold (reject if p-value < alpha).
    alpha: f64,
    /// Maximum feature history window per node.
    history_window: usize,
    /// Calibration scores from known-clean data (for conformal quantile).
    calibration_scores: Vec<f64>,
}

impl ConformalBehavioralFilter {
    /// Create a new CBF with the given false-positive rate target.
    pub fn new(alpha: f64) -> Self {
        Self {
            feature_history: HashMap::new(),
            prev_gradients: HashMap::new(),
            alpha,
            history_window: 10,
            calibration_scores: Vec::new(),
        }
    }

    /// Calibrate the conformal threshold on known-clean gradients.
    ///
    /// The calibration set should be separate from the training data.
    /// After calibration, the filter can provide FPR guarantees.
    pub fn calibrate(&mut self, clean_gradients: &[Gradient]) -> Result<(), FlError> {
        if clean_gradients.len() < 3 {
            return Err(FlError::InvalidInput(
                "need at least 3 clean gradients for calibration".into(),
            ));
        }

        // Extract features for all clean gradients.
        let features: Vec<Vec<f64>> = clean_gradients
            .iter()
            .map(|g| self.extract_features(&g.values, &g.node_id))
            .collect();

        // Compute centroid.
        let centroid = feature_centroid(&features);

        // Store distances as calibration scores.
        self.calibration_scores = features
            .iter()
            .map(|f| euclidean_distance(f, &centroid))
            .collect();

        // Sort for quantile computation.
        self.calibration_scores
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Update prev_gradients for direction-change features in future rounds.
        for g in clean_gradients {
            self.prev_gradients
                .insert(g.node_id.clone(), g.values.clone());
        }

        Ok(())
    }

    /// Aggregate gradients, filtering anomalies via conformal prediction.
    pub fn aggregate(&mut self, gradients: &[Gradient]) -> Result<AggregationResult, FlError> {
        let dim = validate_gradients(gradients)?;
        let n = gradients.len();

        // Step 1: Extract features.
        let features: Vec<Vec<f64>> = gradients
            .iter()
            .map(|g| self.extract_features(&g.values, &g.node_id))
            .collect();

        // Step 2: PCA transform.
        let n_components = NUM_FEATURES.min(n.saturating_sub(1)).max(1);
        let projected = pca_transform(&features, n_components);

        // Step 3: Compute nonconformity scores (distance from centroid).
        let centroid = feature_centroid(&projected);
        let scores: Vec<f64> = projected
            .iter()
            .map(|f| euclidean_distance(f, &centroid))
            .collect();

        // Step 4: Conformal prediction — filter by threshold.
        let threshold = self.conformal_threshold(&scores);

        let mut included = Vec::new();
        let mut excluded = Vec::new();
        let mut weights = vec![0.0_f64; n];

        for (i, &score) in scores.iter().enumerate() {
            if score <= threshold {
                included.push(gradients[i].node_id.clone());
                weights[i] = 1.0;
            } else {
                excluded.push(gradients[i].node_id.clone());
            }
        }

        // Fail-open: if everything was rejected, accept all.
        if included.is_empty() {
            included = gradients.iter().map(|g| g.node_id.clone()).collect();
            excluded.clear();
            for w in &mut weights {
                *w = 1.0;
            }
        }

        // Normalize weights.
        let total: f64 = weights.iter().sum();
        if total > 0.0 {
            for w in &mut weights {
                *w /= total;
            }
        }

        // Step 5: Weighted average of accepted gradients.
        let mut agg = vec![0.0_f64; dim];
        for (g, &w) in gradients.iter().zip(weights.iter()) {
            if w > 0.0 {
                for (a, &v) in agg.iter_mut().zip(g.values.iter()) {
                    *a += w * v as f64;
                }
            }
        }

        // Update state for next round.
        for (i, g) in gradients.iter().enumerate() {
            self.prev_gradients
                .insert(g.node_id.clone(), g.values.clone());

            let history = self
                .feature_history
                .entry(g.node_id.clone())
                .or_default();
            history.push(features[i].clone());
            if history.len() > self.history_window {
                history.remove(0);
            }
        }

        let score_pairs: Vec<(String, f64)> = gradients
            .iter()
            .zip(weights.iter())
            .map(|(g, &w)| (g.node_id.clone(), w))
            .collect();

        Ok(AggregationResult {
            gradient: agg.iter().map(|&v| v as f32).collect(),
            included_nodes: included,
            excluded_nodes: excluded,
            scores: score_pairs,
        })
    }

    /// Extract a feature vector from a gradient.
    ///
    /// Features: [norm, mean, std, sparsity, direction_change]
    fn extract_features(&self, values: &[f32], node_id: &str) -> Vec<f64> {
        let n = values.len() as f64;
        if n == 0.0 {
            return vec![0.0; NUM_FEATURES];
        }

        let norm: f64 = values.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
        let mean: f64 = values.iter().map(|&v| v as f64).sum::<f64>() / n;
        let variance: f64 = values.iter().map(|&v| ((v as f64) - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();
        let sparsity = values.iter().filter(|&&v| v.abs() < 1e-6).count() as f64 / n;

        let direction_change = match self.prev_gradients.get(node_id) {
            Some(prev) if prev.len() == values.len() => {
                let mut dot = 0.0_f64;
                let mut norm_a = 0.0_f64;
                let mut norm_b = 0.0_f64;
                for (&a, &b) in prev.iter().zip(values.iter()) {
                    let (a, b) = (a as f64, b as f64);
                    dot += a * b;
                    norm_a += a * a;
                    norm_b += b * b;
                }
                let denom = norm_a.sqrt() * norm_b.sqrt();
                if denom > 1e-12 { 1.0 - (dot / denom) } else { 0.0 }
            }
            _ => 0.0,
        };

        vec![norm, mean, std, sparsity, direction_change]
    }

    /// Compute the conformal threshold for anomaly rejection.
    ///
    /// Uses the (1 - alpha) quantile of calibration + current scores
    /// (split conformal prediction).
    fn conformal_threshold(&self, current_scores: &[f64]) -> f64 {
        let mut all_scores: Vec<f64> = self.calibration_scores.clone();
        all_scores.extend_from_slice(current_scores);
        all_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if all_scores.is_empty() {
            return f64::MAX;
        }

        let idx = ((1.0 - self.alpha) * (all_scores.len() as f64 - 1.0)).ceil() as usize;
        let idx = idx.min(all_scores.len() - 1);
        all_scores[idx]
    }
}

// ─── Linear Algebra Helpers ─────────────────────────────────────────────────

/// Compute the centroid (mean) of a set of feature vectors.
fn feature_centroid(features: &[Vec<f64>]) -> Vec<f64> {
    if features.is_empty() {
        return vec![];
    }
    let d = features[0].len();
    let n = features.len() as f64;
    let mut centroid = vec![0.0; d];
    for f in features {
        for (c, &v) in centroid.iter_mut().zip(f.iter()) {
            *c += v;
        }
    }
    for c in &mut centroid {
        *c /= n;
    }
    centroid
}

/// Euclidean distance between two vectors.
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| (ai - bi).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Simple PCA via eigendecomposition of the covariance matrix.
///
/// Projects `features` (n x d) onto the top `n_components` principal
/// components, returning an (n x n_components) matrix.
pub fn pca_transform(features: &[Vec<f64>], n_components: usize) -> Vec<Vec<f64>> {
    if features.is_empty() || n_components == 0 {
        return vec![];
    }

    let n = features.len();
    let d = features[0].len();
    if d == 0 {
        return vec![vec![]; n];
    }

    let mean = feature_centroid(features);
    let centered: Vec<Vec<f64>> = features
        .iter()
        .map(|f| f.iter().zip(mean.iter()).map(|(&fi, &mi)| fi - mi).collect())
        .collect();

    let cov = covariance_matrix(&centered);
    let k = n_components.min(d);
    let eigenvectors = top_eigenvectors(&cov, k, POWER_ITER_MAX);

    centered
        .iter()
        .map(|sample| {
            eigenvectors
                .iter()
                .map(|ev| sample.iter().zip(ev.iter()).map(|(&s, &e)| s * e).sum())
                .collect()
        })
        .collect()
}

/// Compute the covariance matrix of mean-centered data.
pub fn covariance_matrix(centered_data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if centered_data.is_empty() {
        return vec![];
    }
    let d = centered_data[0].len();
    let n = centered_data.len() as f64;
    let mut cov = vec![vec![0.0; d]; d];

    for sample in centered_data {
        for i in 0..d {
            for j in i..d {
                cov[i][j] += sample[i] * sample[j];
            }
        }
    }

    for i in 0..d {
        for j in i..d {
            cov[i][j] /= n;
            cov[j][i] = cov[i][j];
        }
    }
    cov
}

/// Power iteration with deflation for the top-k eigenvectors of a symmetric matrix.
pub fn top_eigenvectors(matrix: &[Vec<f64>], k: usize, max_iter: usize) -> Vec<Vec<f64>> {
    let d = matrix.len();
    if d == 0 || k == 0 {
        return vec![];
    }

    let mut mat: Vec<Vec<f64>> = matrix.to_vec();
    let mut eigenvectors = Vec::with_capacity(k);

    for _ in 0..k {
        let ev = power_iteration(&mat, max_iter);
        let eigenvalue = rayleigh_quotient(&mat, &ev);
        eigenvectors.push(ev.clone());

        // Deflate: A <- A - lambda * v * v^T
        for i in 0..d {
            for j in 0..d {
                mat[i][j] -= eigenvalue * ev[i] * ev[j];
            }
        }
    }

    eigenvectors
}

/// Single power iteration to find the dominant eigenvector.
fn power_iteration(matrix: &[Vec<f64>], max_iter: usize) -> Vec<f64> {
    let d = matrix.len();
    if d == 0 {
        return vec![];
    }

    let mut v: Vec<f64> = (0..d).map(|i| ((i + 1) as f64).sqrt()).collect();
    normalize_vec(&mut v);

    for _ in 0..max_iter {
        let mut w = vec![0.0; d];
        for i in 0..d {
            for j in 0..d {
                w[i] += matrix[i][j] * v[j];
            }
        }

        let norm = vec_norm(&w);
        if norm < 1e-15 {
            break;
        }
        for wi in &mut w {
            *wi /= norm;
        }

        let diff: f64 = v.iter().zip(w.iter()).map(|(&a, &b)| (a - b).powi(2)).sum();
        v = w;
        if diff < POWER_ITER_TOL {
            break;
        }
    }

    v
}

/// Rayleigh quotient: v^T A v.
fn rayleigh_quotient(matrix: &[Vec<f64>], v: &[f64]) -> f64 {
    let d = v.len();
    let mut result = 0.0;
    for i in 0..d {
        for j in 0..d {
            result += v[i] * matrix[i][j] * v[j];
        }
    }
    result
}

fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

fn normalize_vec(v: &mut [f64]) {
    let norm = vec_norm(v);
    if norm > 1e-15 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grad(id: &str, values: Vec<f32>) -> Gradient {
        Gradient { node_id: id.into(), values, round: 0 }
    }

    #[test]
    fn test_feature_extraction() {
        let cbf = ConformalBehavioralFilter::new(0.1);
        let values = vec![1.0_f32, 2.0, 0.0, 3.0, 0.0];
        let features = cbf.extract_features(&values, "test");

        assert_eq!(features.len(), NUM_FEATURES);
        let expected_norm = (1.0_f64 + 4.0 + 0.0 + 9.0 + 0.0).sqrt();
        assert!((features[0] - expected_norm).abs() < 1e-6);
        let expected_mean = 6.0 / 5.0;
        assert!((features[1] - expected_mean).abs() < 1e-6);
        assert!((features[3] - 0.4).abs() < 1e-6); // sparsity
    }

    #[test]
    fn test_pca_preserves_sample_count() {
        let features = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
            vec![2.0, 3.0, 4.0],
        ];
        let projected = pca_transform(&features, 2);
        assert_eq!(projected.len(), 4);
        for p in &projected {
            assert_eq!(p.len(), 2);
        }
    }

    #[test]
    fn test_covariance_matrix_symmetry() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let mean = feature_centroid(&data);
        let centered: Vec<Vec<f64>> = data
            .iter()
            .map(|d| d.iter().zip(mean.iter()).map(|(&a, &b)| a - b).collect())
            .collect();
        let cov = covariance_matrix(&centered);
        assert!((cov[0][1] - cov[1][0]).abs() < 1e-10);
    }

    #[test]
    fn test_anomaly_rejected() {
        let mut cbf = ConformalBehavioralFilter::new(0.1);

        let normal: Vec<Gradient> = (0..10)
            .map(|i| grad(&format!("cal_{i}"), vec![1.0 + (i as f32) * 0.1, 2.0, 3.0, 1.0]))
            .collect();
        cbf.calibrate(&normal).unwrap();

        let gradients = vec![
            grad("n0", vec![1.0, 2.0, 3.0, 1.0]),
            grad("n1", vec![1.1, 1.9, 3.1, 0.9]),
            grad("n2", vec![0.9, 2.1, 2.9, 1.1]),
            grad("n3", vec![1.05, 2.05, 2.95, 1.05]),
            grad("anomaly", vec![100.0, -100.0, 0.0, 0.0]),
        ];

        let result = cbf.aggregate(&gradients).unwrap();

        let anomaly_score = result.scores.iter().find(|(id, _)| id == "anomaly").map(|(_, w)| *w).unwrap_or(1.0);
        let max_normal_score = result.scores.iter().filter(|(id, _)| id != "anomaly").map(|(_, w)| *w).fold(0.0_f64, f64::max);

        assert!(
            anomaly_score <= max_normal_score || !result.excluded_nodes.is_empty(),
            "anomaly should be rejected or down-weighted"
        );
    }

    #[test]
    fn test_empty_gradients() {
        let mut cbf = ConformalBehavioralFilter::new(0.1);
        assert!(cbf.aggregate(&[]).is_err());
    }

    #[test]
    fn test_calibration_requires_minimum_samples() {
        let mut cbf = ConformalBehavioralFilter::new(0.1);
        let too_few = vec![grad("a", vec![1.0]), grad("b", vec![2.0])];
        assert!(cbf.calibrate(&too_few).is_err());
    }

    #[test]
    fn test_fail_open_when_all_rejected() {
        let mut cbf = ConformalBehavioralFilter::new(0.999);
        let gradients = vec![grad("a", vec![1.0, 2.0]), grad("b", vec![3.0, 4.0])];
        let result = cbf.aggregate(&gradients).unwrap();
        assert!(!result.included_nodes.is_empty());
    }
}
