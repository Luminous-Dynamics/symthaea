// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Recursive Least Squares (RLS) Online Learning
//!
//! Provides Ridge Regression-level accuracy with O(D²) per-sample updates.
//! Uses the Sherman-Morrison-Woodbury formula to maintain the inverse
//! correlation matrix incrementally.
//!
//! This bridges the gap between:
//! - Batch Ridge Regression (high accuracy, O(D³) at end)
//! - Online Prototype Learning (low accuracy, O(D) per sample)
//!
//! RLS gives us: High accuracy + O(D²) per sample (no final matrix inversion)

/// Recursive Least Squares learner for online discriminative classification
///
/// Maintains the inverse correlation matrix P and weight vectors for each class.
/// Mathematically equivalent to ridge regression, computed incrementally.
#[derive(Clone)]
pub struct RlsClassifier {
    /// Weight matrix: [n_classes × feature_dim]
    /// Each row is the weight vector for one class
    pub weights: Vec<Vec<f32>>,

    /// Inverse correlation matrix P: [feature_dim × feature_dim]
    /// Stored as flat row-major array for cache efficiency
    /// P ≈ (X^T X + λI)^{-1}
    p_matrix: Vec<f32>,

    /// Feature dimension
    pub feature_dim: usize,

    /// Number of classes
    pub n_classes: usize,

    /// Forgetting factor (1.0 = no forgetting, 0.99 = slight adaptation)
    pub lambda: f32,

    /// Regularization (initial P = (1/delta) * I)
    pub delta: f32,

    /// Class labels
    pub labels: Vec<String>,

    /// Sample counts per class (for diagnostics)
    pub counts: Vec<usize>,
}

impl RlsClassifier {
    /// Create a new RLS classifier
    ///
    /// # Arguments
    /// * `labels` - Class labels (phonemes)
    /// * `feature_dim` - Input feature dimension
    /// * `delta` - Regularization parameter (P initialized to I/delta)
    /// * `lambda` - Forgetting factor (1.0 = standard RLS, <1.0 = exponential forgetting)
    pub fn new(labels: Vec<String>, feature_dim: usize, delta: f32, lambda: f32) -> Self {
        let n_classes = labels.len();

        // Initialize weights to zero
        let weights = vec![vec![0.0f32; feature_dim]; n_classes];

        // Initialize P = (1/delta) * I (identity scaled by 1/delta)
        // This corresponds to a prior that weights are zero with variance delta
        let mut p_matrix = vec![0.0f32; feature_dim * feature_dim];
        let inv_delta = 1.0 / delta;
        for i in 0..feature_dim {
            p_matrix[i * feature_dim + i] = inv_delta;
        }

        Self {
            weights,
            p_matrix,
            feature_dim,
            n_classes,
            lambda,
            delta,
            labels,
            counts: vec![0; n_classes],
        }
    }

    /// Update the classifier with a new observation
    ///
    /// Uses Sherman-Morrison-Woodbury formula:
    /// 1. k = P * x / (λ + x^T * P * x)  (gain vector)
    /// 2. e = target - w^T * x           (prediction error)
    /// 3. w = w + k * e                  (weight update)
    /// 4. P = (P - k * x^T * P) / λ      (covariance update)
    ///
    /// # Arguments
    /// * `features` - Input feature vector `[feature_dim]`
    /// * `class_idx` - Target class index
    pub fn update(&mut self, features: &[f32], class_idx: usize) {
        if features.len() != self.feature_dim || class_idx >= self.n_classes {
            return;
        }

        self.counts[class_idx] += 1;
        let n = self.feature_dim;

        // Step 1: Compute P * x (intermediate vector)
        let mut px = vec![0.0f32; n];
        for i in 0..n {
            let mut sum = 0.0f32;
            for j in 0..n {
                sum += self.p_matrix[i * n + j] * features[j];
            }
            px[i] = sum;
        }

        // Step 2: Compute x^T * P * x (scalar denominator)
        let mut xpx = 0.0f32;
        for i in 0..n {
            xpx += features[i] * px[i];
        }

        // Step 3: Compute gain vector k = P*x / (λ + x^T*P*x)
        let denom = self.lambda + xpx;
        let mut k = vec![0.0f32; n];
        for i in 0..n {
            k[i] = px[i] / denom;
        }

        // Step 4: Update weights for each class
        // For the target class: target = 1, for others: target = 0
        for c in 0..self.n_classes {
            let target = if c == class_idx { 1.0f32 } else { 0.0f32 };

            // Compute prediction error: e = target - w^T * x
            let mut prediction = 0.0f32;
            for i in 0..n {
                prediction += self.weights[c][i] * features[i];
            }
            let error = target - prediction;

            // Update weights: w = w + k * e
            for i in 0..n {
                self.weights[c][i] += k[i] * error;
            }
        }

        // Step 5: Update P matrix: P = (P - k * x^T * P) / λ
        // This is the Sherman-Morrison update
        // P_new = (1/λ) * (P - k * (P*x)^T) = (1/λ) * (P - k * px^T)
        let inv_lambda = 1.0 / self.lambda;
        for i in 0..n {
            for j in 0..n {
                self.p_matrix[i * n + j] = inv_lambda * (self.p_matrix[i * n + j] - k[i] * px[j]);
            }
        }
    }

    /// Classify an observation
    ///
    /// Returns (class_idx, confidence_score)
    pub fn classify(&self, features: &[f32]) -> (usize, f32) {
        let logits = self.get_logits(features);

        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;

        for (idx, &score) in logits.iter().enumerate() {
            if score > best_score {
                best_score = score;
                best_idx = idx;
            }
        }

        (best_idx, best_score)
    }

    /// Get logits (raw scores) for all classes
    pub fn get_logits(&self, features: &[f32]) -> Vec<f32> {
        let mut logits = Vec::with_capacity(self.n_classes);

        for c in 0..self.n_classes {
            let mut score = 0.0f32;
            for i in 0..self.feature_dim.min(features.len()) {
                score += self.weights[c][i] * features[i];
            }
            logits.push(score);
        }

        logits
    }

    /// Get the label for a class index
    pub fn label(&self, class_idx: usize) -> Option<&str> {
        self.labels.get(class_idx).map(|s| s.as_str())
    }
}

/// Optimized RLS using rank-1 updates with pre-allocated buffers
///
/// Same algorithm as RlsClassifier but with better cache behavior
/// and reduced allocations for high-throughput training.
#[derive(Clone)]
pub struct FastRlsClassifier {
    /// Weight matrix: flat [n_classes * feature_dim]
    pub weights: Vec<f32>,

    /// Inverse correlation matrix P: flat [feature_dim * feature_dim]
    /// Uses f64 for numerical stability during 180K+ updates
    p_matrix: Vec<f64>,

    /// Pre-allocated buffer for P*x (f64 for stability)
    px_buffer: Vec<f64>,

    /// Pre-allocated buffer for gain vector k (f64 for stability)
    k_buffer: Vec<f64>,

    /// Feature dimension
    pub feature_dim: usize,

    /// Number of classes
    pub n_classes: usize,

    /// Forgetting factor
    pub lambda: f64,

    /// Class labels
    pub labels: Vec<String>,

    /// Sample counts
    pub counts: Vec<usize>,
}

impl FastRlsClassifier {
    /// Create a new fast RLS classifier
    pub fn new(labels: Vec<String>, feature_dim: usize, delta: f32, lambda: f32) -> Self {
        let n_classes = labels.len();

        // Flat weight matrix (f32 for memory efficiency)
        let weights = vec![0.0f32; n_classes * feature_dim];

        // Initialize P = (1/delta) * I using f64 for numerical stability
        let mut p_matrix = vec![0.0f64; feature_dim * feature_dim];
        let inv_delta = 1.0f64 / (delta as f64);
        for i in 0..feature_dim {
            p_matrix[i * feature_dim + i] = inv_delta;
        }

        Self {
            weights,
            p_matrix,
            px_buffer: vec![0.0f64; feature_dim],
            k_buffer: vec![0.0f64; feature_dim],
            feature_dim,
            n_classes,
            lambda: lambda as f64,
            labels,
            counts: vec![0; n_classes],
        }
    }

    /// Update with a new observation (optimized version with f64 stability)
    pub fn update(&mut self, features: &[f32], class_idx: usize) {
        if features.len() != self.feature_dim || class_idx >= self.n_classes {
            return;
        }

        self.counts[class_idx] += 1;
        let n = self.feature_dim;

        // Convert features to f64 for stable computation
        // Compute P * x into buffer
        for i in 0..n {
            let mut sum = 0.0f64;
            let row_start = i * n;
            for j in 0..n {
                sum += self.p_matrix[row_start + j] * (features[j] as f64);
            }
            self.px_buffer[i] = sum;
        }

        // Compute x^T * P * x
        let mut xpx = 0.0f64;
        for i in 0..n {
            xpx += (features[i] as f64) * self.px_buffer[i];
        }

        // Compute gain vector k with numerical stability check
        let denom = self.lambda + xpx;
        if denom.abs() < 1e-10 {
            return; // Skip update if denominator is too small
        }
        let inv_denom = 1.0 / denom;
        for i in 0..n {
            self.k_buffer[i] = self.px_buffer[i] * inv_denom;
        }

        // Update weights for each class (using f32 for weights, f64 for computation)
        for c in 0..self.n_classes {
            let target = if c == class_idx { 1.0f64 } else { 0.0f64 };

            // Compute prediction
            let w_offset = c * n;
            let mut prediction = 0.0f64;
            for i in 0..n {
                prediction += (self.weights[w_offset + i] as f64) * (features[i] as f64);
            }

            // Update: w += k * error
            let error = target - prediction;
            for i in 0..n {
                let delta_w = self.k_buffer[i] * error;
                self.weights[w_offset + i] += delta_w as f32;
            }
        }

        // Update P matrix: P = (P - k * px^T) / λ
        let inv_lambda = 1.0 / self.lambda;
        for i in 0..n {
            let row_start = i * n;
            let ki = self.k_buffer[i];
            for j in 0..n {
                self.p_matrix[row_start + j] =
                    inv_lambda * (self.p_matrix[row_start + j] - ki * self.px_buffer[j]);
            }
        }
    }

    /// Classify an observation
    pub fn classify(&self, features: &[f32]) -> (usize, f32) {
        let n = self.feature_dim.min(features.len());
        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;

        for c in 0..self.n_classes {
            let w_offset = c * self.feature_dim;
            let mut score = 0.0f32;
            for i in 0..n {
                score += self.weights[w_offset + i] * features[i];
            }
            if score > best_score {
                best_score = score;
                best_idx = c;
            }
        }

        (best_idx, best_score)
    }

    /// Get logits for all classes
    pub fn get_logits(&self, features: &[f32]) -> Vec<f32> {
        let n = self.feature_dim.min(features.len());
        let mut logits = Vec::with_capacity(self.n_classes);

        for c in 0..self.n_classes {
            let w_offset = c * self.feature_dim;
            let mut score = 0.0f32;
            for i in 0..n {
                score += self.weights[w_offset + i] * features[i];
            }
            logits.push(score);
        }

        logits
    }

    /// Get the label for a class index
    pub fn label(&self, class_idx: usize) -> Option<&str> {
        self.labels.get(class_idx).map(|s| s.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rls_basic() {
        let labels = vec!["A".into(), "B".into()];
        let mut rls = RlsClassifier::new(labels, 4, 1.0, 1.0);

        // Train with some separable data
        for _ in 0..100 {
            rls.update(&[1.0, 0.0, 0.0, 0.0], 0); // Class A
            rls.update(&[0.0, 1.0, 0.0, 0.0], 1); // Class B
        }

        // Test classification
        let (class_a, _) = rls.classify(&[0.9, 0.1, 0.0, 0.0]);
        let (class_b, _) = rls.classify(&[0.1, 0.9, 0.0, 0.0]);

        assert_eq!(class_a, 0, "Should classify as A");
        assert_eq!(class_b, 1, "Should classify as B");
    }

    #[test]
    fn test_fast_rls_matches_standard() {
        let labels = vec!["X".into(), "Y".into(), "Z".into()];
        let mut rls = RlsClassifier::new(labels.clone(), 8, 0.1, 1.0);
        let mut fast = FastRlsClassifier::new(labels, 8, 0.1, 1.0);

        // Train both with same data
        let samples = [
            ([1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0),
            ([0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0], 1),
            ([0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0], 2),
        ];

        for _ in 0..50 {
            for (features, class) in &samples {
                rls.update(features, *class);
                fast.update(features, *class);
            }
        }

        // Both should give same classification
        for (features, expected) in &samples {
            let (c1, _) = rls.classify(features);
            let (c2, _) = fast.classify(features);
            assert_eq!(c1, *expected);
            assert_eq!(c2, *expected);
        }
    }
}
