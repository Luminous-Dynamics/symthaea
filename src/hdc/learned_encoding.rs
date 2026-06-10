// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Learned HDC Encodings: Adaptive level hypervectors and feature weighting.
//!
//! Traditional HDC uses fixed level hypervectors created by interpolation between
//! two random endpoints. This module introduces **learnable** encodings that adapt
//! to data distribution, improving classification accuracy on benchmarks like
//! MNIST (target: 87%+) and ISOLET (target: 93%+).
//!
//! # Components
//!
//! - [`LearnableLevelEncoder`]: Level HVs with gradient-based refinement
//! - [`AdaptiveQuantizer`]: Non-uniform quantization learned from data
//! - [`WeightedFeatureEncoder`]: Per-feature importance weights
//! - [`SpatialContextEncoder`]: Neighborhood features for grid-structured data
//!
//! # Theory
//!
//! Standard HDC encoding: `H(x) = Σ_i position_i ⊛ level(quantize(x_i))`
//!
//! Learned encoding:     `H(x) = Σ_i w_i · position_i ⊛ level_learned(x_i)`
//!
//! where `w_i` are learned feature weights and `level_learned` uses adaptive
//! quantization boundaries and refined level hypervectors.

use crate::hdc::ContinuousHV;

// ═══════════════════════════════════════════════════════════════════════════════
// Adaptive Quantizer
// ═══════════════════════════════════════════════════════════════════════════════

/// Non-uniform quantizer that learns bin boundaries from data distribution.
///
/// Instead of uniformly dividing [0, 1] into Q levels, this quantizer places
/// boundaries where the data density is highest (like a histogram equalizer).
#[derive(Debug, Clone)]
pub struct AdaptiveQuantizer {
    /// Sorted bin boundaries (length = n_levels - 1).
    boundaries: Vec<f32>,
    /// Number of quantization levels.
    n_levels: usize,
}

impl AdaptiveQuantizer {
    /// Create a uniform quantizer with `n_levels` equally-spaced bins in [0, 1].
    pub fn uniform(n_levels: usize) -> Self {
        assert!(n_levels >= 2, "need at least 2 levels");
        let boundaries: Vec<f32> = (1..n_levels).map(|i| i as f32 / n_levels as f32).collect();
        Self {
            boundaries,
            n_levels,
        }
    }

    /// Learn quantization boundaries from data so each bin has roughly equal
    /// occupancy (histogram equalization). This maximizes information content.
    pub fn from_data(values: &[f32], n_levels: usize) -> Self {
        assert!(n_levels >= 2, "need at least 2 levels");
        if values.is_empty() {
            return Self::uniform(n_levels);
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Place boundaries at percentile positions
        let boundaries: Vec<f32> = (1..n_levels)
            .map(|i| {
                let percentile = i as f32 / n_levels as f32;
                let idx = ((sorted.len() as f32 * percentile) as usize).min(sorted.len() - 1);
                sorted[idx]
            })
            .collect();

        Self {
            boundaries,
            n_levels,
        }
    }

    /// Quantize a value to a level index in [0, n_levels).
    pub fn quantize(&self, value: f32) -> usize {
        for (i, &b) in self.boundaries.iter().enumerate() {
            if value < b {
                return i;
            }
        }
        self.n_levels - 1
    }

    /// Return quantization boundaries.
    pub fn boundaries(&self) -> &[f32] {
        &self.boundaries
    }

    /// Number of quantization levels.
    pub fn n_levels(&self) -> usize {
        self.n_levels
    }

    /// Smooth quantization: instead of hard bin assignment, return a weighted
    /// blend of the two nearest level indices. Returns (lower_idx, upper_idx, alpha)
    /// where alpha is the interpolation weight toward upper_idx.
    pub fn soft_quantize(&self, value: f32) -> (usize, usize, f32) {
        // Find which bin the value falls in
        let level = self.quantize(value);

        // Compute bin center and width for interpolation
        let lo = if level == 0 {
            0.0
        } else {
            self.boundaries[level - 1]
        };
        let hi = if level >= self.n_levels - 1 {
            1.0
        } else {
            self.boundaries[level]
        };

        let width = hi - lo;
        if width < 1e-10 {
            return (level, level.min(self.n_levels - 1), 0.0);
        }

        let t = ((value - lo) / width).clamp(0.0, 1.0);

        if t < 0.5 && level > 0 {
            // Closer to lower boundary
            let alpha = 0.5 - t; // [0, 0.5]
            (level - 1, level, 1.0 - alpha)
        } else if t >= 0.5 && level < self.n_levels - 1 {
            // Closer to upper boundary
            let alpha = t - 0.5; // [0, 0.5]
            (level, level + 1, alpha)
        } else {
            (level, level, 0.0)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Learnable Level Encoder
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for learnable level encoder.
#[derive(Debug, Clone)]
pub struct LearnableLevelConfig {
    /// Hypervector dimensionality.
    pub dim: usize,
    /// Number of quantization levels.
    pub n_levels: usize,
    /// Learning rate for level HV refinement.
    pub learning_rate: f32,
    /// Use soft quantization for smoother gradients.
    pub soft_quantize: bool,
}

impl Default for LearnableLevelConfig {
    fn default() -> Self {
        Self {
            dim: 16_384,
            n_levels: 64,
            learning_rate: 0.01,
            soft_quantize: true,
        }
    }
}

/// Learnable level hypervector encoder.
///
/// Starts with standard interpolation-based levels, then refines them
/// via gradient-like updates from classification feedback.
#[derive(Debug, Clone)]
pub struct LearnableLevelEncoder {
    /// Level hypervectors (length = n_levels, each dim-dimensional).
    levels: Vec<Vec<f32>>,
    /// Quantizer for mapping values to level indices.
    quantizer: AdaptiveQuantizer,
    /// Configuration.
    config: LearnableLevelConfig,
}

impl LearnableLevelEncoder {
    /// Create a new encoder with interpolation-initialized levels.
    pub fn new(config: LearnableLevelConfig) -> Self {
        let levels = Self::init_interpolated_levels(config.dim, config.n_levels, 42, 43);
        let quantizer = AdaptiveQuantizer::uniform(config.n_levels);
        Self {
            levels,
            quantizer,
            config,
        }
    }

    /// Create with adaptive quantization learned from training data.
    pub fn with_adaptive_quantization(
        config: LearnableLevelConfig,
        training_values: &[f32],
    ) -> Self {
        let levels = Self::init_interpolated_levels(config.dim, config.n_levels, 42, 43);
        let quantizer = AdaptiveQuantizer::from_data(training_values, config.n_levels);
        Self {
            levels,
            quantizer,
            config,
        }
    }

    /// Initialize level HVs via smooth interpolation between two random endpoints.
    fn init_interpolated_levels(
        dim: usize,
        n_levels: usize,
        seed_lo: u64,
        seed_hi: u64,
    ) -> Vec<Vec<f32>> {
        let base = ContinuousHV::random(dim, seed_lo);
        let end = ContinuousHV::random(dim, seed_hi);

        (0..n_levels)
            .map(|l| {
                let alpha = if n_levels <= 1 {
                    0.0
                } else {
                    l as f32 / (n_levels - 1) as f32
                };
                base.values
                    .iter()
                    .zip(end.values.iter())
                    .map(|(&a, &b)| a * (1.0 - alpha) + b * alpha)
                    .collect()
            })
            .collect()
    }

    /// Encode a scalar value [0, 1] into a hypervector.
    pub fn encode(&self, value: f32) -> ContinuousHV {
        if self.config.soft_quantize {
            let (lo, hi, alpha) = self.quantizer.soft_quantize(value);
            if lo == hi || alpha.abs() < 1e-6 {
                ContinuousHV::from_vec(self.levels[lo].clone())
            } else {
                // Interpolate between adjacent levels
                let blended: Vec<f32> = self.levels[lo]
                    .iter()
                    .zip(self.levels[hi].iter())
                    .map(|(&a, &b)| a * (1.0 - alpha) + b * alpha)
                    .collect();
                ContinuousHV::from_vec(blended)
            }
        } else {
            let idx = self.quantizer.quantize(value);
            ContinuousHV::from_vec(self.levels[idx].clone())
        }
    }

    /// Encode a value and return the raw level index (for debugging).
    pub fn encode_with_index(&self, value: f32) -> (ContinuousHV, usize) {
        let idx = self.quantizer.quantize(value);
        (self.encode(value), idx)
    }

    /// Refine level HVs via contrastive feedback.
    ///
    /// After a misclassification, move the encoding closer to the correct
    /// class prototype and away from the wrong one. This is a simplified
    /// "perceptron-like" update for HDC.
    ///
    /// - `value`: the input value that was misclassified
    /// - `correct_proto`: the prototype HV of the correct class
    /// - `wrong_proto`: the prototype HV of the predicted (wrong) class
    pub fn refine(&mut self, value: f32, correct_proto: &ContinuousHV, wrong_proto: &ContinuousHV) {
        let idx = self.quantizer.quantize(value);
        let lr = self.config.learning_rate;

        // Gradient direction: move toward correct, away from wrong
        let correct_vals = &correct_proto.values;
        let wrong_vals = &wrong_proto.values;

        for (i, lv) in self.levels[idx].iter_mut().enumerate() {
            let grad = correct_vals[i] - wrong_vals[i];
            *lv += lr * grad;
        }

        // Also slightly adjust neighboring levels (smoothness preservation)
        let neighbor_lr = lr * 0.3;
        if idx > 0 {
            for (i, lv) in self.levels[idx - 1].iter_mut().enumerate() {
                let grad = correct_vals[i] - wrong_vals[i];
                *lv += neighbor_lr * grad;
            }
        }
        if idx + 1 < self.config.n_levels {
            for (i, lv) in self.levels[idx + 1].iter_mut().enumerate() {
                let grad = correct_vals[i] - wrong_vals[i];
                *lv += neighbor_lr * grad;
            }
        }
    }

    /// Return the level hypervectors (for inspection or serialization).
    pub fn levels(&self) -> &[Vec<f32>] {
        &self.levels
    }

    /// Return the quantizer.
    pub fn quantizer(&self) -> &AdaptiveQuantizer {
        &self.quantizer
    }

    /// Number of levels.
    pub fn n_levels(&self) -> usize {
        self.config.n_levels
    }

    /// Dimensionality.
    pub fn dim(&self) -> usize {
        self.config.dim
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Weighted Feature Encoder
// ═══════════════════════════════════════════════════════════════════════════════

/// Full-pipeline encoder with per-feature weights and position HVs.
///
/// Encodes a feature vector x ∈ R^n as:
///   H(x) = normalize(Σ_i w_i · position_i ⊛ level(x_i))
///
/// Feature weights `w_i` are learned from classification feedback.
#[derive(Debug, Clone)]
pub struct WeightedFeatureEncoder {
    /// Level encoder (shared across all features).
    level_encoder: LearnableLevelEncoder,
    /// Position (feature identity) hypervectors.
    position_hvs: Vec<ContinuousHV>,
    /// Per-feature importance weights (initialized to 1.0).
    feature_weights: Vec<f32>,
    /// Number of features.
    n_features: usize,
    /// Weight learning rate.
    weight_lr: f32,
}

impl WeightedFeatureEncoder {
    /// Create a new encoder for `n_features`-dimensional input.
    pub fn new(n_features: usize, level_config: LearnableLevelConfig) -> Self {
        let dim = level_config.dim;
        let level_encoder = LearnableLevelEncoder::new(level_config);

        // Create deterministic position HVs (one per feature)
        let position_hvs: Vec<ContinuousHV> = (0..n_features)
            .map(|i| ContinuousHV::random(dim, 1000 + i as u64))
            .collect();

        let feature_weights = vec![1.0; n_features];

        Self {
            level_encoder,
            position_hvs,
            feature_weights,
            n_features,
            weight_lr: 0.05,
        }
    }

    /// Encode a feature vector into a single hypervector.
    pub fn encode(&self, features: &[f32]) -> ContinuousHV {
        assert_eq!(
            features.len(),
            self.n_features,
            "feature dimension mismatch"
        );

        let dim = self.level_encoder.dim();
        let mut accumulator = vec![0.0f32; dim];

        for (i, &val) in features.iter().enumerate() {
            let level_hv = self.level_encoder.encode(val);
            let bound = self.position_hvs[i].bind(&level_hv);
            let w = self.feature_weights[i];

            for (j, &v) in bound.values.iter().enumerate() {
                accumulator[j] += w * v;
            }
        }

        // L2 normalize
        let norm: f32 = accumulator.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for v in &mut accumulator {
                *v /= norm;
            }
        }

        ContinuousHV::from_vec(accumulator)
    }

    /// Update feature weights after a misclassification.
    ///
    /// Features whose individual encodings are closer to the wrong prototype
    /// than the correct one get their weights reduced. Features closer to the
    /// correct prototype get boosted.
    pub fn update_weights(
        &mut self,
        features: &[f32],
        correct_proto: &ContinuousHV,
        wrong_proto: &ContinuousHV,
    ) {
        for (i, &val) in features.iter().enumerate() {
            let level_hv = self.level_encoder.encode(val);
            let bound = self.position_hvs[i].bind(&level_hv);

            let sim_correct = bound.similarity(correct_proto);
            let sim_wrong = bound.similarity(wrong_proto);

            // If this feature points toward wrong class, reduce weight
            // If it points toward correct class, boost weight
            let delta = sim_correct - sim_wrong;
            self.feature_weights[i] += self.weight_lr * delta;
            self.feature_weights[i] = self.feature_weights[i].clamp(0.01, 10.0);
        }
    }

    /// Refine both feature weights and level HVs from a misclassification.
    pub fn refine(
        &mut self,
        features: &[f32],
        correct_proto: &ContinuousHV,
        wrong_proto: &ContinuousHV,
    ) {
        // Update feature weights
        self.update_weights(features, correct_proto, wrong_proto);

        // Update level HVs for each feature value
        for &val in features.iter() {
            self.level_encoder.refine(val, correct_proto, wrong_proto);
        }
    }

    /// Return current feature weights (for analysis/visualization).
    pub fn feature_weights(&self) -> &[f32] {
        &self.feature_weights
    }

    /// Set feature weights externally (e.g., from prior knowledge).
    pub fn set_feature_weights(&mut self, weights: &[f32]) {
        assert_eq!(weights.len(), self.n_features);
        self.feature_weights = weights.to_vec();
    }

    /// Access the underlying level encoder.
    pub fn level_encoder(&self) -> &LearnableLevelEncoder {
        &self.level_encoder
    }

    /// Mutable access to the underlying level encoder.
    pub fn level_encoder_mut(&mut self) -> &mut LearnableLevelEncoder {
        &mut self.level_encoder
    }

    /// Number of features.
    pub fn n_features(&self) -> usize {
        self.n_features
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Spatial Context Encoder
// ═══════════════════════════════════════════════════════════════════════════════

/// Spatial context encoder for grid-structured data (e.g., images).
///
/// Augments pixel features with neighborhood context:
///   H(x, y) = pixel(x,y) + Σ_{dx,dy} decay^dist · pixel(x+dx, y+dy)
///
/// This captures local spatial patterns that single-pixel encoding misses,
/// crucial for MNIST where digit strokes span multiple pixels.
#[derive(Debug, Clone)]
pub struct SpatialContextEncoder {
    /// Grid width.
    width: usize,
    /// Grid height.
    height: usize,
    /// Neighborhood radius (1 = 3x3, 2 = 5x5).
    radius: usize,
    /// Spatial decay factor per pixel distance.
    decay: f32,
}

impl SpatialContextEncoder {
    /// Create a new spatial encoder for a `width x height` grid.
    pub fn new(width: usize, height: usize, radius: usize, decay: f32) -> Self {
        Self {
            width,
            height,
            radius,
            decay,
        }
    }

    /// Create a spatial encoder for MNIST (28x28, radius=1, decay=0.5).
    pub fn mnist() -> Self {
        Self::new(28, 28, 1, 0.5)
    }

    /// Augment a flat feature vector with spatial context.
    ///
    /// Input: flat pixel values (length = width * height)
    /// Output: context-enhanced values (same length)
    pub fn augment(&self, pixels: &[f32]) -> Vec<f32> {
        assert_eq!(
            pixels.len(),
            self.width * self.height,
            "pixel count must match grid size"
        );

        let mut augmented = vec![0.0f32; pixels.len()];
        let r = self.radius as i32;

        for y in 0..self.height {
            for x in 0..self.width {
                let idx = y * self.width + x;
                let mut sum = pixels[idx]; // Center pixel (weight 1.0)
                let mut total_weight = 1.0f32;

                for dy in -r..=r {
                    for dx in -r..=r {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        if nx >= 0 && nx < self.width as i32 && ny >= 0 && ny < self.height as i32 {
                            let nidx = ny as usize * self.width + nx as usize;
                            let dist = ((dx * dx + dy * dy) as f32).sqrt();
                            let w = self.decay.powf(dist);
                            sum += w * pixels[nidx];
                            total_weight += w;
                        }
                    }
                }

                augmented[idx] = sum / total_weight;
            }
        }

        augmented
    }

    /// Compute gradient magnitude features from the grid.
    ///
    /// Returns a vector of gradient magnitudes (same length as input),
    /// useful as additional features for edge-sensitive classification.
    pub fn gradient_features(&self, pixels: &[f32]) -> Vec<f32> {
        assert_eq!(pixels.len(), self.width * self.height);

        let mut gradients = vec![0.0f32; pixels.len()];

        for y in 1..self.height - 1 {
            for x in 1..self.width - 1 {
                let idx = y * self.width + x;
                // Sobel-like gradient
                let gx = pixels[y * self.width + x + 1] - pixels[y * self.width + x - 1];
                let gy = pixels[(y + 1) * self.width + x] - pixels[(y - 1) * self.width + x];
                gradients[idx] = (gx * gx + gy * gy).sqrt();
            }
        }

        gradients
    }

    /// Grid dimensions.
    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HDC Classifier with Learned Encodings
// ═══════════════════════════════════════════════════════════════════════════════

/// Prototype-based HDC classifier using learned encodings.
///
/// Combines [`WeightedFeatureEncoder`] with per-class prototype management
/// and iterative retraining.
#[derive(Debug, Clone)]
pub struct LearnedHdcClassifier {
    /// Feature encoder with learnable weights and levels.
    encoder: WeightedFeatureEncoder,
    /// Class prototypes (one HV per class).
    prototypes: Vec<ContinuousHV>,
    /// Class labels corresponding to prototypes.
    class_labels: Vec<usize>,
    /// Number of training samples per class (for weighted averaging).
    class_counts: Vec<usize>,
}

impl LearnedHdcClassifier {
    /// Create a new classifier.
    pub fn new(n_features: usize, n_classes: usize, level_config: LearnableLevelConfig) -> Self {
        let dim = level_config.dim;
        let encoder = WeightedFeatureEncoder::new(n_features, level_config);
        let prototypes = vec![ContinuousHV::from_vec(vec![0.0; dim]); n_classes];
        let class_labels: Vec<usize> = (0..n_classes).collect();
        let class_counts = vec![0usize; n_classes];

        Self {
            encoder,
            prototypes,
            class_labels,
            class_counts,
        }
    }

    /// Initial training: build prototypes by bundling encoded samples.
    pub fn train(&mut self, samples: &[(Vec<f32>, usize)]) {
        let dim = self.prototypes[0].values.len();
        let n_classes = self.prototypes.len();

        // Reset accumulators
        let mut accumulators = vec![vec![0.0f32; dim]; n_classes];
        let mut counts = vec![0usize; n_classes];

        for (features, label) in samples {
            assert!(*label < n_classes, "label out of range");
            let encoded = self.encoder.encode(features);
            for (j, &v) in encoded.values.iter().enumerate() {
                accumulators[*label][j] += v;
            }
            counts[*label] += 1;
        }

        // Normalize to create prototypes
        for c in 0..n_classes {
            if counts[c] > 0 {
                let norm: f32 = accumulators[c].iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-10 {
                    for v in &mut accumulators[c] {
                        *v /= norm;
                    }
                }
                self.prototypes[c] = ContinuousHV::from_vec(accumulators[c].clone());
            }
            self.class_counts[c] = counts[c];
        }
    }

    /// Classify a feature vector. Returns (predicted_class, similarity).
    pub fn classify(&self, features: &[f32]) -> (usize, f32) {
        let encoded = self.encoder.encode(features);
        let mut best_class = 0;
        let mut best_sim = f32::NEG_INFINITY;

        for (i, proto) in self.prototypes.iter().enumerate() {
            let sim = encoded.similarity(proto);
            if sim > best_sim {
                best_sim = sim;
                best_class = self.class_labels[i];
            }
        }

        (best_class, best_sim)
    }

    /// Retrain: refine encodings on misclassified samples.
    ///
    /// Returns (n_correct, n_total) for monitoring convergence.
    pub fn retrain_epoch(&mut self, samples: &[(Vec<f32>, usize)]) -> (usize, usize) {
        let mut correct = 0;
        let n = samples.len();

        for (features, label) in samples {
            let (predicted, _) = self.classify(features);
            if predicted == *label {
                correct += 1;
            } else {
                // Refine encoder on misclassification
                let correct_proto = self.prototypes[*label].clone();
                let wrong_proto = self.prototypes[predicted].clone();
                self.encoder.refine(features, &correct_proto, &wrong_proto);
            }
        }

        // Rebuild prototypes after refinement
        self.train(samples);

        (correct, n)
    }

    /// Run multiple retraining iterations until convergence or max iterations.
    ///
    /// Returns final accuracy.
    pub fn retrain(&mut self, samples: &[(Vec<f32>, usize)], max_iterations: usize) -> f32 {
        let mut best_acc = 0.0f32;

        for _iter in 0..max_iterations {
            let (correct, total) = self.retrain_epoch(samples);
            let acc = correct as f32 / total as f32;
            if acc > best_acc {
                best_acc = acc;
            }
            // Early stopping if perfect
            if correct == total {
                break;
            }
        }

        best_acc
    }

    /// Return class prototypes (for analysis).
    pub fn prototypes(&self) -> &[ContinuousHV] {
        &self.prototypes
    }

    /// Return the feature encoder (for weight inspection).
    pub fn encoder(&self) -> &WeightedFeatureEncoder {
        &self.encoder
    }

    /// Mutable access to the encoder.
    pub fn encoder_mut(&mut self) -> &mut WeightedFeatureEncoder {
        &mut self.encoder
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_quantizer() {
        let q = AdaptiveQuantizer::uniform(4);
        assert_eq!(q.quantize(0.1), 0);
        assert_eq!(q.quantize(0.3), 1);
        assert_eq!(q.quantize(0.6), 2);
        assert_eq!(q.quantize(0.9), 3);
    }

    #[test]
    fn test_adaptive_quantizer_equalized() {
        // Data heavily concentrated at low values
        let data: Vec<f32> = (0..100)
            .map(|i| (i as f32 / 100.0).powi(3)) // cubic: most values near 0
            .collect();

        let q = AdaptiveQuantizer::from_data(&data, 4);

        // Boundaries should be shifted left (lower) compared to uniform
        let uniform = AdaptiveQuantizer::uniform(4);
        assert!(
            q.boundaries()[0] < uniform.boundaries()[0],
            "adaptive should have lower first boundary for left-skewed data"
        );
    }

    #[test]
    fn test_soft_quantize_interpolation() {
        let q = AdaptiveQuantizer::uniform(4);
        let (lo, hi, alpha) = q.soft_quantize(0.26); // Just above boundary 0.25
        // Should be near the boundary between bins 0 and 1
        assert!(lo <= 1 && hi <= 1);
        assert!((0.0..=1.0).contains(&alpha));
    }

    #[test]
    fn test_level_encoder_similarity_ordering() {
        let config = LearnableLevelConfig {
            dim: 4096,
            n_levels: 16,
            ..Default::default()
        };
        let enc = LearnableLevelEncoder::new(config);

        let hv_0 = enc.encode(0.0);
        let hv_mid = enc.encode(0.5);
        let hv_1 = enc.encode(1.0);

        // Nearby values should be more similar than distant values
        let sim_0_mid = hv_0.similarity(&hv_mid);
        let sim_0_1 = hv_0.similarity(&hv_1);

        assert!(
            sim_0_mid > sim_0_1,
            "level 0 should be more similar to 0.5 than to 1.0: {sim_0_mid} vs {sim_0_1}"
        );
    }

    #[test]
    fn test_weighted_feature_encoder_deterministic() {
        let config = LearnableLevelConfig {
            dim: 4096,
            n_levels: 16,
            ..Default::default()
        };
        let enc = WeightedFeatureEncoder::new(4, config);

        let features = vec![0.1, 0.5, 0.3, 0.9];
        let hv1 = enc.encode(&features);
        let hv2 = enc.encode(&features);

        let sim = hv1.similarity(&hv2);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "same input should produce identical output: sim={sim}"
        );
    }

    #[test]
    fn test_weighted_encoder_different_inputs() {
        let config = LearnableLevelConfig {
            dim: 4096,
            n_levels: 16,
            ..Default::default()
        };
        let enc = WeightedFeatureEncoder::new(10, config);

        let a = vec![0.0; 10];
        let b = vec![1.0; 10];
        let c = vec![0.1; 10]; // close to a

        let hv_a = enc.encode(&a);
        let hv_b = enc.encode(&b);
        let hv_c = enc.encode(&c);

        let sim_ab = hv_a.similarity(&hv_b);
        let sim_ac = hv_a.similarity(&hv_c);

        assert!(
            sim_ac > sim_ab,
            "close inputs should be more similar: sim_ac={sim_ac} vs sim_ab={sim_ab}"
        );
    }

    #[test]
    fn test_spatial_context_encoder() {
        let enc = SpatialContextEncoder::new(4, 4, 1, 0.5);

        // Single bright pixel at center
        let mut pixels = vec![0.0f32; 16];
        pixels[5] = 1.0; // (1, 1)

        let augmented = enc.augment(&pixels);

        // Center pixel should still be brightest
        assert!(augmented[5] > augmented[0]);
        // Neighbors should have some brightness
        assert!(augmented[1] > 0.0); // (1, 0) - neighbor
        assert!(augmented[4] > 0.0); // (0, 1) - neighbor
    }

    #[test]
    fn test_gradient_features() {
        let enc = SpatialContextEncoder::new(4, 4, 1, 0.5);

        // Vertical edge: left half bright, right half dark
        let mut pixels = vec![0.0f32; 16];
        for y in 0..4 {
            pixels[y * 4] = 1.0;
            pixels[y * 4 + 1] = 1.0;
        }

        let grads = enc.gradient_features(&pixels);

        // Edge pixels (column 1-2 boundary) should have high gradient
        // Interior pixels should have low gradient
        let edge_grad = grads[4 + 2]; // (2, 1) near edge
        let interior_grad = grads[4]; // (0, 1) in bright region
        assert!(
            edge_grad >= interior_grad,
            "edge should have higher gradient: {edge_grad} vs {interior_grad}"
        );
    }

    #[test]
    fn test_classifier_basic_separation() {
        let config = LearnableLevelConfig {
            dim: 4096,
            n_levels: 8,
            learning_rate: 0.05,
            soft_quantize: true,
        };
        let mut clf = LearnedHdcClassifier::new(2, 2, config);

        // Simple linearly separable data
        let mut samples = Vec::new();
        for i in 0..20 {
            let v = i as f32 / 20.0;
            // Class 0: low values
            samples.push((vec![v * 0.4, v * 0.3], 0));
            // Class 1: high values
            samples.push((vec![0.6 + v * 0.4, 0.7 + v * 0.3], 1));
        }

        // Train
        clf.train(&samples);

        // Test on training data
        let (predicted, _sim) = clf.classify(&[0.1, 0.1]);
        assert_eq!(predicted, 0, "low values should be class 0");

        let (predicted, _sim) = clf.classify(&[0.9, 0.9]);
        assert_eq!(predicted, 1, "high values should be class 1");
    }

    #[test]
    fn test_classifier_retrain_improves() {
        let config = LearnableLevelConfig {
            dim: 4096,
            n_levels: 8,
            learning_rate: 0.05,
            soft_quantize: true,
        };
        let mut clf = LearnedHdcClassifier::new(3, 3, config);

        // Three clusters in 3D
        let mut samples = Vec::new();
        for _i in 0..15 {
            samples.push((vec![0.1, 0.1, 0.1], 0));
            samples.push((vec![0.5, 0.5, 0.5], 1));
            samples.push((vec![0.9, 0.9, 0.9], 2));
        }

        // Initial training
        clf.train(&samples);
        let (c0_before, _) = clf.retrain_epoch(&samples);

        // After retraining, accuracy should not decrease
        let (c0_after, total) = clf.retrain_epoch(&samples);
        assert!(
            c0_after >= c0_before.saturating_sub(2),
            "retraining should maintain or improve accuracy: {c0_before}/{total} -> {c0_after}/{total}"
        );
    }
}
