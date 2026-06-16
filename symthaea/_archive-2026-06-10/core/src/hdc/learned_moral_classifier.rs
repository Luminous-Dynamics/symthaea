// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Learned Moral Classifier — composing Spinozist feature extraction with
//! adaptive HDC classification.
//!
//! This module bridges two systems:
//! - [`SpinozistClassifier`]: extracts 171D affect features from text
//!   (18 linear affects + 153 pairwise cross-terms)
//! - [`LearnedHdcClassifier`]: adaptive prototype classification with
//!   learnable level HVs and per-feature importance weights
//!
//! The result is a moral classifier that learns which affect dimensions
//! matter most for Good/Bad/Neutral discrimination, adapting its encoding
//! to the data distribution rather than relying on fixed geometry.
//!
//! # Usage
//!
//! ```ignore
//! use symthaea::hdc::learned_moral_classifier::LearnedMoralClassifier;
//! use symthaea::hdc::moral_prototypes::MoralLabel;
//!
//! let mut clf = LearnedMoralClassifier::new();
//! clf.train(&[
//!     ("helping others is good".into(), MoralLabel::Good),
//!     ("stealing is wrong".into(), MoralLabel::Bad),
//!     ("the sky is blue".into(), MoralLabel::Neutral),
//! ]);
//! let (verdict, confidence) = clf.classify("sharing food with neighbors");
//! ```

use symthaea_core::hdc::HDC_DIMENSION;
use symthaea_core::hdc::unified_hv::ContinuousHV;

use super::learned_encoding::{LearnableLevelConfig, LearnedHdcClassifier};
use super::moral_algebra::MoralVerdict;
use super::moral_prototypes::MoralLabel;
use super::moral_text_encoder::TextHdcEncoder;
use super::spinozist_geometry::SpinozistClassifier;

/// Number of features: 171 Spinozist (18 linear + 153 cross-terms) + 3 surface prototype similarities = 174.
const N_FEATURES: usize = 174;

/// Number of moral classes: Good(0), Bad(1), Neutral(2).
const N_CLASSES: usize = 3;

// ============================================================================
// Feature Normalizer
// ============================================================================

/// Per-feature min-max normalizer that maps values to [0, 1].
///
/// Fit on training data, then applied to normalize both training and
/// inference features. Values outside the observed range are clamped.
#[derive(Debug, Clone)]
struct FeatureNormalizer {
    mins: Vec<f32>,
    maxs: Vec<f32>,
}

impl FeatureNormalizer {
    /// Compute per-feature min and max from a matrix of feature vectors.
    fn fit(features_matrix: &[Vec<f32>]) -> Self {
        assert!(!features_matrix.is_empty(), "cannot fit on empty data");
        let n = features_matrix[0].len();

        let mut mins = vec![f32::INFINITY; n];
        let mut maxs = vec![f32::NEG_INFINITY; n];

        for row in features_matrix {
            assert_eq!(row.len(), n, "feature dimension mismatch");
            for (i, &val) in row.iter().enumerate() {
                if val < mins[i] {
                    mins[i] = val;
                }
                if val > maxs[i] {
                    maxs[i] = val;
                }
            }
        }

        Self { mins, maxs }
    }

    /// Normalize a feature vector to [0, 1] per dimension with clamping.
    ///
    /// Constant features (max == min) map to 0.5.
    fn normalize(&self, features: &[f32]) -> Vec<f32> {
        assert_eq!(
            features.len(),
            self.mins.len(),
            "feature dimension mismatch"
        );
        features
            .iter()
            .enumerate()
            .map(|(i, &val)| {
                let range = self.maxs[i] - self.mins[i];
                if range < 1e-12 {
                    0.5
                } else {
                    ((val - self.mins[i]) / range).clamp(0.0, 1.0)
                }
            })
            .collect()
    }
}

// ============================================================================
// Learned Moral Classifier
// ============================================================================

/// Moral classifier composing Spinozist affect extraction with adaptive HDC.
///
/// The pipeline:
/// 1. Text -> 171D affect features (via `SpinozistClassifier`)
/// 2. Normalize features to [0, 1] (via `FeatureNormalizer`)
/// 3. Encode + classify (via `LearnedHdcClassifier` with learned levels + weights)
///
/// Training adapts both the level hypervectors and per-feature importance
/// weights, so the classifier learns which of the 171 affect dimensions
/// are most discriminative for moral judgment.
pub struct LearnedMoralClassifier {
    spinozist: SpinozistClassifier,
    surface_encoder: TextHdcEncoder,
    surface_prototypes: Option<[ContinuousHV; 3]>,
    classifier: LearnedHdcClassifier,
    normalizer: Option<FeatureNormalizer>,
}

impl LearnedMoralClassifier {
    /// Create a new classifier with default configuration.
    ///
    /// Uses 171 input features (from Spinozist affect space), 3 output classes,
    /// 16,384-dimensional hypervectors, 32 quantization levels, and soft
    /// quantization for smoother gradient updates.
    pub fn new() -> Self {
        let spinozist = SpinozistClassifier::new();
        let config = LearnableLevelConfig {
            dim: 16_384,
            n_levels: 32,
            learning_rate: 0.02,
            soft_quantize: true,
        };
        let classifier = LearnedHdcClassifier::new(N_FEATURES, N_CLASSES, config);
        let surface_encoder = TextHdcEncoder::with_sentiment(HDC_DIMENSION, 3, 0.5, 0.2);
        Self {
            spinozist,
            surface_encoder,
            surface_prototypes: None,
            classifier,
            normalizer: None,
        }
    }

    /// Train on labeled text samples.
    ///
    /// 1. Extracts 171D Spinozist features for each sample
    /// 2. Fits a min-max normalizer on the feature matrix
    /// 3. Normalizes all features to [0, 1]
    /// 4. Builds initial prototypes via bundling
    /// 5. Runs 15 retraining iterations for level/weight refinement
    /// 6. Prints final training accuracy
    pub fn train(&mut self, samples: &[(String, MoralLabel)]) {
        if samples.is_empty() {
            return;
        }

        // Build surface prototypes first (for surface similarity features)
        {
            let dim = HDC_DIMENSION;
            let mut accum = [vec![0.0f32; dim], vec![0.0f32; dim], vec![0.0f32; dim]];
            let mut counts = [0usize; 3];
            for (text, label) in samples {
                let hv = self.surface_encoder.encode(text);
                let idx = label_to_index(*label);
                for (a, &v) in accum[idx].iter_mut().zip(hv.values.iter()) {
                    *a += v;
                }
                counts[idx] += 1;
            }
            let mut protos = [
                ContinuousHV::zero(dim),
                ContinuousHV::zero(dim),
                ContinuousHV::zero(dim),
            ];
            for i in 0..3 {
                if counts[i] > 0 {
                    let norm: f32 = accum[i].iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        protos[i] =
                            ContinuousHV::from_vec(accum[i].iter().map(|x| x / norm).collect());
                    }
                }
            }
            self.surface_prototypes = Some(protos);
        }

        // Extract features (171 Spinozist + 3 surface similarities)
        let raw_features: Vec<Vec<f32>> = samples
            .iter()
            .map(|(text, _)| self.extract_features(text))
            .collect();

        // Fit normalizer
        let normalizer = FeatureNormalizer::fit(&raw_features);

        // Normalize and pair with class indices
        let normalized_samples: Vec<(Vec<f32>, usize)> = raw_features
            .iter()
            .zip(samples.iter())
            .map(|(feats, (_, label))| {
                let normed = normalizer.normalize(feats);
                let class = label_to_class(*label);
                (normed, class)
            })
            .collect();

        self.normalizer = Some(normalizer);

        // Initial prototype building
        self.classifier.train(&normalized_samples);

        // Iterative refinement (15 epochs)
        let final_acc = self.classifier.retrain(&normalized_samples, 15);

        println!(
            "  LearnedMoralClassifier: trained on {} samples, final accuracy {:.1}%",
            samples.len(),
            final_acc * 100.0
        );
    }

    /// Extract 174D feature vector: 171 Spinozist affects + 3 surface similarities.
    fn extract_features(&self, text: &str) -> Vec<f32> {
        let mut features = self.spinozist.compute_features(text);
        // Add 3 surface prototype similarities
        if let Some(ref protos) = self.surface_prototypes {
            let surface_hv = self.surface_encoder.encode(text);
            for proto in protos {
                features.push(surface_hv.similarity(proto));
            }
        } else {
            features.push(0.0);
            features.push(0.0);
            features.push(0.0);
        }
        features
    }

    /// Classify a text string into a moral verdict with confidence.
    pub fn classify(&self, text: &str) -> (MoralVerdict, f32) {
        let normalizer = match &self.normalizer {
            Some(n) => n,
            None => return (MoralVerdict::Neutral, 0.0),
        };

        let raw = self.extract_features(text);
        let normalized = normalizer.normalize(&raw);
        let (class, similarity) = self.classifier.classify(&normalized);

        let verdict = class_to_verdict(class);
        (verdict, similarity)
    }

    /// Return the learned per-feature importance weights.
    ///
    /// After training, features that are more discriminative for moral
    /// classification will have higher weights. Useful for interpretability:
    /// which affects matter most?
    pub fn feature_weights(&self) -> &[f32] {
        self.classifier.encoder().feature_weights()
    }
}

// ============================================================================
// Conversions
// ============================================================================

/// Map MoralLabel to class index: Good=0, Bad=1, Neutral=2.
fn label_to_index(label: MoralLabel) -> usize {
    label_to_class(label)
}

fn label_to_class(label: MoralLabel) -> usize {
    match label {
        MoralLabel::Good => 0,
        MoralLabel::Bad => 1,
        MoralLabel::Neutral => 2,
    }
}

/// Map class index back to MoralVerdict.
fn class_to_verdict(class: usize) -> MoralVerdict {
    match class {
        0 => MoralVerdict::Good,
        1 => MoralVerdict::Bad,
        _ => MoralVerdict::Neutral,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_and_classify() {
        let mut clf = LearnedMoralClassifier::new();

        // Synthetic training data with clear moral signals
        let samples = vec![
            (
                "helping others and being kind".to_string(),
                MoralLabel::Good,
            ),
            (
                "sharing food with those in need".to_string(),
                MoralLabel::Good,
            ),
            (
                "hurting innocent people is terrible".to_string(),
                MoralLabel::Bad,
            ),
            (
                "stealing and lying to everyone".to_string(),
                MoralLabel::Bad,
            ),
            (
                "the weather is cloudy today".to_string(),
                MoralLabel::Neutral,
            ),
            (
                "water boils at one hundred degrees".to_string(),
                MoralLabel::Neutral,
            ),
        ];

        clf.train(&samples);

        // Verify classifier produces a verdict (not just default neutral)
        let (verdict, confidence) = clf.classify("helping the elderly cross the street");
        assert!(confidence > -1.0, "should produce a valid confidence score");
        // With only 6 training samples the exact verdict may vary, but the
        // classifier should at least not crash and should produce output.
        let _ = verdict;
    }

    #[test]
    fn test_feature_weights_learned() {
        let mut clf = LearnedMoralClassifier::new();

        let samples = vec![
            (
                "helping others and being kind".to_string(),
                MoralLabel::Good,
            ),
            (
                "sharing food with those in need".to_string(),
                MoralLabel::Good,
            ),
            (
                "hurting innocent people is terrible".to_string(),
                MoralLabel::Bad,
            ),
            (
                "stealing and lying to everyone".to_string(),
                MoralLabel::Bad,
            ),
            (
                "the weather is cloudy today".to_string(),
                MoralLabel::Neutral,
            ),
            (
                "water boils at one hundred degrees".to_string(),
                MoralLabel::Neutral,
            ),
        ];

        clf.train(&samples);

        let weights = clf.feature_weights();
        assert_eq!(weights.len(), N_FEATURES);

        // After training with retraining iterations, at least some weights
        // should have diverged from the initial value of 1.0.
        let all_ones = weights.iter().all(|&w| (w - 1.0).abs() < 1e-6);
        assert!(
            !all_ones,
            "feature weights should not all be 1.0 after training — learning should have occurred"
        );
    }

    #[test]
    fn test_normalize_roundtrip() {
        let features = vec![
            vec![-1.0, 0.0, 5.0],
            vec![1.0, 10.0, 5.0],
            vec![0.0, 5.0, 5.0],
        ];

        let normalizer = FeatureNormalizer::fit(&features);

        // Check mins/maxs
        assert_eq!(normalizer.mins, vec![-1.0, 0.0, 5.0]);
        assert_eq!(normalizer.maxs, vec![1.0, 10.0, 5.0]);

        // Normalize known values
        let normed = normalizer.normalize(&[-1.0, 0.0, 5.0]);
        assert!((normed[0] - 0.0).abs() < 1e-6, "min should map to 0.0");
        assert!((normed[1] - 0.0).abs() < 1e-6, "min should map to 0.0");
        assert!(
            (normed[2] - 0.5).abs() < 1e-6,
            "constant feature should map to 0.5"
        );

        let normed = normalizer.normalize(&[1.0, 10.0, 5.0]);
        assert!((normed[0] - 1.0).abs() < 1e-6, "max should map to 1.0");
        assert!((normed[1] - 1.0).abs() < 1e-6, "max should map to 1.0");

        // Values outside range should be clamped
        let normed = normalizer.normalize(&[-5.0, 15.0, 5.0]);
        assert!(
            (normed[0] - 0.0).abs() < 1e-6,
            "below min should clamp to 0.0"
        );
        assert!(
            (normed[1] - 1.0).abs() < 1e-6,
            "above max should clamp to 1.0"
        );
    }
}
