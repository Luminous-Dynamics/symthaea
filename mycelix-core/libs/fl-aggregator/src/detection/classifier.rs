//! ML Classifiers for Byzantine Detection.
//!
//! Provides multiple classifier implementations:
//! - **ThresholdClassifier**: Fast rule-based classification
//! - **DecisionTreeClassifier**: Simple tree-based classification
//! - **EnsembleClassifier**: Combines multiple classifiers
//!
//! For production, these can be replaced with linfa-based SVM/RF or
//! models loaded from Python sklearn exports.

use crate::detection::CompositeFeatures;
use ndarray::{Array1, Array2, ArrayView1};
use serde::{Deserialize, Serialize};

/// Classification result.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClassifierOutput {
    /// Node appears to be honest.
    Honest = 0,
    /// Node appears to be Byzantine.
    Byzantine = 1,
}

/// Classifier type enumeration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ClassifierType {
    /// Simple threshold-based rules.
    Threshold,
    /// Decision tree classifier.
    DecisionTree,
    /// Ensemble of multiple classifiers.
    Ensemble,
}

/// Configuration for classifiers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClassifierConfig {
    /// Type of classifier.
    pub classifier_type: ClassifierType,

    /// Thresholds for threshold-based classifier.
    pub thresholds: ThresholdConfig,

    /// Random state for reproducibility.
    pub random_state: u64,
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        Self {
            classifier_type: ClassifierType::Ensemble,
            thresholds: ThresholdConfig::default(),
            random_state: 42,
        }
    }
}

/// Thresholds for rule-based classification.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThresholdConfig {
    /// PoGQ below this → Byzantine.
    pub pogq_byzantine: f32,
    /// TCDM below this → Byzantine.
    pub tcdm_byzantine: f32,
    /// Z-score above this → Byzantine.
    pub zscore_byzantine: f32,
    /// Entropy below this → Byzantine (concentrated attack pattern).
    pub entropy_byzantine: f32,
    /// Gradient norm above this → Byzantine.
    pub norm_byzantine: f32,
}

impl Default for ThresholdConfig {
    fn default() -> Self {
        Self {
            pogq_byzantine: 0.3,
            tcdm_byzantine: 0.3,
            zscore_byzantine: 10.0,
            entropy_byzantine: 0.5,
            norm_byzantine: 100.0,
        }
    }
}

/// Trait for classifiers.
pub trait Classifier: Send + Sync {
    /// Train the classifier on labeled data.
    fn fit(&mut self, features: &Array2<f32>, labels: &Array1<u8>);

    /// Predict class for a single sample.
    fn predict(&self, features: ArrayView1<f32>) -> ClassifierOutput;

    /// Predict probability of Byzantine (class 1).
    fn predict_proba(&self, features: ArrayView1<f32>) -> f32;

    /// Check if classifier is trained.
    fn is_fitted(&self) -> bool;
}

/// Threshold-based classifier using simple rules.
#[derive(Clone, Debug)]
pub struct ThresholdClassifier {
    config: ThresholdConfig,
    /// Learned weights for each feature (from training data).
    weights: Option<Array1<f32>>,
    fitted: bool,
}

impl ThresholdClassifier {
    /// Create a new threshold classifier.
    pub fn new(config: ThresholdConfig) -> Self {
        Self {
            config,
            weights: None,
            fitted: false,
        }
    }

    /// Classify using composite features directly.
    pub fn classify_features(&self, features: &CompositeFeatures) -> ClassifierOutput {
        // Rule-based classification
        let byzantine_score = self.compute_byzantine_score(features);
        if byzantine_score > 0.5 {
            ClassifierOutput::Byzantine
        } else {
            ClassifierOutput::Honest
        }
    }

    /// Compute Byzantine probability score.
    fn compute_byzantine_score(&self, features: &CompositeFeatures) -> f32 {
        let mut score = 0.0f32;
        let mut count = 0;

        // PoGQ: low = Byzantine
        if features.pogq_score < self.config.pogq_byzantine {
            score += 1.0;
        } else if features.pogq_score < 0.5 {
            score += 0.5;
        }
        count += 1;

        // TCDM: low = Byzantine
        if features.tcdm_score < self.config.tcdm_byzantine {
            score += 1.0;
        } else if features.tcdm_score < 0.5 {
            score += 0.3;
        }
        count += 1;

        // Z-score: high = Byzantine
        if features.zscore_magnitude > self.config.zscore_byzantine {
            score += 1.0;
        } else if features.zscore_magnitude > 5.0 {
            score += 0.5;
        }
        count += 1;

        // Entropy: low = Byzantine (concentrated attack)
        if features.entropy_score < self.config.entropy_byzantine {
            score += 0.5;
        }
        count += 1;

        // Norm: very high = Byzantine
        if features.gradient_norm > self.config.norm_byzantine {
            score += 0.8;
        }
        count += 1;

        score / count as f32
    }
}

impl Classifier for ThresholdClassifier {
    fn fit(&mut self, features: &Array2<f32>, _labels: &Array1<u8>) {
        // Learn optimal thresholds from data
        // For now, use default thresholds
        let n_features = features.ncols();
        self.weights = Some(Array1::ones(n_features) / n_features as f32);
        self.fitted = true;
    }

    fn predict(&self, features: ArrayView1<f32>) -> ClassifierOutput {
        let proba = self.predict_proba(features);
        if proba > 0.5 {
            ClassifierOutput::Byzantine
        } else {
            ClassifierOutput::Honest
        }
    }

    fn predict_proba(&self, features: ArrayView1<f32>) -> f32 {
        // features: [pogq, tcdm, zscore, entropy, norm]
        let mut score = 0.0f32;

        // PoGQ: invert (low pogq = high byzantine score)
        score += (1.0 - features[0]).max(0.0) * 0.3;

        // TCDM: invert
        score += (1.0 - features[1]).max(0.0) * 0.2;

        // Z-score: normalize to [0, 1] range
        let zscore_norm = (features[2] / 20.0).clamp(0.0, 1.0);
        score += zscore_norm * 0.25;

        // Entropy: invert (low entropy = suspicious)
        let entropy_norm = (1.0 - features[3] / 5.0).clamp(0.0, 1.0);
        score += entropy_norm * 0.1;

        // Norm: normalize
        let norm_score = (features[4] / 100.0).clamp(0.0, 1.0);
        score += norm_score * 0.15;

        score.clamp(0.0, 1.0)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

/// Simple decision tree node.
#[derive(Clone, Debug, Serialize, Deserialize)]
enum TreeNode {
    /// Leaf node with class prediction and confidence.
    Leaf {
        class: u8,
        confidence: f32,
    },
    /// Split node.
    Split {
        feature_idx: usize,
        threshold: f32,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

/// Decision tree classifier.
#[derive(Clone, Debug)]
pub struct DecisionTreeClassifier {
    root: Option<TreeNode>,
    #[allow(dead_code)]
    max_depth: usize,
    #[allow(dead_code)]
    min_samples_split: usize,
    fitted: bool,
}

impl DecisionTreeClassifier {
    /// Create a new decision tree classifier.
    pub fn new(max_depth: usize, min_samples_split: usize) -> Self {
        Self {
            root: None,
            max_depth,
            min_samples_split,
            fitted: false,
        }
    }

    /// Build a simple decision tree based on domain knowledge.
    fn build_default_tree() -> TreeNode {
        // Hand-crafted tree based on feature importance
        TreeNode::Split {
            feature_idx: 0, // PoGQ
            threshold: 0.3,
            left: Box::new(TreeNode::Leaf {
                class: 1,
                confidence: 0.95,
            }), // Low PoGQ = Byzantine
            right: Box::new(TreeNode::Split {
                feature_idx: 2, // Z-score
                threshold: 10.0,
                left: Box::new(TreeNode::Split {
                    feature_idx: 1, // TCDM
                    threshold: 0.4,
                    left: Box::new(TreeNode::Leaf {
                        class: 1,
                        confidence: 0.75,
                    }),
                    right: Box::new(TreeNode::Leaf {
                        class: 0,
                        confidence: 0.85,
                    }),
                }),
                right: Box::new(TreeNode::Leaf {
                    class: 1,
                    confidence: 0.90,
                }), // High Z-score = Byzantine
            }),
        }
    }

    /// Traverse tree to get prediction.
    fn traverse(&self, node: &TreeNode, features: ArrayView1<f32>) -> (u8, f32) {
        match node {
            TreeNode::Leaf { class, confidence } => (*class, *confidence),
            TreeNode::Split {
                feature_idx,
                threshold,
                left,
                right,
            } => {
                if features[*feature_idx] < *threshold {
                    self.traverse(left, features)
                } else {
                    self.traverse(right, features)
                }
            }
        }
    }
}

impl Default for DecisionTreeClassifier {
    fn default() -> Self {
        Self::new(5, 2)
    }
}

impl Classifier for DecisionTreeClassifier {
    fn fit(&mut self, _features: &Array2<f32>, _labels: &Array1<u8>) {
        // For now, use default domain-knowledge tree
        // Full implementation would use ID3/CART algorithm
        self.root = Some(Self::build_default_tree());
        self.fitted = true;
    }

    fn predict(&self, features: ArrayView1<f32>) -> ClassifierOutput {
        let (class, _) = match &self.root {
            Some(root) => self.traverse(root, features),
            None => (0, 0.5), // Default to honest if not fitted
        };

        if class == 1 {
            ClassifierOutput::Byzantine
        } else {
            ClassifierOutput::Honest
        }
    }

    fn predict_proba(&self, features: ArrayView1<f32>) -> f32 {
        let (class, confidence) = match &self.root {
            Some(root) => self.traverse(root, features),
            None => return 0.5,
        };

        if class == 1 {
            confidence
        } else {
            1.0 - confidence
        }
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

/// Ensemble classifier combining multiple classifiers.
pub struct EnsembleClassifier {
    classifiers: Vec<Box<dyn Classifier>>,
    weights: Vec<f32>,
    voting: VotingStrategy,
    fitted: bool,
}

/// Voting strategy for ensemble.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum VotingStrategy {
    /// Hard voting (majority).
    Hard,
    /// Soft voting (average probabilities).
    Soft,
    /// Weighted soft voting.
    WeightedSoft,
}

impl EnsembleClassifier {
    /// Create a new ensemble classifier.
    pub fn new(voting: VotingStrategy) -> Self {
        Self {
            classifiers: Vec::new(),
            weights: Vec::new(),
            voting,
            fitted: false,
        }
    }

    /// Create default ensemble with threshold and decision tree classifiers.
    pub fn default_ensemble() -> Self {
        let mut ensemble = Self::new(VotingStrategy::Soft);

        // Add threshold classifier
        ensemble.add_classifier(
            Box::new(ThresholdClassifier::new(ThresholdConfig::default())),
            0.4,
        );

        // Add decision tree classifier
        ensemble.add_classifier(Box::new(DecisionTreeClassifier::default()), 0.6);

        ensemble
    }

    /// Add a classifier to the ensemble.
    pub fn add_classifier(&mut self, classifier: Box<dyn Classifier>, weight: f32) {
        self.classifiers.push(classifier);
        self.weights.push(weight);
    }
}

impl Classifier for EnsembleClassifier {
    fn fit(&mut self, features: &Array2<f32>, labels: &Array1<u8>) {
        for classifier in &mut self.classifiers {
            classifier.fit(features, labels);
        }
        self.fitted = true;
    }

    fn predict(&self, features: ArrayView1<f32>) -> ClassifierOutput {
        let proba = self.predict_proba(features);
        if proba > 0.5 {
            ClassifierOutput::Byzantine
        } else {
            ClassifierOutput::Honest
        }
    }

    fn predict_proba(&self, features: ArrayView1<f32>) -> f32 {
        if self.classifiers.is_empty() {
            return 0.5;
        }

        match self.voting {
            VotingStrategy::Hard => {
                let votes: f32 = self
                    .classifiers
                    .iter()
                    .map(|c| {
                        if c.predict(features) == ClassifierOutput::Byzantine {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .sum();
                votes / self.classifiers.len() as f32
            }
            VotingStrategy::Soft => {
                let sum: f32 = self
                    .classifiers
                    .iter()
                    .map(|c| c.predict_proba(features))
                    .sum();
                sum / self.classifiers.len() as f32
            }
            VotingStrategy::WeightedSoft => {
                let total_weight: f32 = self.weights.iter().sum();
                if total_weight < 1e-10 {
                    return 0.5;
                }

                let weighted_sum: f32 = self
                    .classifiers
                    .iter()
                    .zip(self.weights.iter())
                    .map(|(c, w)| c.predict_proba(features) * w)
                    .sum();

                weighted_sum / total_weight
            }
        }
    }

    fn is_fitted(&self) -> bool {
        self.fitted && self.classifiers.iter().all(|c| c.is_fitted())
    }
}

/// Create a classifier based on type.
pub fn create_classifier(config: &ClassifierConfig) -> Box<dyn Classifier> {
    match config.classifier_type {
        ClassifierType::Threshold => {
            Box::new(ThresholdClassifier::new(config.thresholds.clone()))
        }
        ClassifierType::DecisionTree => Box::new(DecisionTreeClassifier::default()),
        ClassifierType::Ensemble => Box::new(EnsembleClassifier::default_ensemble()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_features_honest() -> Array1<f32> {
        // High PoGQ, high TCDM, low Z-score, normal entropy, normal norm
        Array1::from(vec![0.85, 0.80, 2.0, 3.5, 10.0])
    }

    fn test_features_byzantine() -> Array1<f32> {
        // Low PoGQ, low TCDM, high Z-score
        Array1::from(vec![0.15, 0.20, 15.0, 1.0, 50.0])
    }

    #[test]
    fn test_threshold_classifier_honest() {
        let classifier = ThresholdClassifier::new(ThresholdConfig::default());
        let features = test_features_honest();

        let result = classifier.predict(features.view());
        assert_eq!(result, ClassifierOutput::Honest);
    }

    #[test]
    fn test_threshold_classifier_byzantine() {
        let classifier = ThresholdClassifier::new(ThresholdConfig::default());
        let features = test_features_byzantine();

        let result = classifier.predict(features.view());
        assert_eq!(result, ClassifierOutput::Byzantine);
    }

    #[test]
    fn test_decision_tree_classifier() {
        let mut classifier = DecisionTreeClassifier::default();

        // Fit (uses default tree)
        let features = Array2::zeros((1, 5));
        let labels = Array1::zeros(1);
        classifier.fit(&features, &labels);

        // Test predictions
        let honest = classifier.predict(test_features_honest().view());
        assert_eq!(honest, ClassifierOutput::Honest);

        let byzantine = classifier.predict(test_features_byzantine().view());
        assert_eq!(byzantine, ClassifierOutput::Byzantine);
    }

    #[test]
    fn test_ensemble_classifier() {
        let mut ensemble = EnsembleClassifier::default_ensemble();

        // Fit
        let features = Array2::zeros((1, 5));
        let labels = Array1::zeros(1);
        ensemble.fit(&features, &labels);

        // Test
        let honest_proba = ensemble.predict_proba(test_features_honest().view());
        let byzantine_proba = ensemble.predict_proba(test_features_byzantine().view());

        assert!(honest_proba < 0.5);
        assert!(byzantine_proba > 0.5);
    }
}
