//! Byzantine Detector - High-Level API.
//!
//! Unified interface combining fast PoGQ heuristics with ML classification.

use crate::detection::{
    classifier::{create_classifier, Classifier, ClassifierConfig, ClassifierOutput},
    features::{extract_features_batch, CompositeFeatures, FeatureExtractor},
};
use crate::Gradient;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

/// Classification result.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Classification {
    /// Node is classified as honest.
    Honest,
    /// Node is classified as Byzantine (malicious).
    Byzantine,
}

impl From<ClassifierOutput> for Classification {
    fn from(output: ClassifierOutput) -> Self {
        match output {
            ClassifierOutput::Honest => Classification::Honest,
            ClassifierOutput::Byzantine => Classification::Byzantine,
        }
    }
}

/// Detection result with confidence.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DetectionResult {
    /// Classification decision.
    pub classification: Classification,

    /// Confidence in the decision [0, 1].
    pub confidence: f32,

    /// Which path was used for decision.
    pub decision_path: DecisionPath,

    /// Extracted features (optional, for debugging).
    pub features: Option<CompositeFeatures>,
}

/// Path taken for decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecisionPath {
    /// Fast rejection based on low PoGQ.
    FastRejectPoGQ,
    /// Fast acceptance based on high PoGQ.
    FastAcceptPoGQ,
    /// ML classifier used for borderline case.
    MLClassifier,
    /// Fallback (classifier not trained).
    Fallback,
}

/// Configuration for the Byzantine detector.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DetectorConfig {
    /// PoGQ below this → instant Byzantine classification.
    pub pogq_low_threshold: f32,

    /// PoGQ above this → instant Honest classification.
    pub pogq_high_threshold: f32,

    /// ML probability threshold for Byzantine classification.
    pub ml_threshold: f32,

    /// Number of rounds to retain for TCDM.
    pub history_window: usize,

    /// Classifier configuration.
    pub classifier_config: ClassifierConfig,

    /// Whether to include features in results (for debugging).
    pub include_features_in_result: bool,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            pogq_low_threshold: 0.3,
            pogq_high_threshold: 0.7,
            ml_threshold: 0.5,
            history_window: 5,
            classifier_config: ClassifierConfig::default(),
            include_features_in_result: false,
        }
    }
}

impl DetectorConfig {
    /// Create config optimized for high precision (minimize false positives).
    pub fn high_precision() -> Self {
        Self {
            pogq_low_threshold: 0.2,
            pogq_high_threshold: 0.8,
            ml_threshold: 0.7,
            ..Default::default()
        }
    }

    /// Create config optimized for high recall (minimize false negatives).
    pub fn high_recall() -> Self {
        Self {
            pogq_low_threshold: 0.4,
            pogq_high_threshold: 0.6,
            ml_threshold: 0.3,
            ..Default::default()
        }
    }
}

/// Statistics tracking for detector.
#[derive(Debug, Default)]
pub struct DetectorStats {
    /// Total classifications performed.
    pub total_classifications: AtomicU64,
    /// Fast rejections (PoGQ < threshold).
    pub pogq_fast_reject: AtomicU64,
    /// Fast acceptances (PoGQ > threshold).
    pub pogq_fast_accept: AtomicU64,
    /// ML classifier invocations.
    pub ml_classifications: AtomicU64,
    /// Fallback decisions.
    pub fallback_decisions: AtomicU64,
}

impl DetectorStats {
    /// Create new stats tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get snapshot of current stats.
    pub fn snapshot(&self) -> DetectorStatsSnapshot {
        let total = self.total_classifications.load(Ordering::Relaxed);
        DetectorStatsSnapshot {
            total_classifications: total,
            pogq_fast_reject: self.pogq_fast_reject.load(Ordering::Relaxed),
            pogq_fast_accept: self.pogq_fast_accept.load(Ordering::Relaxed),
            ml_classifications: self.ml_classifications.load(Ordering::Relaxed),
            fallback_decisions: self.fallback_decisions.load(Ordering::Relaxed),
        }
    }

    /// Reset all counters.
    pub fn reset(&self) {
        self.total_classifications.store(0, Ordering::Relaxed);
        self.pogq_fast_reject.store(0, Ordering::Relaxed);
        self.pogq_fast_accept.store(0, Ordering::Relaxed);
        self.ml_classifications.store(0, Ordering::Relaxed);
        self.fallback_decisions.store(0, Ordering::Relaxed);
    }
}

/// Snapshot of detector stats.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DetectorStatsSnapshot {
    pub total_classifications: u64,
    pub pogq_fast_reject: u64,
    pub pogq_fast_accept: u64,
    pub ml_classifications: u64,
    pub fallback_decisions: u64,
}

impl DetectorStatsSnapshot {
    /// Get percentage of fast path decisions.
    pub fn fast_path_percentage(&self) -> f64 {
        if self.total_classifications == 0 {
            return 0.0;
        }
        let fast = self.pogq_fast_reject + self.pogq_fast_accept;
        fast as f64 / self.total_classifications as f64 * 100.0
    }
}

/// High-level Byzantine detection system.
pub struct ByzantineDetector {
    config: DetectorConfig,
    feature_extractor: FeatureExtractor,
    classifier: Box<dyn Classifier>,
    is_trained: bool,
    stats: DetectorStats,
}

impl ByzantineDetector {
    /// Create a new Byzantine detector.
    pub fn new(config: DetectorConfig) -> Self {
        let feature_extractor = FeatureExtractor::new(config.history_window);
        let classifier = create_classifier(&config.classifier_config);

        Self {
            config,
            feature_extractor,
            classifier,
            is_trained: false,
            stats: DetectorStats::new(),
        }
    }

    /// Train the detector on labeled data.
    ///
    /// # Arguments
    /// * `features` - Feature matrix (n_samples, n_features)
    /// * `labels` - Binary labels (0=honest, 1=Byzantine)
    pub fn train(&mut self, features: &Array2<f32>, labels: &Array1<u8>) {
        self.classifier.fit(features, labels);
        self.is_trained = true;
    }

    /// Train from raw gradients.
    ///
    /// # Arguments
    /// * `gradients` - List of gradient vectors
    /// * `node_ids` - Corresponding node IDs
    /// * `labels` - Binary labels (0=honest, 1=Byzantine)
    pub fn train_from_gradients(
        &mut self,
        gradients: &[Gradient],
        node_ids: &[String],
        labels: &[u8],
    ) {
        // Extract features
        let features_list = extract_features_batch(gradients, node_ids, 0, &mut self.feature_extractor);

        // Convert to matrix
        let n_samples = features_list.len();
        let n_features = CompositeFeatures::NUM_FEATURES;
        let mut feature_matrix = Array2::zeros((n_samples, n_features));

        for (i, feat) in features_list.iter().enumerate() {
            let arr = feat.to_array();
            for j in 0..n_features {
                feature_matrix[[i, j]] = arr[j];
            }
        }

        let label_array = Array1::from(labels.to_vec());
        self.train(&feature_matrix, &label_array);
    }

    /// Classify a single node's gradient.
    ///
    /// # Arguments
    /// * `gradient` - Gradient vector submitted by node
    /// * `node_id` - ID of the node
    /// * `round_num` - Current training round
    /// * `all_gradients` - All gradients in round (for global stats)
    pub fn classify(
        &mut self,
        gradient: &Gradient,
        node_id: &str,
        round_num: u64,
        all_gradients: Option<&[Gradient]>,
    ) -> DetectionResult {
        // Extract features
        let features = self.feature_extractor.extract_features(
            gradient,
            node_id,
            round_num,
            all_gradients,
        );

        self.stats
            .total_classifications
            .fetch_add(1, Ordering::Relaxed);

        // Fast path 1: PoGQ too low → Byzantine
        if features.pogq_score < self.config.pogq_low_threshold {
            self.stats.pogq_fast_reject.fetch_add(1, Ordering::Relaxed);
            let confidence = 1.0 - features.pogq_score / self.config.pogq_low_threshold;
            return DetectionResult {
                classification: Classification::Byzantine,
                confidence,
                decision_path: DecisionPath::FastRejectPoGQ,
                features: self.maybe_include_features(features),
            };
        }

        // Fast path 2: PoGQ high enough → Honest
        if features.pogq_score > self.config.pogq_high_threshold {
            self.stats.pogq_fast_accept.fetch_add(1, Ordering::Relaxed);
            let confidence =
                (features.pogq_score - self.config.pogq_high_threshold) / (1.0 - self.config.pogq_high_threshold);
            return DetectionResult {
                classification: Classification::Honest,
                confidence,
                decision_path: DecisionPath::FastAcceptPoGQ,
                features: self.maybe_include_features(features),
            };
        }

        // Slow path: ML classifier for borderline cases
        if !self.is_trained {
            self.stats.fallback_decisions.fetch_add(1, Ordering::Relaxed);
            // Fallback: use PoGQ threshold at 0.5
            let classification = if features.pogq_score < 0.5 {
                Classification::Byzantine
            } else {
                Classification::Honest
            };
            return DetectionResult {
                classification,
                confidence: 0.5,
                decision_path: DecisionPath::Fallback,
                features: self.maybe_include_features(features),
            };
        }

        self.stats.ml_classifications.fetch_add(1, Ordering::Relaxed);

        // Run ML classifier
        let feature_vector = features.to_array();
        let proba = self.classifier.predict_proba(feature_vector.view());

        let (classification, confidence) = if proba >= self.config.ml_threshold {
            (Classification::Byzantine, proba)
        } else {
            (Classification::Honest, 1.0 - proba)
        };

        DetectionResult {
            classification,
            confidence,
            decision_path: DecisionPath::MLClassifier,
            features: self.maybe_include_features(features),
        }
    }

    /// Classify multiple nodes efficiently.
    pub fn classify_batch(
        &mut self,
        gradients: &[Gradient],
        node_ids: &[String],
        round_num: u64,
    ) -> Vec<DetectionResult> {
        // Extract features for all gradients
        let features_list =
            extract_features_batch(gradients, node_ids, round_num, &mut self.feature_extractor);

        features_list
            .into_iter()
            .map(|features| {
                self.stats
                    .total_classifications
                    .fetch_add(1, Ordering::Relaxed);

                // Fast paths
                if features.pogq_score < self.config.pogq_low_threshold {
                    self.stats.pogq_fast_reject.fetch_add(1, Ordering::Relaxed);
                    let confidence = 1.0 - features.pogq_score / self.config.pogq_low_threshold;
                    return DetectionResult {
                        classification: Classification::Byzantine,
                        confidence,
                        decision_path: DecisionPath::FastRejectPoGQ,
                        features: self.maybe_include_features(features),
                    };
                }

                if features.pogq_score > self.config.pogq_high_threshold {
                    self.stats.pogq_fast_accept.fetch_add(1, Ordering::Relaxed);
                    let confidence = (features.pogq_score - self.config.pogq_high_threshold)
                        / (1.0 - self.config.pogq_high_threshold);
                    return DetectionResult {
                        classification: Classification::Honest,
                        confidence,
                        decision_path: DecisionPath::FastAcceptPoGQ,
                        features: self.maybe_include_features(features),
                    };
                }

                // ML classifier
                if !self.is_trained {
                    self.stats.fallback_decisions.fetch_add(1, Ordering::Relaxed);
                    let classification = if features.pogq_score < 0.5 {
                        Classification::Byzantine
                    } else {
                        Classification::Honest
                    };
                    return DetectionResult {
                        classification,
                        confidence: 0.5,
                        decision_path: DecisionPath::Fallback,
                        features: self.maybe_include_features(features),
                    };
                }

                self.stats.ml_classifications.fetch_add(1, Ordering::Relaxed);

                let feature_vector = features.to_array();
                let proba = self.classifier.predict_proba(feature_vector.view());

                let (classification, confidence) = if proba >= self.config.ml_threshold {
                    (Classification::Byzantine, proba)
                } else {
                    (Classification::Honest, 1.0 - proba)
                };

                DetectionResult {
                    classification,
                    confidence,
                    decision_path: DecisionPath::MLClassifier,
                    features: self.maybe_include_features(features),
                }
            })
            .collect()
    }

    /// Get detector statistics.
    pub fn stats(&self) -> DetectorStatsSnapshot {
        self.stats.snapshot()
    }

    /// Reset detector statistics.
    pub fn reset_stats(&self) {
        self.stats.reset();
    }

    /// Reset all state including feature history.
    pub fn reset(&mut self) {
        self.feature_extractor.reset();
        self.stats.reset();
    }

    /// Check if detector is trained.
    pub fn is_trained(&self) -> bool {
        self.is_trained
    }

    /// Get current configuration.
    pub fn config(&self) -> &DetectorConfig {
        &self.config
    }

    fn maybe_include_features(&self, features: CompositeFeatures) -> Option<CompositeFeatures> {
        if self.config.include_features_in_result {
            Some(features)
        } else {
            None
        }
    }
}

impl Default for ByzantineDetector {
    fn default() -> Self {
        Self::new(DetectorConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_fast_reject() {
        let mut detector = ByzantineDetector::default();

        // Create gradients where target is very different from others
        let honest_gradients = vec![
            array![1.0, 2.0, 3.0],
            array![1.1, 2.1, 3.1],
            array![1.2, 2.2, 3.2],
            array![0.9, 1.9, 2.9],
        ];

        // Byzantine gradient - opposite direction
        let byzantine_gradient = array![-10.0, -20.0, -30.0];

        let result = detector.classify(
            &byzantine_gradient,
            "byzantine_node",
            0,
            Some(&honest_gradients),
        );

        assert_eq!(result.classification, Classification::Byzantine);
        assert_eq!(result.decision_path, DecisionPath::FastRejectPoGQ);
    }

    #[test]
    fn test_fast_accept() {
        let mut detector = ByzantineDetector::default();

        let gradients = vec![
            array![1.0, 2.0, 3.0],
            array![1.0, 2.0, 3.0],
            array![1.0, 2.0, 3.0],
        ];

        // Identical to median
        let honest_gradient = array![1.0, 2.0, 3.0];

        let result = detector.classify(&honest_gradient, "honest_node", 0, Some(&gradients));

        assert_eq!(result.classification, Classification::Honest);
        assert_eq!(result.decision_path, DecisionPath::FastAcceptPoGQ);
    }

    #[test]
    fn test_batch_classification() {
        let mut detector = ByzantineDetector::default();

        let gradients = vec![
            array![1.0, 2.0, 3.0],       // Honest
            array![-10.0, -20.0, -30.0], // Byzantine
            array![1.1, 2.1, 3.1],       // Honest
        ];
        let node_ids = vec![
            "node_1".to_string(),
            "node_2".to_string(),
            "node_3".to_string(),
        ];

        let results = detector.classify_batch(&gradients, &node_ids, 0);

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].classification, Classification::Honest);
        assert_eq!(results[1].classification, Classification::Byzantine);
        assert_eq!(results[2].classification, Classification::Honest);
    }

    #[test]
    fn test_statistics() {
        let mut detector = ByzantineDetector::default();

        let gradients = vec![array![1.0, 2.0, 3.0]];

        // Classify several times
        detector.classify(&array![1.0, 2.0, 3.0], "node_1", 0, Some(&gradients));
        detector.classify(&array![-10.0, -20.0, -30.0], "node_2", 0, Some(&gradients));

        let stats = detector.stats();
        assert_eq!(stats.total_classifications, 2);
        assert!(stats.pogq_fast_reject > 0 || stats.pogq_fast_accept > 0);
    }
}
