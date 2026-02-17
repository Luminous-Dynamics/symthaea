//! ML-Enhanced Byzantine Detection System
//!
//! High-accuracy detection combining fast heuristics with ML classifiers:
//!
//! - **PoGQ (Proof of Gradient Quality)**: Cosine similarity to median gradient
//! - **TCDM (Temporal Consistency)**: Correlation with node's historical gradients
//! - **Z-Score**: Statistical outlier detection
//! - **Entropy**: Shannon entropy of gradient distribution
//!
//! ## Decision Logic
//!
//! 1. Extract composite features (PoGQ, TCDM, Z-score, Entropy)
//! 2. If PoGQ < 0.3 → Fast reject (BYZANTINE)
//! 3. If PoGQ > 0.7 → Fast accept (HONEST)
//! 4. Else → ML classifier decision (SVM/RF ensemble)
//!
//! ## Usage
//!
//! ```rust,ignore
//! use fl_aggregator::detection::{ByzantineDetector, DetectorConfig};
//!
//! let config = DetectorConfig::default();
//! let mut detector = ByzantineDetector::new(config);
//!
//! // Classify a gradient
//! let result = detector.classify(&gradient, "node_1", 0, Some(&all_gradients));
//! match result.classification {
//!     Classification::Honest => println!("Node is honest"),
//!     Classification::Byzantine => println!("Node is Byzantine!"),
//! }
//! ```

pub mod features;
pub mod classifier;
pub mod detector;

pub use features::{CompositeFeatures, FeatureExtractor};
pub use classifier::{Classifier, ClassifierConfig, ClassifierType};
pub use detector::{ByzantineDetector, DetectorConfig, Classification, DetectionResult};
