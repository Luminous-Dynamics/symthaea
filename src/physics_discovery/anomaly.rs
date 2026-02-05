//! # Anomaly Detection for Physics Data
//!
//! Uses HDC-encoded measurements and Phi analysis to detect anomalies
//! in physical data that might indicate new physics or systematic errors.

use super::encoders::{EncodedMeasurement, PhysicsEncoder, PhysicalQuantity};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Type of detected anomaly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalyType {
    /// Value outside expected range
    OutOfRange,
    /// Unexpected correlation between variables
    UnexpectedCorrelation,
    /// Break in expected trend
    TrendBreak,
    /// Value inconsistent with physical law
    PhysicsViolation,
    /// Sudden discontinuity
    Discontinuity,
    /// Periodic pattern where none expected
    UnexpectedPeriodicity,
    /// Missing expected signal
    MissingSignal,
    /// Signal where none expected
    UnexpectedSignal,
}

/// Report of a detected anomaly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyReport {
    /// Type of anomaly
    pub anomaly_type: AnomalyType,
    /// Confidence score (0-1)
    pub confidence: f64,
    /// Phi value at anomaly (integrated information)
    pub phi_value: f64,
    /// When/where the anomaly occurred
    pub location: AnomalyLocation,
    /// Description
    pub description: String,
    /// Suggested interpretations
    pub interpretations: Vec<AnomalyInterpretation>,
    /// Recommended follow-up
    pub recommendations: Vec<String>,
}

/// Location of an anomaly in data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyLocation {
    /// Index in dataset
    pub index: usize,
    /// Timestamp if applicable
    pub timestamp: Option<f64>,
    /// Which variables involved
    pub variables: Vec<String>,
    /// The anomalous values
    pub values: HashMap<String, f64>,
}

/// Possible interpretation of an anomaly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyInterpretation {
    /// Hypothesis
    pub hypothesis: String,
    /// Prior probability (based on physics knowledge)
    pub prior_probability: f64,
    /// How well it explains the anomaly
    pub explanatory_power: f64,
    /// Testable predictions
    pub predictions: Vec<String>,
}

/// Configuration for anomaly detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectorConfig {
    /// Threshold for similarity to be considered "normal"
    pub similarity_threshold: f64,
    /// Minimum Phi value to flag as interesting
    pub phi_threshold: f64,
    /// Window size for trend detection
    pub trend_window: usize,
    /// Whether to check physics consistency
    pub check_physics: bool,
    /// Known physics laws to check against
    pub physics_laws: Vec<PhysicsLaw>,
}

impl Default for AnomalyDetectorConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.7,
            phi_threshold: 0.3,
            trend_window: 10,
            check_physics: true,
            physics_laws: vec![
                PhysicsLaw::EnergyConservation,
                PhysicsLaw::Gamow,
            ],
        }
    }
}

/// Known physics laws for consistency checking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhysicsLaw {
    /// Energy must be conserved
    EnergyConservation,
    /// Momentum must be conserved
    MomentumConservation,
    /// Gamow tunneling for fusion
    Gamow,
    /// Arrhenius temperature dependence
    Arrhenius,
    /// Power law scaling
    PowerLaw,
    /// Exponential decay
    ExponentialDecay,
}

/// Anomaly detector using HDC and Phi.
pub struct AnomalyDetector {
    config: AnomalyDetectorConfig,
    /// Normal patterns learned from data
    normal_prototypes: Vec<Vec<f32>>,
    /// Encoder for new measurements
    encoder: PhysicsEncoder,
    /// History for trend detection
    history: Vec<EncodedMeasurement>,
}

impl AnomalyDetector {
    /// Create new detector.
    pub fn new(config: AnomalyDetectorConfig) -> Self {
        Self {
            config,
            normal_prototypes: Vec::new(),
            encoder: PhysicsEncoder::new(Default::default()),
            history: Vec::new(),
        }
    }

    /// Learn normal patterns from dataset.
    pub fn learn_normal(&mut self, measurements: &[EncodedMeasurement]) {
        if measurements.is_empty() {
            return;
        }

        // Create prototype for normal data
        let prototype = self.encoder.bundle(measurements);
        self.normal_prototypes.push(prototype);

        // Also learn prototypes for subgroups by quantity type
        let mut by_quantity: HashMap<PhysicalQuantity, Vec<&EncodedMeasurement>> = HashMap::new();
        for m in measurements {
            by_quantity.entry(m.quantity).or_default().push(m);
        }

        for (_quantity, group) in by_quantity {
            if group.len() >= 5 {
                let owned: Vec<_> = group.into_iter().cloned().collect();
                let proto = self.encoder.bundle(&owned);
                self.normal_prototypes.push(proto);
            }
        }
    }

    /// Check a measurement for anomalies.
    pub fn check(&mut self, measurement: &EncodedMeasurement) -> Option<AnomalyReport> {
        let mut anomalies = Vec::new();

        // Check similarity to normal prototypes
        if let Some(anomaly) = self.check_similarity(measurement) {
            anomalies.push(anomaly);
        }

        // Check for trend breaks
        if let Some(anomaly) = self.check_trend(measurement) {
            anomalies.push(anomaly);
        }

        // Check physics consistency
        if self.config.check_physics {
            if let Some(anomaly) = self.check_physics_laws(measurement) {
                anomalies.push(anomaly);
            }
        }

        // Add to history
        self.history.push(measurement.clone());
        if self.history.len() > 1000 {
            self.history.remove(0);
        }

        // Return most significant anomaly
        anomalies.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        anomalies.into_iter().next()
    }

    /// Analyze dataset for anomalies.
    pub fn analyze_dataset(&mut self, measurements: &[EncodedMeasurement]) -> Vec<AnomalyReport> {
        // First pass: learn normal patterns
        self.learn_normal(measurements);

        // Second pass: check each measurement
        let mut reports = Vec::new();
        for (i, m) in measurements.iter().enumerate() {
            if let Some(mut report) = self.check(m) {
                report.location.index = i;
                reports.push(report);
            }
        }

        reports
    }

    /// Compute Phi (integrated information) for a set of measurements.
    ///
    /// High Phi indicates the measurements form a coherent structure that
    /// cannot be decomposed into independent parts - potentially interesting
    /// physics.
    pub fn compute_phi(&self, measurements: &[EncodedMeasurement]) -> f64 {
        if measurements.len() < 2 {
            return 0.0;
        }

        // Simplified Phi computation:
        // Compare information in whole vs sum of parts
        let whole_prototype = self.encoder.bundle(measurements);
        let whole_entropy = self.entropy(&whole_prototype);

        // Partition into halves
        let mid = measurements.len() / 2;
        let part1 = self.encoder.bundle(&measurements[..mid]);
        let part2 = self.encoder.bundle(&measurements[mid..]);

        let parts_entropy = (self.entropy(&part1) + self.entropy(&part2)) / 2.0;

        // Phi ≈ entropy of whole - average entropy of parts
        // Normalized to [0, 1]
        ((whole_entropy - parts_entropy) / whole_entropy.max(0.001)).clamp(0.0, 1.0)
    }

    // Private methods

    fn check_similarity(&self, measurement: &EncodedMeasurement) -> Option<AnomalyReport> {
        if self.normal_prototypes.is_empty() {
            return None;
        }

        // Find max similarity to any normal prototype
        let max_similarity = self.normal_prototypes
            .iter()
            .map(|proto| self.cosine_similarity(&measurement.vector, proto))
            .fold(0.0f64, |a, b| a.max(b));

        if max_similarity < self.config.similarity_threshold {
            let confidence = 1.0 - max_similarity;

            Some(AnomalyReport {
                anomaly_type: AnomalyType::OutOfRange,
                confidence,
                phi_value: 0.0, // Would compute if we had context
                location: AnomalyLocation {
                    index: 0,
                    timestamp: measurement.timestamp,
                    variables: vec![format!("{:?}", measurement.quantity)],
                    values: [("value".to_string(), measurement.value)].into_iter().collect(),
                },
                description: format!(
                    "{:?} value {:.3e} is dissimilar to normal patterns (similarity: {:.2})",
                    measurement.quantity, measurement.value, max_similarity
                ),
                interpretations: vec![
                    AnomalyInterpretation {
                        hypothesis: "Measurement error".to_string(),
                        prior_probability: 0.5,
                        explanatory_power: 0.7,
                        predictions: vec!["Repeat measurement should yield different result".to_string()],
                    },
                    AnomalyInterpretation {
                        hypothesis: "New physics regime".to_string(),
                        prior_probability: 0.1,
                        explanatory_power: 0.9,
                        predictions: vec!["Should be reproducible under same conditions".to_string()],
                    },
                ],
                recommendations: vec![
                    "Verify measurement equipment".to_string(),
                    "Check for systematic errors".to_string(),
                    "Attempt to reproduce".to_string(),
                ],
            })
        } else {
            None
        }
    }

    fn check_trend(&self, measurement: &EncodedMeasurement) -> Option<AnomalyReport> {
        if self.history.len() < self.config.trend_window {
            return None;
        }

        // Get recent history of same quantity
        let recent: Vec<_> = self.history
            .iter()
            .rev()
            .take(self.config.trend_window)
            .filter(|m| m.quantity == measurement.quantity)
            .collect();

        if recent.len() < 3 {
            return None;
        }

        // Simple trend detection: check if new value breaks linear trend
        let values: Vec<f64> = recent.iter().map(|m| m.value).collect();
        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
        let std: f64 = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64).sqrt();

        if std > 0.0 {
            let z_score = (measurement.value - mean).abs() / std;
            if z_score > 3.0 {
                return Some(AnomalyReport {
                    anomaly_type: AnomalyType::TrendBreak,
                    confidence: (1.0 - 1.0 / z_score).clamp(0.0, 1.0),
                    phi_value: 0.0,
                    location: AnomalyLocation {
                        index: 0,
                        timestamp: measurement.timestamp,
                        variables: vec![format!("{:?}", measurement.quantity)],
                        values: [("value".to_string(), measurement.value)].into_iter().collect(),
                    },
                    description: format!(
                        "{:?} value {:.3e} deviates {:.1}σ from recent trend",
                        measurement.quantity, measurement.value, z_score
                    ),
                    interpretations: vec![
                        AnomalyInterpretation {
                            hypothesis: "Phase transition".to_string(),
                            prior_probability: 0.2,
                            explanatory_power: 0.8,
                            predictions: vec!["Hysteresis on return path".to_string()],
                        },
                    ],
                    recommendations: vec![
                        "Check for parameter changes".to_string(),
                        "Map transition boundary".to_string(),
                    ],
                });
            }
        }

        None
    }

    fn check_physics_laws(&self, measurement: &EncodedMeasurement) -> Option<AnomalyReport> {
        // Check against Gamow for fusion rates
        if measurement.quantity == PhysicalQuantity::ReactionRate {
            // Gamow predicts σv should be extremely small at low temperature
            // If we see high rates at low T, that's anomalous
            if let Some(temp) = self.find_recent_value(PhysicalQuantity::Temperature) {
                if temp < 1000.0 && measurement.value > 1e-30 {
                    // Room temperature with measurable fusion rate = anomaly
                    return Some(AnomalyReport {
                        anomaly_type: AnomalyType::PhysicsViolation,
                        confidence: 0.95,
                        phi_value: 0.0,
                        location: AnomalyLocation {
                            index: 0,
                            timestamp: measurement.timestamp,
                            variables: vec!["ReactionRate".to_string(), "Temperature".to_string()],
                            values: [
                                ("rate".to_string(), measurement.value),
                                ("temperature".to_string(), temp),
                            ].into_iter().collect(),
                        },
                        description: format!(
                            "Reaction rate {:.2e} at T={:.0}K exceeds Gamow prediction by many orders of magnitude",
                            measurement.value, temp
                        ),
                        interpretations: vec![
                            AnomalyInterpretation {
                                hypothesis: "Enhanced screening".to_string(),
                                prior_probability: 0.3,
                                explanatory_power: 0.5,
                                predictions: vec!["Rate should scale with screening energy".to_string()],
                            },
                            AnomalyInterpretation {
                                hypothesis: "Hot spots".to_string(),
                                prior_probability: 0.4,
                                explanatory_power: 0.7,
                                predictions: vec!["Rate should scale with trigger power".to_string()],
                            },
                            AnomalyInterpretation {
                                hypothesis: "Measurement artifact".to_string(),
                                prior_probability: 0.3,
                                explanatory_power: 0.8,
                                predictions: vec!["H control should show similar signal".to_string()],
                            },
                        ],
                        recommendations: vec![
                            "Run hydrogen control".to_string(),
                            "Verify neutron energy spectrum".to_string(),
                            "Map rate vs temperature".to_string(),
                        ],
                    });
                }
            }
        }

        None
    }

    fn find_recent_value(&self, quantity: PhysicalQuantity) -> Option<f64> {
        self.history
            .iter()
            .rev()
            .find(|m| m.quantity == quantity)
            .map(|m| m.value)
    }

    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f64 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            (dot / (norm_a * norm_b)) as f64
        } else {
            0.0
        }
    }

    fn entropy(&self, vector: &[f32]) -> f64 {
        // Simplified entropy: variance of vector elements
        let n = vector.len() as f64;
        if n == 0.0 {
            return 0.0;
        }

        let mean: f32 = vector.iter().sum::<f32>() / n as f32;
        let variance: f64 = vector.iter().map(|x| (x - mean).powi(2) as f64).sum::<f64>() / n;

        variance.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::encoders::PhysicsEncoderConfig;

    #[test]
    fn test_detector_creation() {
        let detector = AnomalyDetector::new(AnomalyDetectorConfig::default());
        assert!(detector.normal_prototypes.is_empty());
    }

    #[test]
    fn test_learn_normal() {
        let mut detector = AnomalyDetector::new(AnomalyDetectorConfig::default());
        let mut encoder = PhysicsEncoder::new(PhysicsEncoderConfig::default());

        let measurements: Vec<_> = (0..100)
            .map(|i| encoder.encode(300.0 + (i as f64) * 0.1, PhysicalQuantity::Temperature, "K"))
            .collect();

        detector.learn_normal(&measurements);
        assert!(!detector.normal_prototypes.is_empty());
    }

    #[test]
    fn test_phi_computation() {
        let detector = AnomalyDetector::new(AnomalyDetectorConfig::default());
        let mut encoder = PhysicsEncoder::new(PhysicsEncoderConfig::default());

        let measurements: Vec<_> = (0..10)
            .map(|i| encoder.encode(300.0 + i as f64, PhysicalQuantity::Temperature, "K"))
            .collect();

        let phi = detector.compute_phi(&measurements);
        assert!(phi >= 0.0 && phi <= 1.0);
    }
}
