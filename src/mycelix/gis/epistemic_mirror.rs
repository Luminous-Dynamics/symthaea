// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Epistemic Mirror (GIS v3)
//!
//! Meta-cognitive reflection system for understanding and monitoring one's own
//! epistemic state. The mirror enables introspection on knowledge quality,
//! confidence calibration, and bias detection.
//!
//! ## Design Philosophy
//!
//! "Know thyself" - The Epistemic Mirror embodies the Socratic ideal of
//! intellectual humility through continuous self-examination. It provides:
//!
//! 1. **Self-Knowledge Quantification**: What do I know, and how well?
//! 2. **Confidence Calibration**: Are my confidence levels justified?
//! 3. **Bias Detection**: What systematic errors might I be making?
//! 4. **Epistemic Health Monitoring**: Overall knowledge quality metrics
//!
//! ## Integration with GIS
//!
//! The Epistemic Mirror connects with:
//! - **Socratic Defense**: Informs when beliefs need examination
//! - **Curiosity Engine**: Prioritizes knowledge gaps to explore
//! - **Moral Uncertainty**: Tracks ethical confidence calibration
//! - **Dark Spot DHT**: Reports self-identified blind spots to network

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};

/// Epistemic Mirror - Self-reflective knowledge monitoring system
#[derive(Debug)]
pub struct EpistemicMirror {
    /// Knowledge domain assessments
    domains: HashMap<String, DomainAssessment>,

    /// Prediction history for calibration analysis
    predictions: VecDeque<PredictionRecord>,

    /// Detected biases
    detected_biases: Vec<BiasRecord>,

    /// Configuration
    config: MirrorConfig,

    /// Aggregated statistics
    stats: EpistemicStats,
}

impl EpistemicMirror {
    /// Create a new Epistemic Mirror
    pub fn new() -> Self {
        Self::with_config(MirrorConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: MirrorConfig) -> Self {
        Self {
            domains: HashMap::new(),
            predictions: VecDeque::with_capacity(config.prediction_history_size),
            detected_biases: Vec::new(),
            config,
            stats: EpistemicStats::default(),
        }
    }

    /// Register or update knowledge in a domain
    pub fn register_knowledge(
        &mut self,
        domain: &str,
        topic: &str,
        confidence: f32,
        depth: KnowledgeDepth,
    ) {
        let assessment = self
            .domains
            .entry(domain.to_string())
            .or_insert_with(|| DomainAssessment::new(domain));

        assessment.topics.insert(
            topic.to_string(),
            TopicKnowledge {
                confidence: confidence.clamp(0.0, 1.0),
                depth,
                last_updated: SystemTime::now(),
                times_used: 0,
                times_verified: 0,
                times_wrong: 0,
            },
        );

        self.stats.total_topics += 1;
    }

    /// Record a prediction for calibration tracking
    ///
    /// When you make a prediction with a confidence level, record it here.
    /// Later, when the outcome is known, call `verify_prediction`.
    pub fn record_prediction(
        &mut self,
        prediction_id: &str,
        statement: &str,
        domain: &str,
        confidence: f32,
    ) {
        let record = PredictionRecord {
            id: prediction_id.to_string(),
            statement: statement.to_string(),
            domain: domain.to_string(),
            confidence: confidence.clamp(0.0, 1.0),
            predicted_at: SystemTime::now(),
            outcome: None,
            verified_at: None,
        };

        self.predictions.push_back(record);
        self.stats.predictions_made += 1;

        // Maintain history size
        while self.predictions.len() > self.config.prediction_history_size {
            self.predictions.pop_front();
        }
    }

    /// Verify a prediction's outcome
    ///
    /// Returns the calibration error (difference between confidence and accuracy).
    pub fn verify_prediction(&mut self, prediction_id: &str, was_correct: bool) -> Option<f32> {
        let record = self
            .predictions
            .iter_mut()
            .find(|p| p.id == prediction_id)?;

        record.outcome = Some(was_correct);
        record.verified_at = Some(SystemTime::now());

        let actual = if was_correct { 1.0 } else { 0.0 };
        let error = (record.confidence - actual).abs();

        // Update domain stats
        if let Some(domain) = self.domains.get_mut(&record.domain) {
            if was_correct {
                domain.correct_predictions += 1;
            } else {
                domain.incorrect_predictions += 1;
            }
        }

        self.stats.predictions_verified += 1;

        Some(error)
    }

    /// Calculate confidence calibration across all verified predictions
    ///
    /// Returns a CalibrationReport showing how well confidence matches accuracy.
    pub fn calculate_calibration(&self) -> CalibrationReport {
        let verified: Vec<_> = self
            .predictions
            .iter()
            .filter(|p| p.outcome.is_some())
            .collect();

        if verified.is_empty() {
            return CalibrationReport::default();
        }

        // Group by confidence buckets (0-10%, 10-20%, ..., 90-100%)
        let mut buckets: [CalibrationBucket; 10] = Default::default();

        for pred in &verified {
            let bucket_idx = ((pred.confidence * 10.0) as usize).min(9);
            buckets[bucket_idx].count += 1;
            buckets[bucket_idx].confidence_sum += pred.confidence;
            if pred.outcome.unwrap_or(false) {
                buckets[bucket_idx].correct_count += 1;
            }
        }

        // Calculate metrics
        let mut total_brier_score = 0.0f32;
        let mut total_log_score = 0.0f32;

        for pred in &verified {
            let actual = if pred.outcome.unwrap_or(false) {
                1.0
            } else {
                0.0
            };

            // Brier score: (confidence - actual)^2
            total_brier_score += (pred.confidence - actual).powi(2);

            // Log score: actual * log(confidence) + (1-actual) * log(1-confidence)
            let conf_clamped = pred.confidence.clamp(0.001, 0.999);
            total_log_score +=
                actual * conf_clamped.ln() + (1.0 - actual) * (1.0 - conf_clamped).ln();
        }

        let brier_score = total_brier_score / verified.len() as f32;
        let log_score = -total_log_score / verified.len() as f32; // Negative for consistency

        // Calculate overconfidence/underconfidence
        let calibration_error = self.calculate_expected_calibration_error(&buckets);

        CalibrationReport {
            brier_score,
            log_score,
            expected_calibration_error: calibration_error,
            buckets: buckets.to_vec(),
            total_predictions: verified.len(),
            diagnosis: self.diagnose_calibration(brier_score, calibration_error),
        }
    }

    fn calculate_expected_calibration_error(&self, buckets: &[CalibrationBucket; 10]) -> f32 {
        let total: usize = buckets.iter().map(|b| b.count).sum();
        if total == 0 {
            return 0.0;
        }

        let mut ece = 0.0f32;

        for bucket in buckets {
            if bucket.count == 0 {
                continue;
            }

            let avg_confidence = bucket.confidence_sum / bucket.count as f32;
            let accuracy = bucket.correct_count as f32 / bucket.count as f32;
            let weight = bucket.count as f32 / total as f32;

            ece += weight * (avg_confidence - accuracy).abs();
        }

        ece
    }

    fn diagnose_calibration(&self, brier: f32, ece: f32) -> CalibrationDiagnosis {
        if brier < 0.1 && ece < 0.05 {
            CalibrationDiagnosis::WellCalibrated
        } else if ece > 0.15 {
            // Check direction
            let overconfident_buckets = self
                .predictions
                .iter()
                .filter(|p| p.outcome == Some(false) && p.confidence > 0.7)
                .count();
            let underconfident_buckets = self
                .predictions
                .iter()
                .filter(|p| p.outcome == Some(true) && p.confidence < 0.3)
                .count();

            if overconfident_buckets > underconfident_buckets {
                CalibrationDiagnosis::Overconfident {
                    severity: (ece - 0.1).max(0.0),
                    recommendation: "Consider lowering confidence on uncertain claims".to_string(),
                }
            } else if underconfident_buckets > overconfident_buckets {
                CalibrationDiagnosis::Underconfident {
                    severity: (ece - 0.1).max(0.0),
                    recommendation: "Your predictions are better than your confidence suggests"
                        .to_string(),
                }
            } else {
                CalibrationDiagnosis::Miscalibrated {
                    details: format!("ECE: {ece:.3}, Brier: {brier:.3}"),
                }
            }
        } else {
            CalibrationDiagnosis::Adequate {
                improvement_areas: vec!["Continue tracking predictions for more data".to_string()],
            }
        }
    }

    /// Perform self-examination to identify potential biases
    pub fn detect_biases(&mut self) -> Vec<BiasAlert> {
        let mut alerts = Vec::new();

        // Bias 1: Confirmation bias (ignoring disconfirming evidence)
        if let Some(alert) = self.detect_confirmation_bias() {
            alerts.push(alert);
        }

        // Bias 2: Recency bias (overweighting recent information)
        if let Some(alert) = self.detect_recency_bias() {
            alerts.push(alert);
        }

        // Bias 3: Domain overconfidence (high confidence in weak domains)
        if let Some(alert) = self.detect_domain_overconfidence() {
            alerts.push(alert);
        }

        // Bias 4: Anchoring (not updating enough on new evidence)
        if let Some(alert) = self.detect_anchoring_bias() {
            alerts.push(alert);
        }

        // Record detected biases
        for alert in &alerts {
            self.detected_biases.push(BiasRecord {
                bias_type: alert.bias_type.clone(),
                severity: alert.severity,
                detected_at: SystemTime::now(),
                domain: alert.domain.clone(),
            });
        }

        alerts
    }

    fn detect_confirmation_bias(&self) -> Option<BiasAlert> {
        // Look for pattern: high confidence maintained despite incorrect predictions
        let recent_wrong: Vec<_> = self
            .predictions
            .iter()
            .filter(|p| p.outcome == Some(false))
            .filter(|p| p.confidence > 0.7)
            .collect();

        if recent_wrong.len() >= 3 {
            let domains: Vec<_> = recent_wrong.iter().map(|p| p.domain.clone()).collect();

            return Some(BiasAlert {
                bias_type: BiasType::Confirmation,
                severity: (recent_wrong.len() as f32 / 10.0).min(1.0),
                description: format!(
                    "High-confidence predictions ({} cases) were incorrect. \
                    May be ignoring disconfirming evidence.",
                    recent_wrong.len()
                ),
                domain: Some(domains.first().cloned().unwrap_or_default()),
                mitigation: vec![
                    "Actively seek disconfirming evidence".to_string(),
                    "Apply steelman principle to opposing views".to_string(),
                    "Pre-commit to updating beliefs on specific evidence".to_string(),
                ],
            });
        }

        None
    }

    fn detect_recency_bias(&self) -> Option<BiasAlert> {
        // Check if recent predictions are weighted much higher than older ones
        let cutoff = SystemTime::now() - Duration::from_secs(7 * 24 * 3600); // 1 week

        let recent: Vec<_> = self
            .predictions
            .iter()
            .filter(|p| p.predicted_at > cutoff)
            .collect();

        let older: Vec<_> = self
            .predictions
            .iter()
            .filter(|p| p.predicted_at <= cutoff)
            .collect();

        if recent.len() < 5 || older.len() < 5 {
            return None;
        }

        let recent_avg_conf: f32 =
            recent.iter().map(|p| p.confidence).sum::<f32>() / recent.len() as f32;
        let older_avg_conf: f32 =
            older.iter().map(|p| p.confidence).sum::<f32>() / older.len() as f32;

        // If recent predictions have significantly higher confidence
        if recent_avg_conf > older_avg_conf + 0.15 {
            return Some(BiasAlert {
                bias_type: BiasType::Recency,
                severity: (recent_avg_conf - older_avg_conf - 0.1) * 2.0,
                description: format!(
                    "Recent predictions have {:.0}% higher confidence than older ones. \
                    May be overweighting recent information.",
                    (recent_avg_conf - older_avg_conf) * 100.0
                ),
                domain: None,
                mitigation: vec![
                    "Consider long-term base rates".to_string(),
                    "Weight historical accuracy appropriately".to_string(),
                ],
            });
        }

        None
    }

    fn detect_domain_overconfidence(&self) -> Option<BiasAlert> {
        for (domain_name, domain) in &self.domains {
            let total = domain.correct_predictions + domain.incorrect_predictions;
            if total < 5 {
                continue;
            }

            let accuracy = domain.correct_predictions as f32 / total as f32;

            // Get average confidence in this domain
            let domain_preds: Vec<_> = self
                .predictions
                .iter()
                .filter(|p| &p.domain == domain_name)
                .collect();

            if domain_preds.is_empty() {
                continue;
            }

            let avg_confidence: f32 =
                domain_preds.iter().map(|p| p.confidence).sum::<f32>() / domain_preds.len() as f32;

            // Significant overconfidence: confidence much higher than accuracy
            if avg_confidence > accuracy + 0.2 {
                return Some(BiasAlert {
                    bias_type: BiasType::DomainOverconfidence,
                    severity: (avg_confidence - accuracy - 0.1) * 2.0,
                    description: format!(
                        "In domain '{}': confidence {:.0}% but accuracy only {:.0}%.",
                        domain_name,
                        avg_confidence * 100.0,
                        accuracy * 100.0
                    ),
                    domain: Some(domain_name.clone()),
                    mitigation: vec![
                        format!(
                            "Lower confidence in '{}' claims by {:.0}%",
                            domain_name,
                            (avg_confidence - accuracy) * 100.0
                        ),
                        "Seek external verification before high-confidence claims".to_string(),
                    ],
                });
            }
        }

        None
    }

    fn detect_anchoring_bias(&self) -> Option<BiasAlert> {
        // Look for pattern: confidence doesn't change much despite new evidence
        // This is simplified - would need revision history for full implementation
        let verified_wrong: Vec<_> = self
            .predictions
            .iter()
            .filter(|p| p.outcome == Some(false))
            .collect();

        if verified_wrong.len() < 3 {
            return None;
        }

        // Check if there are subsequent predictions on same domain with unchanged confidence
        let mut same_domain_unchanged = 0;

        for wrong in &verified_wrong {
            let later_same_domain: Vec<_> = self
                .predictions
                .iter()
                .filter(|p| p.domain == wrong.domain)
                .filter(|p| p.predicted_at > wrong.predicted_at)
                .collect();

            for later in later_same_domain {
                if (later.confidence - wrong.confidence).abs() < 0.1 {
                    same_domain_unchanged += 1;
                }
            }
        }

        if same_domain_unchanged >= 3 {
            return Some(BiasAlert {
                bias_type: BiasType::Anchoring,
                severity: (same_domain_unchanged as f32 / 10.0).min(1.0),
                description: format!(
                    "Confidence levels not adjusting after {} incorrect predictions. \
                    May be anchored to initial estimates.",
                    verified_wrong.len()
                ),
                domain: None,
                mitigation: vec![
                    "Explicitly recalculate confidence after each disconfirmation".to_string(),
                    "Use Bayesian updating formula".to_string(),
                ],
            });
        }

        None
    }

    /// Generate a comprehensive epistemic health report
    pub fn health_report(&self) -> EpistemicHealthReport {
        let calibration = self.calculate_calibration();

        let domain_health: Vec<_> = self
            .domains
            .iter()
            .map(|(name, domain)| {
                let total = domain.correct_predictions + domain.incorrect_predictions;
                let accuracy = if total > 0 {
                    domain.correct_predictions as f32 / total as f32
                } else {
                    0.5 // Unknown - assume baseline
                };

                DomainHealth {
                    domain: name.clone(),
                    topic_count: domain.topics.len(),
                    accuracy,
                    coverage: self.estimate_domain_coverage(domain),
                    staleness: self.calculate_domain_staleness(domain),
                }
            })
            .collect();

        let overall_health = self.calculate_overall_health(&calibration, &domain_health);

        EpistemicHealthReport {
            overall_health,
            calibration,
            domain_health,
            active_biases: self
                .detected_biases
                .iter()
                .filter(|b| {
                    SystemTime::now()
                        .duration_since(b.detected_at)
                        .map(|d| d < Duration::from_secs(7 * 24 * 3600))
                        .unwrap_or(false)
                })
                .count(),
            recommendations: self.generate_recommendations(&overall_health),
        }
    }

    fn estimate_domain_coverage(&self, domain: &DomainAssessment) -> f32 {
        // Estimate based on topic diversity and depth
        let topic_count = domain.topics.len() as f32;
        let avg_depth = domain
            .topics
            .values()
            .map(|t| match t.depth {
                KnowledgeDepth::Surface => 0.3,
                KnowledgeDepth::Working => 0.5,
                KnowledgeDepth::Deep => 0.8,
                KnowledgeDepth::Expert => 1.0,
            })
            .sum::<f32>()
            / topic_count.max(1.0);

        (topic_count / 20.0).min(1.0) * avg_depth
    }

    fn calculate_domain_staleness(&self, domain: &DomainAssessment) -> f32 {
        let now = SystemTime::now();
        let one_year = Duration::from_secs(365 * 24 * 3600);

        let total_staleness: f32 = domain
            .topics
            .values()
            .map(|t| {
                now.duration_since(t.last_updated)
                    .map(|d| (d.as_secs_f32() / one_year.as_secs_f32()).min(1.0))
                    .unwrap_or(1.0)
            })
            .sum();

        total_staleness / domain.topics.len().max(1) as f32
    }

    fn calculate_overall_health(
        &self,
        calibration: &CalibrationReport,
        domains: &[DomainHealth],
    ) -> f32 {
        // Combine metrics into overall health score (0-1)
        let calibration_score = 1.0 - calibration.expected_calibration_error.min(0.5) * 2.0;

        let avg_domain_health = if domains.is_empty() {
            0.5
        } else {
            domains
                .iter()
                .map(|d| d.accuracy * (1.0 - d.staleness) * d.coverage.sqrt())
                .sum::<f32>()
                / domains.len() as f32
        };

        let bias_penalty = (self.detected_biases.len() as f32 * 0.05).min(0.3);

        ((calibration_score + avg_domain_health) / 2.0 - bias_penalty).max(0.0)
    }

    fn generate_recommendations(&self, health: &f32) -> Vec<String> {
        let mut recommendations = Vec::new();

        if *health < 0.5 {
            recommendations
                .push("Critical: Epistemic health is low. Focus on calibration.".to_string());
        }

        if self.stats.predictions_verified < 10 {
            recommendations
                .push("Record more predictions to improve calibration tracking.".to_string());
        }

        if !self.detected_biases.is_empty() {
            recommendations.push(format!(
                "Address {} detected bias(es) before making important decisions.",
                self.detected_biases.len()
            ));
        }

        let stale_domains: Vec<_> = self
            .domains
            .iter()
            .filter(|(_, d)| self.calculate_domain_staleness(d) > 0.5)
            .map(|(n, _)| n.clone())
            .collect();

        if !stale_domains.is_empty() {
            recommendations.push(format!(
                "Update knowledge in stale domains: {}",
                stale_domains.join(", ")
            ));
        }

        if recommendations.is_empty() {
            recommendations.push("Epistemic health is good. Continue monitoring.".to_string());
        }

        recommendations
    }

    /// Ask reflective questions about own knowledge
    pub fn self_reflect(&self, domain: &str) -> ReflectionQuestions {
        let domain_data = self.domains.get(domain);

        let mut questions = vec![
            format!("What are my key assumptions about {}?", domain),
            format!("What would change my mind about {}?", domain),
            "What evidence would disprove my current beliefs?".to_string(),
            "Am I confusing familiarity with understanding?".to_string(),
        ];

        if let Some(d) = domain_data {
            if d.correct_predictions + d.incorrect_predictions < 5 {
                questions.push("I have limited verified predictions in this domain. Should I lower confidence?".to_string());
            }

            let accuracy = if d.correct_predictions + d.incorrect_predictions > 0 {
                d.correct_predictions as f32
                    / (d.correct_predictions + d.incorrect_predictions) as f32
            } else {
                0.5
            };

            if accuracy < 0.7 {
                questions.push(format!(
                    "My accuracy in {} is {:.0}%. What am I systematically getting wrong?",
                    domain,
                    accuracy * 100.0
                ));
            }
        }

        ReflectionQuestions {
            domain: domain.to_string(),
            questions,
            suggested_actions: vec![
                "Seek expert review of assumptions".to_string(),
                "Find and analyze past errors".to_string(),
                "Request external calibration feedback".to_string(),
            ],
        }
    }

    /// Get statistics
    pub fn statistics(&self) -> &EpistemicStats {
        &self.stats
    }
}

impl Default for EpistemicMirror {
    fn default() -> Self {
        Self::new()
    }
}

// --- Types ---

/// Assessment of knowledge in a specific domain
#[derive(Debug, Clone)]
pub struct DomainAssessment {
    pub name: String,
    pub topics: HashMap<String, TopicKnowledge>,
    pub correct_predictions: usize,
    pub incorrect_predictions: usize,
    pub created_at: SystemTime,
}

impl DomainAssessment {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            topics: HashMap::new(),
            correct_predictions: 0,
            incorrect_predictions: 0,
            created_at: SystemTime::now(),
        }
    }
}

/// Knowledge about a specific topic
#[derive(Debug, Clone)]
pub struct TopicKnowledge {
    pub confidence: f32,
    pub depth: KnowledgeDepth,
    pub last_updated: SystemTime,
    pub times_used: usize,
    pub times_verified: usize,
    pub times_wrong: usize,
}

/// Depth levels of knowledge
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KnowledgeDepth {
    /// Can recognize and recall basic facts
    Surface,
    /// Can apply knowledge to common situations
    Working,
    /// Can analyze, synthesize, and explain edge cases
    Deep,
    /// Can innovate and identify novel patterns
    Expert,
}

/// Record of a prediction for calibration
#[derive(Debug, Clone)]
#[allow(dead_code)] // RESERVED(future): calibration tracking structure
struct PredictionRecord {
    id: String,
    statement: String,
    domain: String,
    confidence: f32,
    predicted_at: SystemTime,
    outcome: Option<bool>,
    verified_at: Option<SystemTime>,
}

/// Calibration report
#[derive(Debug, Clone, Default)]
pub struct CalibrationReport {
    /// Brier score (lower is better, 0 is perfect)
    pub brier_score: f32,
    /// Log score (lower is better)
    pub log_score: f32,
    /// Expected Calibration Error
    pub expected_calibration_error: f32,
    /// Confidence buckets
    pub buckets: Vec<CalibrationBucket>,
    /// Total predictions analyzed
    pub total_predictions: usize,
    /// Diagnosis
    pub diagnosis: CalibrationDiagnosis,
}

/// Calibration bucket for a confidence range
#[derive(Debug, Clone, Default)]
pub struct CalibrationBucket {
    pub count: usize,
    pub correct_count: usize,
    pub confidence_sum: f32,
}

/// Calibration diagnosis
#[derive(Debug, Clone, Default)]
pub enum CalibrationDiagnosis {
    #[default]
    InsufficientData,
    WellCalibrated,
    Overconfident {
        severity: f32,
        recommendation: String,
    },
    Underconfident {
        severity: f32,
        recommendation: String,
    },
    Miscalibrated {
        details: String,
    },
    Adequate {
        improvement_areas: Vec<String>,
    },
}

/// Alert about a detected bias
#[derive(Debug, Clone)]
pub struct BiasAlert {
    pub bias_type: BiasType,
    pub severity: f32,
    pub description: String,
    pub domain: Option<String>,
    pub mitigation: Vec<String>,
}

/// Types of cognitive biases
#[derive(Debug, Clone, PartialEq)]
pub enum BiasType {
    /// Ignoring disconfirming evidence
    Confirmation,
    /// Overweighting recent information
    Recency,
    /// Too confident in weak knowledge areas
    DomainOverconfidence,
    /// Not updating on new evidence
    Anchoring,
    /// Overconfidence after success
    HotHand,
    /// Assuming others share our knowledge
    CurseOfKnowledge,
}

/// Record of a detected bias
#[derive(Debug, Clone)]
#[allow(dead_code)] // RESERVED(future): bias detection record
struct BiasRecord {
    bias_type: BiasType,
    severity: f32,
    detected_at: SystemTime,
    domain: Option<String>,
}

/// Domain health assessment
#[derive(Debug, Clone)]
pub struct DomainHealth {
    pub domain: String,
    pub topic_count: usize,
    pub accuracy: f32,
    pub coverage: f32,
    pub staleness: f32,
}

/// Comprehensive epistemic health report
#[derive(Debug, Clone)]
pub struct EpistemicHealthReport {
    pub overall_health: f32,
    pub calibration: CalibrationReport,
    pub domain_health: Vec<DomainHealth>,
    pub active_biases: usize,
    pub recommendations: Vec<String>,
}

/// Reflective questions for self-examination
#[derive(Debug, Clone)]
pub struct ReflectionQuestions {
    pub domain: String,
    pub questions: Vec<String>,
    pub suggested_actions: Vec<String>,
}

/// Configuration for Epistemic Mirror
#[derive(Debug, Clone)]
pub struct MirrorConfig {
    pub prediction_history_size: usize,
    pub calibration_threshold: f32,
    pub bias_detection_sensitivity: f32,
}

impl Default for MirrorConfig {
    fn default() -> Self {
        Self {
            prediction_history_size: 1000,
            calibration_threshold: 0.1,
            bias_detection_sensitivity: 0.5,
        }
    }
}

/// Statistics for Epistemic Mirror
#[derive(Debug, Default, Clone)]
pub struct EpistemicStats {
    pub total_topics: usize,
    pub predictions_made: usize,
    pub predictions_verified: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knowledge_registration() {
        let mut mirror = EpistemicMirror::new();

        mirror.register_knowledge("physics", "thermodynamics", 0.8, KnowledgeDepth::Working);

        assert!(mirror.domains.contains_key("physics"));
        assert_eq!(mirror.stats.total_topics, 1);
    }

    #[test]
    fn test_prediction_recording() {
        let mut mirror = EpistemicMirror::new();

        mirror.record_prediction("pred1", "X will happen", "science", 0.7);
        assert_eq!(mirror.stats.predictions_made, 1);

        let error = mirror.verify_prediction("pred1", true);
        assert!(error.is_some());
        assert_eq!(mirror.stats.predictions_verified, 1);
    }

    #[test]
    fn test_calibration_calculation() {
        let mut mirror = EpistemicMirror::new();

        // Record several predictions with known outcomes
        for i in 0..10 {
            let id = format!("pred{}", i);
            let confidence = if i % 2 == 0 { 0.8 } else { 0.3 };
            let correct = i % 2 == 0; // High confidence ones are correct

            mirror.record_prediction(&id, "test", "test_domain", confidence);
            mirror.verify_prediction(&id, correct);
        }

        let report = mirror.calculate_calibration();
        assert!(report.total_predictions == 10);
        // Well-calibrated case should have low ECE
        assert!(report.expected_calibration_error < 0.3);
    }

    #[test]
    fn test_overconfidence_detection() {
        let mut mirror = EpistemicMirror::new();

        // Register domain
        mirror.register_knowledge("test", "topic1", 0.9, KnowledgeDepth::Surface);

        // Make overconfident predictions that turn out wrong
        for i in 0..5 {
            let id = format!("pred{}", i);
            mirror.record_prediction(&id, "confident claim", "test", 0.9);
            mirror.verify_prediction(&id, false); // All wrong!
        }

        let alerts = mirror.detect_biases();
        assert!(!alerts.is_empty());
    }

    #[test]
    fn test_health_report() {
        let mut mirror = EpistemicMirror::new();

        mirror.register_knowledge("math", "algebra", 0.9, KnowledgeDepth::Deep);
        mirror.record_prediction("p1", "2+2=4", "math", 0.99);
        mirror.verify_prediction("p1", true);

        let report = mirror.health_report();
        assert!(report.overall_health > 0.0);
        assert!(!report.recommendations.is_empty());
    }

    #[test]
    fn test_self_reflection() {
        let mut mirror = EpistemicMirror::new();
        mirror.register_knowledge("ai", "neural_nets", 0.7, KnowledgeDepth::Working);

        let reflection = mirror.self_reflect("ai");
        assert!(!reflection.questions.is_empty());
        assert_eq!(reflection.domain, "ai");
    }
}
