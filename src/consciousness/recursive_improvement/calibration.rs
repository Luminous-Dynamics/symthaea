// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Calibration Tracking System (MAGI Loop Step 2)
//!
//! This module implements proper calibration for world predictions:
//! - **Brier Scores**: Proper scoring rule for probability calibration
//! - **ECE**: Expected Calibration Error for measuring calibration quality
//! - **Per-Domain Tracking**: Separate calibration for different prediction domains
//! - **Confidence Adjustment**: Automatically adjust confidence based on history
//!
//! ## Why Calibration Matters
//!
//! A well-calibrated system:
//! - When it says 80% confident, it's right ~80% of the time
//! - Knows when to be humble vs confident
//! - Earns trust through track record
//!
//! A poorly calibrated system:
//! - Claims high confidence but is often wrong (overconfident)
//! - Claims low confidence even when usually right (underconfident)
//! - Cannot be trusted to assess its own reliability
//!
//! ## Brier Score
//!
//! The Brier score is: B = (1/N) * Σ(fᵢ - oᵢ)²
//! - fᵢ is the forecast probability
//! - oᵢ is the outcome (1 if occurred, 0 if not)
//! - Lower is better (0 = perfect, 1 = maximally wrong)
//!
//! ## Expected Calibration Error (ECE)
//!
//! ECE = Σ(|Bᵢ|/N) * |acc(Bᵢ) - conf(Bᵢ)|
//! - Bins predictions by confidence level
//! - Compares accuracy in each bin to average confidence
//! - Measures systematic over/under-confidence

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use super::types::instant_now;
use super::world_prediction::{OutcomeCategory, PredictionDomain, Resolution, WorldPrediction};

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for calibration tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationConfig {
    /// Number of bins for ECE calculation
    pub ece_bins: usize,

    /// Maximum predictions to keep in history
    pub max_history: usize,

    /// Minimum predictions before meaningful ECE
    pub min_predictions_for_ece: usize,

    /// Rolling window size for recent calibration
    pub rolling_window: usize,

    /// Confidence adjustment rate (how fast to adapt)
    pub adjustment_rate: f64,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            ece_bins: 10,
            max_history: 10000,
            min_predictions_for_ece: 50,
            rolling_window: 100,
            adjustment_rate: 0.1,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CALIBRATION BIN
// ═══════════════════════════════════════════════════════════════════════════════

/// A bin for ECE calculation
#[derive(Debug, Clone, Default)]
struct CalibrationBin {
    /// Sum of confidences in this bin
    confidence_sum: f64,
    /// Number of correct predictions
    correct_count: usize,
    /// Total predictions in this bin
    total_count: usize,
}

impl CalibrationBin {
    /// Average confidence in this bin
    fn average_confidence(&self) -> f64 {
        if self.total_count == 0 {
            0.5
        } else {
            self.confidence_sum / self.total_count as f64
        }
    }

    /// Accuracy in this bin
    fn accuracy(&self) -> f64 {
        if self.total_count == 0 {
            0.0
        } else {
            self.correct_count as f64 / self.total_count as f64
        }
    }

    /// Calibration error for this bin
    fn calibration_error(&self) -> f64 {
        (self.accuracy() - self.average_confidence()).abs()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// DOMAIN CALIBRATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Calibration statistics for a single domain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainCalibration {
    /// Domain being tracked
    pub domain: PredictionDomain,

    /// Rolling Brier score (recent predictions)
    pub rolling_brier: f64,

    /// Lifetime Brier score (all predictions)
    pub lifetime_brier: f64,

    /// Expected Calibration Error.
    ///
    /// HONESTY NOTE (2026-07-16): 0.0 here means "not yet computed" until
    /// `ece_computed` is true (needs `min_predictions_for_ece` resolved
    /// predictions) — it is NOT evidence of perfect calibration. Check
    /// `ece_computed` before treating this as a measurement.
    pub ece: f64,

    /// Whether `ece` has actually been computed (vs. its 0.0 initial value).
    /// Before this flag existed, `is_well_calibrated` returned true for any
    /// system with <50 resolved predictions regardless of how miscalibrated
    /// it was (E5 observed ece=0.0/"well calibrated" at 21% accuracy).
    #[serde(default)]
    pub ece_computed: bool,

    /// Number of predictions made
    pub prediction_count: usize,

    /// Number of correct predictions
    pub correct_count: usize,

    /// Sum of squared errors (for Brier)
    brier_sum: f64,

    /// Confidence adjustment factor (multiply by this)
    pub confidence_adjustment: f64,

    /// Is this domain overconfident (ECE indicates too high confidence)?
    pub is_overconfident: bool,

    /// Recent predictions for rolling calculations
    #[serde(skip)]
    recent_predictions: VecDeque<PredictionOutcome>,

    /// When this was last updated
    #[serde(skip, default = "instant_now")]
    pub last_updated: Instant,
}

/// Record of a prediction and its outcome
#[derive(Debug, Clone)]
struct PredictionOutcome {
    confidence: f64,
    was_correct: bool,
    brier_component: f64,
}

impl DomainCalibration {
    /// Create new domain calibration
    pub fn new(domain: PredictionDomain) -> Self {
        Self {
            domain,
            rolling_brier: 0.0,
            lifetime_brier: 0.0,
            ece: 0.0,
            ece_computed: false,
            prediction_count: 0,
            correct_count: 0,
            brier_sum: 0.0,
            confidence_adjustment: 1.0, // No adjustment initially
            is_overconfident: false,
            recent_predictions: VecDeque::new(),
            last_updated: Instant::now(),
        }
    }

    /// Record a prediction outcome
    pub fn record_outcome(
        &mut self,
        confidence: f64,
        was_correct: bool,
        config: &CalibrationConfig,
    ) {
        // Calculate Brier component: (confidence - outcome)^2
        let outcome = if was_correct { 1.0 } else { 0.0 };
        let brier_component = (confidence - outcome).powi(2);

        // Update counts
        self.prediction_count += 1;
        if was_correct {
            self.correct_count += 1;
        }
        self.brier_sum += brier_component;

        // Update lifetime Brier score
        self.lifetime_brier = self.brier_sum / self.prediction_count as f64;

        // Add to recent predictions
        self.recent_predictions.push_back(PredictionOutcome {
            confidence,
            was_correct,
            brier_component,
        });

        // Trim recent predictions to rolling window
        while self.recent_predictions.len() > config.rolling_window {
            self.recent_predictions.pop_front();
        }

        // Update rolling Brier score
        if !self.recent_predictions.is_empty() {
            self.rolling_brier = self
                .recent_predictions
                .iter()
                .map(|p| p.brier_component)
                .sum::<f64>()
                / self.recent_predictions.len() as f64;
        }

        // Update ECE if we have enough predictions
        if self.prediction_count >= config.min_predictions_for_ece {
            self.update_ece(config.ece_bins);
        }

        // Update confidence adjustment based on calibration
        self.update_confidence_adjustment(config);

        self.last_updated = Instant::now();
    }

    /// Update ECE calculation
    fn update_ece(&mut self, num_bins: usize) {
        if self.recent_predictions.is_empty() {
            self.ece = 0.0;
            self.ece_computed = false; // no data — 0.0 is a sentinel, not a measurement
            return;
        }

        // Create bins
        let mut bins: Vec<CalibrationBin> = vec![CalibrationBin::default(); num_bins];

        // Populate bins
        for prediction in &self.recent_predictions {
            // Determine which bin this confidence falls into
            let bin_idx = (prediction.confidence * num_bins as f64).floor() as usize;
            let bin_idx = bin_idx.min(num_bins - 1);

            bins[bin_idx].confidence_sum += prediction.confidence;
            bins[bin_idx].total_count += 1;
            if prediction.was_correct {
                bins[bin_idx].correct_count += 1;
            }
        }

        // Calculate ECE
        let total_predictions = self.recent_predictions.len() as f64;
        self.ece = bins
            .iter()
            .map(|bin| {
                if bin.total_count == 0 {
                    0.0
                } else {
                    (bin.total_count as f64 / total_predictions) * bin.calibration_error()
                }
            })
            .sum();

        // Determine if overconfident
        // Overconfident if average confidence > accuracy
        let avg_confidence: f64 = self
            .recent_predictions
            .iter()
            .map(|p| p.confidence)
            .sum::<f64>()
            / total_predictions;
        let accuracy = self
            .recent_predictions
            .iter()
            .filter(|p| p.was_correct)
            .count() as f64
            / total_predictions;

        self.is_overconfident = avg_confidence > accuracy;
        // ECE is now a real measurement, not the 0.0 initial value.
        self.ece_computed = true;
    }

    /// Update confidence adjustment based on calibration
    fn update_confidence_adjustment(&mut self, config: &CalibrationConfig) {
        if self.recent_predictions.is_empty() {
            return;
        }

        // Calculate how much to adjust confidence
        // If overconfident: adjustment < 1.0 (reduce confidence)
        // If underconfident: adjustment > 1.0 (increase confidence)
        let avg_confidence: f64 = self
            .recent_predictions
            .iter()
            .map(|p| p.confidence)
            .sum::<f64>()
            / self.recent_predictions.len() as f64;

        let accuracy = self
            .recent_predictions
            .iter()
            .filter(|p| p.was_correct)
            .count() as f64
            / self.recent_predictions.len() as f64;

        if avg_confidence > 0.01 {
            let ideal_adjustment = accuracy / avg_confidence;
            // Smooth adjustment (don't change too fast)
            self.confidence_adjustment = self.confidence_adjustment
                * (1.0 - config.adjustment_rate)
                + ideal_adjustment * config.adjustment_rate;
            // Clamp to reasonable range
            self.confidence_adjustment = self.confidence_adjustment.clamp(0.5, 2.0);
        }
    }

    /// Adjust a raw confidence value based on calibration
    pub fn adjust_confidence(&self, raw_confidence: f64) -> f64 {
        (raw_confidence * self.confidence_adjustment).clamp(0.0, 1.0)
    }

    /// Get accuracy (percentage correct)
    pub fn accuracy(&self) -> f64 {
        if self.prediction_count == 0 {
            0.0
        } else {
            self.correct_count as f64 / self.prediction_count as f64
        }
    }

    /// Restore domain calibration from persisted state
    ///
    /// This reconstructs the calibration state from a persistence snapshot,
    /// allowing the system to resume with its learned wisdom intact.
    pub fn from_persisted(
        domain: PredictionDomain,
        rolling_brier: f64,
        lifetime_brier: f64,
        ece: f64,
        prediction_count: usize,
        correct_count: usize,
        brier_sum: f64,
        confidence_adjustment: f64,
        is_overconfident: bool,
    ) -> Self {
        Self {
            domain,
            rolling_brier,
            lifetime_brier,
            ece,
            // Persisted snapshots predate the ece_computed flag. Trust a
            // persisted nonzero ECE as computed; a persisted 0.0 is ambiguous
            // (could be the old never-computed sentinel) — treat as uncomputed
            // and let the next update_ece() re-establish it honestly.
            ece_computed: ece != 0.0,
            prediction_count,
            correct_count,
            brier_sum,
            confidence_adjustment,
            is_overconfident,
            recent_predictions: VecDeque::new(), // Will rebuild as new predictions come in
            last_updated: Instant::now(),
        }
    }

    /// Get the brier_sum for persistence
    pub fn brier_sum(&self) -> f64 {
        self.brier_sum
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BRIER SCORE TRACKER
// ═══════════════════════════════════════════════════════════════════════════════

/// Main calibration tracking system
#[derive(Debug)]
pub struct BrierScoreTracker {
    /// Configuration
    config: CalibrationConfig,

    /// Per-domain calibration
    domain_calibration: HashMap<PredictionDomain, DomainCalibration>,

    /// Global calibration (across all domains)
    global_calibration: DomainCalibration,

    /// History of all predictions (for detailed analysis)
    prediction_history: VecDeque<ResolvedPredictionRecord>,

    /// When this tracker was created
    _created_at: Instant,
}

/// Record of a resolved prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedPredictionRecord {
    /// Prediction ID
    pub prediction_id: String,

    /// Domain of prediction
    pub domain: PredictionDomain,

    /// Stated confidence
    pub confidence: f64,

    /// Was prediction correct?
    pub was_correct: bool,

    /// Brier score component
    pub brier_component: f64,

    /// Predicted outcome
    pub predicted_outcome: OutcomeCategory,

    /// Actual outcome
    pub actual_outcome: OutcomeCategory,

    /// Timestamp
    pub timestamp: u64,
}

impl BrierScoreTracker {
    /// Create a new Brier score tracker
    pub fn new(config: CalibrationConfig) -> Self {
        // Initialize domain calibration for all domains
        let mut domain_calibration = HashMap::new();
        for domain in PredictionDomain::all() {
            domain_calibration.insert(domain, DomainCalibration::new(domain));
        }

        Self {
            config,
            domain_calibration,
            global_calibration: DomainCalibration::new(PredictionDomain::Factual), // Placeholder domain
            prediction_history: VecDeque::new(),
            _created_at: Instant::now(),
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(CalibrationConfig::default())
    }

    /// Record a resolved prediction
    pub fn record_prediction(&mut self, prediction: &WorldPrediction) {
        // Extract resolution information
        let (was_correct, actual_outcome) = match &prediction.resolution {
            Resolution::True {
                observed_outcome, ..
            } => (true, *observed_outcome),
            Resolution::False {
                observed_outcome, ..
            } => (false, *observed_outcome),
            _ => return, // Can't record pending/unclear predictions
        };

        let confidence = prediction.confidence;
        let brier_component = (confidence - if was_correct { 1.0 } else { 0.0 }).powi(2);

        // Update domain calibration
        if let Some(domain_cal) = self.domain_calibration.get_mut(&prediction.domain) {
            domain_cal.record_outcome(confidence, was_correct, &self.config);
        }

        // Update global calibration
        self.global_calibration
            .record_outcome(confidence, was_correct, &self.config);

        // Add to history
        let record = ResolvedPredictionRecord {
            prediction_id: prediction.id.clone(),
            domain: prediction.domain,
            confidence,
            was_correct,
            brier_component,
            predicted_outcome: prediction.predicted_outcome,
            actual_outcome,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        };
        self.prediction_history.push_back(record);

        // Trim history
        while self.prediction_history.len() > self.config.max_history {
            self.prediction_history.pop_front();
        }
    }

    /// Get overall Brier score
    pub fn brier_score(&self) -> f64 {
        self.global_calibration.lifetime_brier
    }

    /// Get rolling Brier score (recent predictions)
    pub fn rolling_brier_score(&self) -> f64 {
        self.global_calibration.rolling_brier
    }

    /// Get Expected Calibration Error
    pub fn expected_calibration_error(&self) -> f64 {
        self.global_calibration.ece
    }

    /// Get calibration for a specific domain
    pub fn domain_calibration(&self, domain: PredictionDomain) -> Option<&DomainCalibration> {
        self.domain_calibration.get(&domain)
    }

    /// Adjust confidence based on domain calibration
    pub fn adjust_confidence(&self, domain: PredictionDomain, raw_confidence: f64) -> f64 {
        self.domain_calibration
            .get(&domain)
            .map(|cal| cal.adjust_confidence(raw_confidence))
            .unwrap_or(raw_confidence)
    }

    /// Check if system is well-calibrated (ECE below threshold).
    ///
    /// Requires the ECE to have actually been computed: before 2026-07-16 this
    /// returned true for any system with fewer than `min_predictions_for_ece`
    /// resolved predictions (ece still at its 0.0 initial value), which let a
    /// 21%-accuracy overconfident run report "well calibrated" (E5 finding).
    /// A system that hasn't measured its calibration is NOT well-calibrated —
    /// it is unmeasured.
    pub fn is_well_calibrated(&self, threshold: f64) -> bool {
        self.global_calibration.ece_computed && self.global_calibration.ece < threshold
    }

    /// Check if a domain is overconfident
    pub fn is_domain_overconfident(&self, domain: PredictionDomain) -> bool {
        self.domain_calibration
            .get(&domain)
            .map(|cal| cal.is_overconfident)
            .unwrap_or(false)
    }

    /// Get total prediction count
    pub fn total_predictions(&self) -> usize {
        self.global_calibration.prediction_count
    }

    /// Rigorous calibration analytics over the recorded prediction history:
    /// Murphy's Brier decomposition (reliability / resolution / uncertainty),
    /// expected calibration error, and a Wilson accuracy interval. Powered by
    /// `symthaea-statistics`. `None` if no predictions have resolved yet.
    pub fn calibration_report(
        &self,
        n_bins: usize,
    ) -> Option<super::calibration_analytics::CalibrationReport> {
        let pairs: Vec<(f64, bool)> = self
            .prediction_history
            .iter()
            .map(|r| (r.confidence, r.was_correct))
            .collect();
        super::calibration_analytics::analyze_pairs(&pairs, n_bins)
    }

    /// Get overall accuracy
    pub fn accuracy(&self) -> f64 {
        self.global_calibration.accuracy()
    }

    /// Get calibration summary for all domains
    pub fn calibration_summary(&self) -> CalibrationSummary {
        let domain_stats: HashMap<PredictionDomain, DomainStats> = self
            .domain_calibration
            .iter()
            .filter(|(_, cal)| cal.prediction_count > 0)
            .map(|(domain, cal)| {
                (
                    *domain,
                    DomainStats {
                        brier_score: cal.lifetime_brier,
                        ece: cal.ece,
                        accuracy: cal.accuracy(),
                        prediction_count: cal.prediction_count,
                        confidence_adjustment: cal.confidence_adjustment,
                        is_overconfident: cal.is_overconfident,
                    },
                )
            })
            .collect();

        CalibrationSummary {
            global_brier: self.global_calibration.lifetime_brier,
            global_ece: self.global_calibration.ece,
            global_accuracy: self.accuracy(),
            total_predictions: self.total_predictions(),
            domain_stats,
            is_well_calibrated: self.is_well_calibrated(0.15),
        }
    }

    /// Restore calibration tracker from persisted state
    ///
    /// This is the key method for warm starts - it reconstructs the entire
    /// calibration state from a persistence snapshot, allowing the system
    /// to resume with all its learned wisdom intact.
    ///
    /// # Arguments
    /// * `config` - Calibration configuration
    /// * `domain_data` - Per-domain persisted calibration data
    /// * `global_stats` - Global calibration statistics
    ///
    /// # Returns
    /// A fully reconstructed BrierScoreTracker with the persisted state
    pub fn from_persisted(
        config: CalibrationConfig,
        domain_data: &HashMap<PredictionDomain, super::persistence::PersistedDomainCalibration>,
        global_stats: &super::persistence::GlobalCalibrationStats,
    ) -> Self {
        // Initialize all domains (some may not have persisted data)
        let mut domain_calibration = HashMap::new();
        for domain in PredictionDomain::all() {
            let cal = if let Some(persisted) = domain_data.get(&domain) {
                DomainCalibration::from_persisted(
                    domain,
                    persisted.rolling_brier,
                    persisted.lifetime_brier,
                    persisted.ece,
                    persisted.prediction_count,
                    persisted.correct_count,
                    persisted.brier_sum,
                    persisted.confidence_adjustment,
                    persisted.is_overconfident,
                )
            } else {
                DomainCalibration::new(domain)
            };
            domain_calibration.insert(domain, cal);
        }

        // Reconstruct global calibration
        let global_calibration = DomainCalibration::from_persisted(
            PredictionDomain::Factual, // Placeholder domain for global
            global_stats.rolling_brier,
            global_stats.lifetime_brier,
            global_stats.ece,
            global_stats.total_predictions,
            global_stats.correct_predictions,
            global_stats.brier_sum,
            global_stats.confidence_adjustment,
            !global_stats.is_well_calibrated,
        );

        Self {
            config,
            domain_calibration,
            global_calibration,
            prediction_history: VecDeque::new(), // History is not persisted (too large)
            _created_at: Instant::now(),
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &CalibrationConfig {
        &self.config
    }
}

/// Summary of calibration state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationSummary {
    /// Global Brier score
    pub global_brier: f64,
    /// Global ECE
    pub global_ece: f64,
    /// Global accuracy
    pub global_accuracy: f64,
    /// Total predictions
    pub total_predictions: usize,
    /// Per-domain statistics
    pub domain_stats: HashMap<PredictionDomain, DomainStats>,
    /// Is system well-calibrated?
    pub is_well_calibrated: bool,
}

/// Statistics for a single domain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainStats {
    pub brier_score: f64,
    pub ece: f64,
    pub accuracy: f64,
    pub prediction_count: usize,
    pub confidence_adjustment: f64,
    pub is_overconfident: bool,
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::recursive_improvement::world_prediction::{
        ResolutionContract, RiskTier, WorldActionContext,
    };

    fn create_test_prediction(confidence: f64, _domain: PredictionDomain) -> WorldPrediction {
        let action =
            WorldActionContext::new("test", "test effect").with_risk_tier(RiskTier::Observation);
        let contract = ResolutionContract::shell_command();

        WorldPrediction::new(
            "Test prediction",
            OutcomeCategory::Success,
            confidence,
            action,
            contract,
        )
    }

    #[test]
    fn test_domain_calibration_creation() {
        let cal = DomainCalibration::new(PredictionDomain::CodeExecution);
        assert_eq!(cal.prediction_count, 0);
        assert_eq!(cal.confidence_adjustment, 1.0);
    }

    #[test]
    fn test_brier_score_perfect_calibration() {
        let mut tracker = BrierScoreTracker::with_defaults();

        // Record 10 predictions with 70% confidence
        // 7 correct, 3 incorrect (perfectly calibrated)
        for i in 0..10 {
            let mut pred = create_test_prediction(0.7, PredictionDomain::CodeExecution);
            if i < 7 {
                pred.resolve_true(OutcomeCategory::Success, 1.0);
            } else {
                pred.resolve_false(OutcomeCategory::SafeFailure, 1.0);
            }
            tracker.record_prediction(&pred);
        }

        // Brier score should be (0.7-1)^2 * 7 + (0.7-0)^2 * 3 = 0.09*7 + 0.49*3 = 0.63+1.47 = 2.1
        // Average: 2.1/10 = 0.21
        assert!(tracker.brier_score() > 0.0);
        assert!(tracker.brier_score() < 0.3);
    }

    #[test]
    fn test_brier_score_overconfident() {
        // Use config with lower threshold so 20 predictions is enough
        let config = CalibrationConfig {
            min_predictions_for_ece: 10,
            ..Default::default()
        };
        let mut tracker = BrierScoreTracker::new(config);

        // Record predictions with 90% confidence but only 50% accuracy
        for i in 0..20 {
            let mut pred = create_test_prediction(0.9, PredictionDomain::CodeExecution);
            if i % 2 == 0 {
                pred.resolve_true(OutcomeCategory::Success, 1.0);
            } else {
                pred.resolve_false(OutcomeCategory::SafeFailure, 1.0);
            }
            tracker.record_prediction(&pred);
        }

        // Should detect overconfidence
        assert!(tracker.is_domain_overconfident(PredictionDomain::CodeExecution));
    }

    #[test]
    fn test_confidence_adjustment() {
        let config = CalibrationConfig {
            rolling_window: 10,
            adjustment_rate: 0.5,       // Faster adjustment for test
            min_predictions_for_ece: 5, // Lower threshold so adjustment kicks in
            ..Default::default()
        };
        let mut tracker = BrierScoreTracker::new(config);

        // Record overconfident predictions (90% confidence, 50% accuracy)
        // Note: create_test_prediction uses action_type "test" which maps to CodeExecution
        for i in 0..20 {
            let mut pred = create_test_prediction(0.9, PredictionDomain::CodeExecution);
            if i % 2 == 0 {
                pred.resolve_true(OutcomeCategory::Success, 1.0);
            } else {
                pred.resolve_false(OutcomeCategory::SafeFailure, 1.0);
            }
            tracker.record_prediction(&pred);
        }

        // Confidence adjustment should reduce confidence
        // Domain is CodeExecution because action_type "test" infers to CodeExecution
        let adjusted = tracker.adjust_confidence(PredictionDomain::CodeExecution, 0.9);
        assert!(
            adjusted < 0.9,
            "Confidence should be reduced for overconfident domain"
        );
    }

    #[test]
    fn test_ece_calculation() {
        let config = CalibrationConfig {
            min_predictions_for_ece: 10,
            ece_bins: 5,
            ..Default::default()
        };
        let mut tracker = BrierScoreTracker::new(config);

        // Add predictions with varying confidence
        for i in 0..50 {
            let confidence = (i % 5) as f64 * 0.2 + 0.1; // 0.1, 0.3, 0.5, 0.7, 0.9
            let mut pred = create_test_prediction(confidence, PredictionDomain::Factual);

            // Make outcomes roughly match confidence
            let correct = (i % 100) as f64 / 100.0 < confidence;
            if correct {
                pred.resolve_true(OutcomeCategory::Success, 1.0);
            } else {
                pred.resolve_false(OutcomeCategory::SafeFailure, 1.0);
            }
            tracker.record_prediction(&pred);
        }

        // ECE should be calculated
        let ece = tracker.expected_calibration_error();
        assert!(ece >= 0.0 && ece <= 1.0);
    }

    #[test]
    fn uncalibrated_system_is_not_well_calibrated() {
        // E5 regression (2026-07-08 finding, fixed 2026-07-16): ece's 0.0
        // initial value passed the `< threshold` check before any ECE had been
        // computed, so a 21%-accuracy overconfident run reported
        // "well calibrated". A system that hasn't measured its calibration
        // must not claim to be well-calibrated.
        let config = CalibrationConfig {
            min_predictions_for_ece: 50,
            ..Default::default()
        };
        let mut tracker = BrierScoreTracker::new(config);

        // 20 maximally-overconfident wrong predictions — below the ECE gate,
        // so ECE is never computed. Pre-fix, this claimed well_calibrated=true.
        for _ in 0..20 {
            let mut pred = create_test_prediction(0.9, PredictionDomain::Factual);
            pred.resolve_false(OutcomeCategory::SafeFailure, 1.0);
            tracker.record_prediction(&pred);
        }
        assert!(
            !tracker.is_well_calibrated(0.15),
            "ECE never computed — must not claim calibration"
        );

        // Past the gate, ECE computes and correctly flags the miscalibration.
        for _ in 0..40 {
            let mut pred = create_test_prediction(0.9, PredictionDomain::Factual);
            pred.resolve_false(OutcomeCategory::SafeFailure, 1.0);
            tracker.record_prediction(&pred);
        }
        assert!(
            !tracker.is_well_calibrated(0.15),
            "60 wrong 0.9-confidence predictions is maximal miscalibration"
        );
        assert!(
            tracker.expected_calibration_error() > 0.5,
            "ECE should be large for systematic overconfidence: {}",
            tracker.expected_calibration_error()
        );
    }

    #[test]
    fn test_calibration_summary() {
        let mut tracker = BrierScoreTracker::with_defaults();

        // Add some predictions
        for _ in 0..10 {
            let mut pred = create_test_prediction(0.8, PredictionDomain::CodeExecution);
            pred.resolve_true(OutcomeCategory::Success, 1.0);
            tracker.record_prediction(&pred);
        }

        let summary = tracker.calibration_summary();
        assert_eq!(summary.total_predictions, 10);
        assert!(
            summary
                .domain_stats
                .contains_key(&PredictionDomain::CodeExecution)
        );
    }
}
