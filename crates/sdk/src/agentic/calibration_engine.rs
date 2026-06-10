// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Calibration Engine - SCEI Component
//!
//! The Calibration Engine adjusts confidence scores and agent trust profiles
//! based on historical prediction accuracy. It provides:
//!
//! - **Brier Score Tracking**: Measures prediction accuracy over time
//! - **Calibration Curves**: Maps predicted confidence to actual outcomes
//! - **Agent Calibration Profiles**: Per-agent accuracy tracking
//! - **Domain-Specific Calibration**: Calibration varies by knowledge domain
//! - **Epistemic-Stratified Calibration**: Track calibration by E-level
//! - **Temporal Decay**: Recent predictions weighted more heavily
//! - **Recalibration Triggers**: Automatic adjustments when calibration drifts
//!
//! # Integration with MATL
//!
//! Calibration scores feed into K-Vector updates:
//! - Well-calibrated agents get k_p (performance) boosts
//! - Poorly calibrated agents face k_i (integrity) penalties
//! - Systematic overconfidence triggers k_r (reputation) decay
//!
//! # Epistemic Integration
//!
//! Calibration is tracked per epistemic level (E0-E4):
//! - E0 (Speculative): Expected higher variance, looser calibration
//! - E1-E2 (Observational/Peer-Reviewed): Standard calibration
//! - E3-E4 (Cryptographic/Formal): Tighter calibration expected

use crate::epistemic::EmpiricalLevel;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(feature = "ts-export")]
use ts_rs::TS;

// ============================================================================
// Core Types
// ============================================================================

/// Calibration bin for mapping predicted to actual probabilities
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct CalibrationBin {
    /// Lower bound of predicted probability (inclusive)
    pub lower_bound: f64,
    /// Upper bound of predicted probability (exclusive)
    pub upper_bound: f64,
    /// Number of predictions in this bin
    pub count: u64,
    /// Sum of actual outcomes (0 or 1)
    pub outcome_sum: f64,
    /// Average predicted probability in this bin
    pub avg_predicted: f64,
    /// Actual frequency of positive outcomes
    pub actual_frequency: f64,
}

impl CalibrationBin {
    /// Create a new calibration bin with the given probability range
    pub fn new(lower: f64, upper: f64) -> Self {
        Self {
            lower_bound: lower,
            upper_bound: upper,
            count: 0,
            outcome_sum: 0.0,
            avg_predicted: 0.0,
            actual_frequency: 0.0,
        }
    }

    /// Add a prediction and its actual outcome to this bin
    pub fn add_prediction(&mut self, predicted: f64, outcome: bool) {
        let outcome_val = if outcome { 1.0 } else { 0.0 };
        self.outcome_sum += outcome_val;
        self.avg_predicted =
            (self.avg_predicted * self.count as f64 + predicted) / (self.count + 1) as f64;
        self.count += 1;
        self.actual_frequency = self.outcome_sum / self.count as f64;
    }

    /// Calibration error for this bin (predicted - actual)
    pub fn calibration_error(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.avg_predicted - self.actual_frequency
    }
}

/// Calibration curve with multiple bins
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct CalibrationCurve {
    /// Bins from 0.0 to 1.0
    pub bins: Vec<CalibrationBin>,
    /// Total predictions tracked
    pub total_predictions: u64,
    /// Expected Calibration Error (ECE)
    pub ece: f64,
    /// Maximum Calibration Error (MCE)
    pub mce: f64,
    /// Overall Brier score
    pub brier_score: f64,
}

impl CalibrationCurve {
    /// Create a new calibration curve with n bins
    pub fn new(num_bins: usize) -> Self {
        let bin_width = 1.0 / num_bins as f64;
        let bins = (0..num_bins)
            .map(|i| {
                let lower = i as f64 * bin_width;
                let upper = (i + 1) as f64 * bin_width;
                CalibrationBin::new(lower, upper)
            })
            .collect();

        Self {
            bins,
            total_predictions: 0,
            ece: 0.0,
            mce: 0.0,
            brier_score: 0.0,
        }
    }

    /// Add a prediction and its outcome
    pub fn add_prediction(&mut self, predicted: f64, outcome: bool) {
        let predicted = predicted.clamp(0.0, 0.9999);
        let bin_idx = (predicted * self.bins.len() as f64) as usize;
        let bin_idx = bin_idx.min(self.bins.len() - 1);

        self.bins[bin_idx].add_prediction(predicted, outcome);
        self.total_predictions += 1;

        // Update Brier score (running average)
        let outcome_val = if outcome { 1.0 } else { 0.0 };
        let squared_error = (predicted - outcome_val).powi(2);
        self.brier_score = (self.brier_score * (self.total_predictions - 1) as f64 + squared_error)
            / self.total_predictions as f64;

        // Recalculate ECE and MCE
        self.recalculate_errors();
    }

    fn recalculate_errors(&mut self) {
        let mut ece = 0.0;
        let mut mce = 0.0f64;

        for bin in &self.bins {
            if bin.count > 0 {
                let weight = bin.count as f64 / self.total_predictions as f64;
                let error = bin.calibration_error().abs();
                ece += weight * error;
                mce = mce.max(error);
            }
        }

        self.ece = ece;
        self.mce = mce;
    }

    /// Get calibration quality rating
    pub fn quality_rating(&self) -> CalibrationQuality {
        if self.total_predictions < 50 {
            return CalibrationQuality::Insufficient;
        }

        match self.ece {
            e if e < 0.02 => CalibrationQuality::Excellent,
            e if e < 0.05 => CalibrationQuality::Good,
            e if e < 0.10 => CalibrationQuality::Fair,
            e if e < 0.20 => CalibrationQuality::Poor,
            _ => CalibrationQuality::VeryPoor,
        }
    }

    /// Suggest adjusted probability based on calibration curve
    pub fn recalibrate(&self, predicted: f64) -> f64 {
        if self.total_predictions < 50 {
            return predicted; // Not enough data to recalibrate
        }

        let predicted = predicted.clamp(0.0, 0.9999);
        let bin_idx = (predicted * self.bins.len() as f64) as usize;
        let bin_idx = bin_idx.min(self.bins.len() - 1);

        let bin = &self.bins[bin_idx];
        if bin.count < 10 {
            return predicted; // Not enough data in this bin
        }

        // Adjust towards actual frequency
        let adjustment = bin.calibration_error();
        (predicted - adjustment * 0.5).clamp(0.0, 1.0)
    }
}

/// Calibration quality levels
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub enum CalibrationQuality {
    /// ECE < 0.02
    Excellent,
    /// ECE < 0.05
    Good,
    /// ECE < 0.10
    Fair,
    /// ECE < 0.20
    Poor,
    /// ECE >= 0.20
    VeryPoor,
    /// Not enough predictions yet
    Insufficient,
}

// ============================================================================
// Agent Calibration Profile
// ============================================================================

/// Per-agent calibration tracking
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct AgentCalibrationProfile {
    /// Agent identifier
    pub agent_id: String,
    /// Overall calibration curve
    pub overall_curve: CalibrationCurve,
    /// Domain-specific curves
    pub domain_curves: HashMap<String, CalibrationCurve>,
    /// Recent predictions for trend analysis
    pub recent_brier_scores: Vec<f64>,
    /// Calibration trend (positive = improving)
    pub trend: f64,
    /// Last recalibration timestamp
    pub last_recalibration: u64,
    /// Recalibration trigger count
    pub recalibration_count: u32,
}

impl AgentCalibrationProfile {
    /// Create a new calibration profile for an agent
    pub fn new(agent_id: String) -> Self {
        Self {
            agent_id,
            overall_curve: CalibrationCurve::new(10),
            domain_curves: HashMap::new(),
            recent_brier_scores: Vec::new(),
            trend: 0.0,
            last_recalibration: 0,
            recalibration_count: 0,
        }
    }

    /// Record a prediction and outcome
    pub fn record_prediction(&mut self, predicted: f64, outcome: bool, domain: Option<&str>) {
        self.overall_curve.add_prediction(predicted, outcome);

        if let Some(domain) = domain {
            let curve = self
                .domain_curves
                .entry(domain.to_string())
                .or_insert_with(|| CalibrationCurve::new(10));
            curve.add_prediction(predicted, outcome);
        }

        // Track recent Brier scores for trend
        let outcome_val = if outcome { 1.0 } else { 0.0 };
        let brier = (predicted - outcome_val).powi(2);
        self.recent_brier_scores.push(brier);

        // Keep only last 100 scores
        if self.recent_brier_scores.len() > 100 {
            self.recent_brier_scores.remove(0);
        }

        self.update_trend();
    }

    fn update_trend(&mut self) {
        if self.recent_brier_scores.len() < 20 {
            self.trend = 0.0;
            return;
        }

        let len = self.recent_brier_scores.len();
        let first_half: f64 =
            self.recent_brier_scores[..len / 2].iter().sum::<f64>() / (len / 2) as f64;
        let second_half: f64 =
            self.recent_brier_scores[len / 2..].iter().sum::<f64>() / (len / 2) as f64;

        // Negative trend = improving (lower Brier is better)
        self.trend = first_half - second_half;
    }

    /// Check if recalibration is needed
    pub fn needs_recalibration(&self) -> bool {
        let quality = self.overall_curve.quality_rating();
        matches!(
            quality,
            CalibrationQuality::Poor | CalibrationQuality::VeryPoor
        ) && self.overall_curve.total_predictions >= 100
    }

    /// Get recalibrated probability
    pub fn recalibrate_prediction(&self, predicted: f64, domain: Option<&str>) -> f64 {
        // Try domain-specific first
        if let Some(domain) = domain {
            if let Some(curve) = self.domain_curves.get(domain) {
                if curve.total_predictions >= 50 {
                    return curve.recalibrate(predicted);
                }
            }
        }

        // Fall back to overall
        self.overall_curve.recalibrate(predicted)
    }

    /// Calculate K-Vector adjustment based on calibration
    pub fn kvector_adjustment(&self) -> KVectorCalibrationAdjustment {
        let quality = self.overall_curve.quality_rating();
        let ece = self.overall_curve.ece;
        let trend = self.trend;

        let k_p_delta = match quality {
            CalibrationQuality::Excellent => 0.05,
            CalibrationQuality::Good => 0.02,
            CalibrationQuality::Fair => 0.0,
            CalibrationQuality::Poor => -0.03,
            CalibrationQuality::VeryPoor => -0.08,
            CalibrationQuality::Insufficient => 0.0,
        };

        let k_i_delta = if ece > 0.15 {
            -0.05 // Integrity penalty for systematic miscalibration
        } else if ece < 0.03 {
            0.02 // Integrity boost for excellent calibration
        } else {
            0.0
        };

        let k_r_delta = if trend > 0.05 {
            0.02 // Reputation boost for improving
        } else if trend < -0.05 {
            -0.02 // Reputation penalty for degrading
        } else {
            0.0
        };

        KVectorCalibrationAdjustment {
            k_p_delta,
            k_i_delta,
            k_r_delta,
            quality,
            ece,
            trend,
        }
    }
}

/// K-Vector adjustments from calibration
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct KVectorCalibrationAdjustment {
    /// Performance dimension delta
    pub k_p_delta: f32,
    /// Integrity dimension delta
    pub k_i_delta: f32,
    /// Reputation dimension delta
    pub k_r_delta: f32,
    /// Overall calibration quality
    pub quality: CalibrationQuality,
    /// Expected Calibration Error
    pub ece: f64,
    /// Calibration trend (positive = improving)
    pub trend: f64,
}

// ============================================================================
// Calibration Engine
// ============================================================================

/// Configuration for the calibration engine
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CalibrationEngineConfig {
    /// Number of bins in calibration curves
    pub num_bins: usize,
    /// Minimum predictions before recalibration kicks in
    pub min_predictions: u64,
    /// ECE threshold to trigger recalibration alert
    pub recalibration_threshold: f64,
    /// How often to check for recalibration (predictions)
    pub check_interval: u64,
}

impl Default for CalibrationEngineConfig {
    fn default() -> Self {
        Self {
            num_bins: 10,
            min_predictions: 50,
            recalibration_threshold: 0.15,
            check_interval: 100,
        }
    }
}

/// The Calibration Engine
#[allow(dead_code)]
pub struct CalibrationEngine {
    config: CalibrationEngineConfig,
    profiles: HashMap<String, AgentCalibrationProfile>,
    global_curve: CalibrationCurve,
}

impl CalibrationEngine {
    /// Create a new calibration engine with the given configuration
    pub fn new(config: CalibrationEngineConfig) -> Self {
        Self {
            profiles: HashMap::new(),
            global_curve: CalibrationCurve::new(config.num_bins),
            config,
        }
    }

    /// Record a prediction outcome
    pub fn record(&mut self, agent_id: &str, predicted: f64, outcome: bool, domain: Option<&str>) {
        // Update global curve
        self.global_curve.add_prediction(predicted, outcome);

        // Update agent profile
        let profile = self
            .profiles
            .entry(agent_id.to_string())
            .or_insert_with(|| AgentCalibrationProfile::new(agent_id.to_string()));
        profile.record_prediction(predicted, outcome, domain);
    }

    /// Get agent's calibration profile
    pub fn get_profile(&self, agent_id: &str) -> Option<&AgentCalibrationProfile> {
        self.profiles.get(agent_id)
    }

    /// Get recalibrated prediction for an agent
    pub fn recalibrate(&self, agent_id: &str, predicted: f64, domain: Option<&str>) -> f64 {
        if let Some(profile) = self.profiles.get(agent_id) {
            profile.recalibrate_prediction(predicted, domain)
        } else {
            // Use global curve for unknown agents
            self.global_curve.recalibrate(predicted)
        }
    }

    /// Get K-Vector adjustment for an agent
    pub fn get_kvector_adjustment(&self, agent_id: &str) -> Option<KVectorCalibrationAdjustment> {
        self.profiles.get(agent_id).map(|p| p.kvector_adjustment())
    }

    /// Get agents needing recalibration
    pub fn agents_needing_recalibration(&self) -> Vec<String> {
        self.profiles
            .iter()
            .filter(|(_, p)| p.needs_recalibration())
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Get global calibration statistics
    pub fn global_stats(&self) -> CalibrationStats {
        CalibrationStats {
            total_predictions: self.global_curve.total_predictions,
            ece: self.global_curve.ece,
            mce: self.global_curve.mce,
            brier_score: self.global_curve.brier_score,
            quality: self.global_curve.quality_rating(),
            agent_count: self.profiles.len(),
            poorly_calibrated_agents: self.agents_needing_recalibration().len(),
        }
    }
}

/// Global calibration statistics
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct CalibrationStats {
    /// Total predictions across all agents
    pub total_predictions: u64,
    /// Expected Calibration Error
    pub ece: f64,
    /// Maximum Calibration Error
    pub mce: f64,
    /// Overall Brier score
    pub brier_score: f64,
    /// Overall calibration quality
    pub quality: CalibrationQuality,
    /// Number of agents tracked
    pub agent_count: usize,
    /// Agents with poor/very poor calibration
    pub poorly_calibrated_agents: usize,
}

// ============================================================================
// Epistemic-Stratified Calibration
// ============================================================================

/// Calibration tracking stratified by epistemic level
///
/// Different empirical levels have different expected calibration:
/// - E0 (Speculative): Higher variance acceptable
/// - E3-E4 (Cryptographic/Formal): Tighter calibration expected
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct EpistemicCalibrationProfile {
    /// Calibration curves per E-level (0-4)
    pub e_level_curves: [CalibrationCurve; 5],
    /// Expected ECE thresholds per E-level (looser for speculative)
    pub expected_ece: [f64; 5],
    /// Overconfidence tracking (predicted - actual, positive = overconfident)
    pub overconfidence_by_level: [f64; 5],
    /// Count of predictions per level
    pub predictions_by_level: [u64; 5],
}

impl Default for EpistemicCalibrationProfile {
    fn default() -> Self {
        Self {
            e_level_curves: [
                CalibrationCurve::new(10),
                CalibrationCurve::new(10),
                CalibrationCurve::new(10),
                CalibrationCurve::new(10),
                CalibrationCurve::new(10),
            ],
            // E0: 0.20 ECE acceptable (speculative)
            // E1: 0.15 ECE acceptable (observational)
            // E2: 0.10 ECE acceptable (peer-reviewed)
            // E3: 0.05 ECE acceptable (cryptographic)
            // E4: 0.03 ECE acceptable (formal proof)
            expected_ece: [0.20, 0.15, 0.10, 0.05, 0.03],
            overconfidence_by_level: [0.0; 5],
            predictions_by_level: [0; 5],
        }
    }
}

impl EpistemicCalibrationProfile {
    /// Record a prediction with its epistemic level
    pub fn record(&mut self, e_level: EmpiricalLevel, predicted: f64, outcome: bool) {
        let idx = e_level.as_index();
        self.e_level_curves[idx].add_prediction(predicted, outcome);
        self.predictions_by_level[idx] += 1;

        // Track overconfidence
        let outcome_val = if outcome { 1.0 } else { 0.0 };
        let n = self.predictions_by_level[idx] as f64;
        self.overconfidence_by_level[idx] =
            (self.overconfidence_by_level[idx] * (n - 1.0) + (predicted - outcome_val)) / n;
    }

    /// Check if calibration at a level is within expected bounds
    pub fn is_level_calibrated(&self, e_level: EmpiricalLevel) -> bool {
        let idx = e_level.as_index();
        let curve = &self.e_level_curves[idx];

        if curve.total_predictions < 30 {
            return true; // Insufficient data
        }

        curve.ece <= self.expected_ece[idx]
    }

    /// Get systematic overconfidence across levels (weighted by prediction count)
    pub fn systematic_overconfidence(&self) -> f64 {
        let total: u64 = self.predictions_by_level.iter().sum();
        if total == 0 {
            return 0.0;
        }

        self.overconfidence_by_level
            .iter()
            .zip(self.predictions_by_level.iter())
            .map(|(oc, count)| oc * (*count as f64 / total as f64))
            .sum()
    }

    /// Get combined epistemic calibration quality
    pub fn epistemic_quality(&self) -> EpistemicCalibrationQuality {
        let mut calibrated_levels = 0;
        let mut total_levels = 0;

        for (idx, curve) in self.e_level_curves.iter().enumerate() {
            if curve.total_predictions >= 30 {
                total_levels += 1;
                if curve.ece <= self.expected_ece[idx] {
                    calibrated_levels += 1;
                }
            }
        }

        if total_levels == 0 {
            return EpistemicCalibrationQuality::Insufficient;
        }

        let ratio = calibrated_levels as f64 / total_levels as f64;
        let overconfidence = self.systematic_overconfidence();

        match (ratio, overconfidence) {
            (r, oc) if r >= 0.8 && oc.abs() < 0.05 => EpistemicCalibrationQuality::WellCalibrated,
            (r, oc) if r >= 0.6 && oc.abs() < 0.10 => {
                EpistemicCalibrationQuality::ModeratelyCalibrated
            }
            (_, oc) if oc > 0.10 => EpistemicCalibrationQuality::SystematicallyOverconfident,
            (_, oc) if oc < -0.10 => EpistemicCalibrationQuality::SystematicallyUnderconfident,
            _ => EpistemicCalibrationQuality::PoorlyCalibrated,
        }
    }
}

/// Epistemic calibration quality assessment
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub enum EpistemicCalibrationQuality {
    /// Well calibrated across epistemic levels
    WellCalibrated,
    /// Moderately calibrated
    ModeratelyCalibrated,
    /// Systematically overconfident (k_i penalty)
    SystematicallyOverconfident,
    /// Systematically underconfident (minor k_p penalty)
    SystematicallyUnderconfident,
    /// Poorly calibrated
    PoorlyCalibrated,
    /// Insufficient data
    Insufficient,
}

// ============================================================================
// Temporal Decay
// ============================================================================

/// Prediction with timestamp for temporal decay
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TimestampedPrediction {
    /// Predicted probability
    pub predicted: f64,
    /// Actual outcome
    pub outcome: bool,
    /// Unix timestamp
    pub timestamp: u64,
    /// Epistemic level (0-4)
    pub e_level: Option<u8>,
    /// Knowledge domain
    pub domain: Option<String>,
}

/// Calibration curve with temporal decay
///
/// Recent predictions are weighted more heavily than old ones.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TemporalCalibrationCurve {
    /// Recent predictions (kept for recalculation)
    pub predictions: Vec<TimestampedPrediction>,
    /// Maximum predictions to keep
    pub max_predictions: usize,
    /// Half-life in seconds (predictions older than this get 0.5 weight)
    pub half_life_secs: u64,
    /// Current weighted Brier score
    pub weighted_brier: f64,
    /// Current weighted ECE
    pub weighted_ece: f64,
}

impl TemporalCalibrationCurve {
    /// Create a new temporal calibration curve
    pub fn new(max_predictions: usize, half_life_secs: u64) -> Self {
        Self {
            predictions: Vec::with_capacity(max_predictions),
            max_predictions,
            half_life_secs,
            weighted_brier: 0.0,
            weighted_ece: 0.0,
        }
    }

    /// Add a new prediction
    pub fn add(
        &mut self,
        predicted: f64,
        outcome: bool,
        timestamp: u64,
        e_level: Option<u8>,
        domain: Option<String>,
    ) {
        self.predictions.push(TimestampedPrediction {
            predicted,
            outcome,
            timestamp,
            e_level,
            domain,
        });

        // Prune old predictions
        if self.predictions.len() > self.max_predictions {
            self.predictions.remove(0);
        }

        self.recalculate(timestamp);
    }

    /// Calculate temporal weight for a prediction
    fn weight(&self, pred_timestamp: u64, current_timestamp: u64) -> f64 {
        if pred_timestamp >= current_timestamp {
            return 1.0;
        }
        let age_secs = current_timestamp - pred_timestamp;
        0.5_f64.powf(age_secs as f64 / self.half_life_secs as f64)
    }

    /// Recalculate weighted metrics
    fn recalculate(&mut self, current_timestamp: u64) {
        if self.predictions.is_empty() {
            self.weighted_brier = 0.0;
            self.weighted_ece = 0.0;
            return;
        }

        let mut total_weight = 0.0;
        let mut weighted_brier_sum = 0.0;

        // Build weighted bins for ECE
        let mut bins: Vec<(f64, f64, f64)> = vec![(0.0, 0.0, 0.0); 10]; // (weighted_sum_pred, weighted_sum_outcome, weight)

        for pred in &self.predictions {
            let w = self.weight(pred.timestamp, current_timestamp);
            total_weight += w;

            let outcome_val = if pred.outcome { 1.0 } else { 0.0 };
            weighted_brier_sum += w * (pred.predicted - outcome_val).powi(2);

            let bin_idx = ((pred.predicted * 10.0) as usize).min(9);
            bins[bin_idx].0 += w * pred.predicted;
            bins[bin_idx].1 += w * outcome_val;
            bins[bin_idx].2 += w;
        }

        self.weighted_brier = if total_weight > 0.0 {
            weighted_brier_sum / total_weight
        } else {
            0.0
        };

        // Calculate weighted ECE
        let mut ece = 0.0;
        for (sum_pred, sum_outcome, bin_weight) in &bins {
            if *bin_weight > 0.01 {
                let avg_pred = sum_pred / bin_weight;
                let avg_outcome = sum_outcome / bin_weight;
                ece += (bin_weight / total_weight) * (avg_pred - avg_outcome).abs();
            }
        }
        self.weighted_ece = ece;
    }

    /// Get recent calibration trend (negative = improving)
    pub fn trend(&self, _current_timestamp: u64) -> f64 {
        if self.predictions.len() < 20 {
            return 0.0;
        }

        let mid = self.predictions.len() / 2;
        let older: Vec<_> = self.predictions[..mid].iter().collect();
        let newer: Vec<_> = self.predictions[mid..].iter().collect();

        let older_brier: f64 = older
            .iter()
            .map(|p| {
                let o = if p.outcome { 1.0 } else { 0.0 };
                (p.predicted - o).powi(2)
            })
            .sum::<f64>()
            / older.len() as f64;

        let newer_brier: f64 = newer
            .iter()
            .map(|p| {
                let o = if p.outcome { 1.0 } else { 0.0 };
                (p.predicted - o).powi(2)
            })
            .sum::<f64>()
            / newer.len() as f64;

        // Positive = degrading, negative = improving
        newer_brier - older_brier
    }
}

// ============================================================================
// Enhanced Agent Profile with Epistemic + Temporal
// ============================================================================

/// Enhanced agent calibration profile with epistemic stratification and temporal decay
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnhancedAgentCalibrationProfile {
    /// Agent identifier
    pub agent_id: String,
    /// Basic calibration (backward compatible)
    pub basic_profile: AgentCalibrationProfile,
    /// Epistemic-stratified calibration
    pub epistemic_profile: EpistemicCalibrationProfile,
    /// Temporal calibration curve
    pub temporal_curve: TemporalCalibrationCurve,
    /// Overconfidence bias (systematic tendency to overestimate)
    pub overconfidence_bias: f64,
    /// Underconfidence bias (systematic tendency to underestimate)
    pub underconfidence_bias: f64,
}

impl EnhancedAgentCalibrationProfile {
    /// Create a new enhanced calibration profile
    pub fn new(agent_id: String) -> Self {
        Self {
            agent_id: agent_id.clone(),
            basic_profile: AgentCalibrationProfile::new(agent_id),
            epistemic_profile: EpistemicCalibrationProfile::default(),
            temporal_curve: TemporalCalibrationCurve::new(500, 86400 * 7), // 7 day half-life
            overconfidence_bias: 0.0,
            underconfidence_bias: 0.0,
        }
    }

    /// Record a prediction with full context
    pub fn record_full(
        &mut self,
        predicted: f64,
        outcome: bool,
        timestamp: u64,
        e_level: Option<EmpiricalLevel>,
        domain: Option<&str>,
    ) {
        // Basic profile
        self.basic_profile
            .record_prediction(predicted, outcome, domain);

        // Epistemic profile
        if let Some(level) = e_level {
            self.epistemic_profile.record(level, predicted, outcome);
        }

        // Temporal curve
        self.temporal_curve.add(
            predicted,
            outcome,
            timestamp,
            e_level.map(|l| l.as_index() as u8),
            domain.map(|s| s.to_string()),
        );

        // Update bias tracking
        let outcome_val = if outcome { 1.0 } else { 0.0 };
        let error = predicted - outcome_val;

        // Exponential moving average of bias
        let alpha = 0.05;
        if error > 0.0 {
            self.overconfidence_bias = self.overconfidence_bias * (1.0 - alpha) + error * alpha;
        } else {
            self.underconfidence_bias =
                self.underconfidence_bias * (1.0 - alpha) + (-error) * alpha;
        }
    }

    /// Get comprehensive K-Vector adjustment
    pub fn comprehensive_kvector_adjustment(
        &self,
        current_timestamp: u64,
    ) -> ComprehensiveCalibrationAdjustment {
        let basic_adj = self.basic_profile.kvector_adjustment();
        let epistemic_quality = self.epistemic_profile.epistemic_quality();
        let trend = self.temporal_curve.trend(current_timestamp);
        let systematic_oc = self.epistemic_profile.systematic_overconfidence();

        // Additional adjustments based on epistemic calibration
        let epistemic_k_p_delta = match epistemic_quality {
            EpistemicCalibrationQuality::WellCalibrated => 0.03,
            EpistemicCalibrationQuality::ModeratelyCalibrated => 0.01,
            EpistemicCalibrationQuality::SystematicallyOverconfident => -0.04,
            EpistemicCalibrationQuality::SystematicallyUnderconfident => -0.01,
            EpistemicCalibrationQuality::PoorlyCalibrated => -0.03,
            EpistemicCalibrationQuality::Insufficient => 0.0,
        };

        // Integrity penalty for systematic overconfidence
        let epistemic_k_i_delta = if systematic_oc > 0.15 {
            -0.06 // Severe integrity penalty
        } else if systematic_oc > 0.10 {
            -0.03
        } else {
            0.0
        };

        // Trend-based reputation adjustment
        let trend_k_r_delta = if trend < -0.05 {
            0.02 // Improving
        } else if trend > 0.05 {
            -0.02 // Degrading
        } else {
            0.0
        };

        ComprehensiveCalibrationAdjustment {
            // Combine basic and epistemic adjustments
            k_p_delta: basic_adj.k_p_delta + epistemic_k_p_delta as f32,
            k_i_delta: basic_adj.k_i_delta + epistemic_k_i_delta as f32,
            k_r_delta: basic_adj.k_r_delta + trend_k_r_delta as f32,
            basic_quality: basic_adj.quality,
            epistemic_quality,
            trend,
            systematic_overconfidence: systematic_oc,
            temporal_brier: self.temporal_curve.weighted_brier,
            temporal_ece: self.temporal_curve.weighted_ece,
        }
    }
}

/// Comprehensive calibration adjustment combining all signals
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct ComprehensiveCalibrationAdjustment {
    /// Performance adjustment
    pub k_p_delta: f32,
    /// Integrity adjustment
    pub k_i_delta: f32,
    /// Reputation adjustment
    pub k_r_delta: f32,
    /// Basic calibration quality
    pub basic_quality: CalibrationQuality,
    /// Epistemic calibration quality
    pub epistemic_quality: EpistemicCalibrationQuality,
    /// Calibration trend (negative = improving)
    pub trend: f64,
    /// Systematic overconfidence measure
    pub systematic_overconfidence: f64,
    /// Temporal-weighted Brier score
    pub temporal_brier: f64,
    /// Temporal-weighted ECE
    pub temporal_ece: f64,
}

// ============================================================================
// InstrumentalActor Integration
// ============================================================================

use super::InstrumentalActor;
use crate::matl::KVector;

/// Apply calibration adjustments to an agent's K-Vector
pub fn apply_calibration_to_agent(
    agent: &mut InstrumentalActor,
    adjustment: &ComprehensiveCalibrationAdjustment,
) -> KVector {
    let mut new_kvector = agent.k_vector;

    // Apply bounded adjustments
    new_kvector.k_p = (new_kvector.k_p + adjustment.k_p_delta).clamp(0.0, 1.0);
    new_kvector.k_i = (new_kvector.k_i + adjustment.k_i_delta).clamp(0.0, 1.0);
    new_kvector.k_r = (new_kvector.k_r + adjustment.k_r_delta).clamp(0.0, 1.0);

    agent.k_vector = new_kvector;
    new_kvector
}

/// Trait for calibration integration with agents
pub trait CalibratedAgent {
    /// Get agent ID for calibration lookup
    fn calibration_id(&self) -> &str;

    /// Record a calibration prediction
    fn record_calibration_prediction(
        &mut self,
        engine: &mut CalibrationEngine,
        predicted: f64,
        outcome: bool,
        domain: Option<&str>,
    );
}

impl CalibratedAgent for InstrumentalActor {
    fn calibration_id(&self) -> &str {
        self.agent_id.as_str()
    }

    fn record_calibration_prediction(
        &mut self,
        engine: &mut CalibrationEngine,
        predicted: f64,
        outcome: bool,
        domain: Option<&str>,
    ) {
        engine.record(self.agent_id.as_str(), predicted, outcome, domain);

        // Auto-adjust K-Vector if calibration suggests it
        if let Some(adj) = engine.get_kvector_adjustment(self.agent_id.as_str()) {
            // Only apply if sufficient data
            if let Some(profile) = engine.get_profile(self.agent_id.as_str()) {
                if profile.overall_curve.total_predictions >= 50 {
                    self.k_vector.k_p = (self.k_vector.k_p + adj.k_p_delta).clamp(0.0, 1.0);
                    self.k_vector.k_i = (self.k_vector.k_i + adj.k_i_delta).clamp(0.0, 1.0);
                    self.k_vector.k_r = (self.k_vector.k_r + adj.k_r_delta).clamp(0.0, 1.0);
                }
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentic::UncertaintyCalibration;

    #[test]
    fn test_calibration_bin() {
        let mut bin = CalibrationBin::new(0.6, 0.7);

        // Perfect calibration: 65% confidence, 65% success rate
        for i in 0..100 {
            bin.add_prediction(0.65, i < 65);
        }

        assert_eq!(bin.count, 100);
        assert!((bin.actual_frequency - 0.65).abs() < 0.01);
        assert!(bin.calibration_error().abs() < 0.01);
    }

    #[test]
    fn test_overconfident_bin() {
        let mut bin = CalibrationBin::new(0.8, 0.9);

        // Overconfident: 85% confidence but only 50% success
        for i in 0..100 {
            bin.add_prediction(0.85, i < 50);
        }

        assert_eq!(bin.count, 100);
        assert!((bin.actual_frequency - 0.50).abs() < 0.01);
        // Calibration error should be ~0.35 (overconfident)
        assert!(bin.calibration_error() > 0.3);
    }

    #[test]
    fn test_calibration_curve() {
        let mut curve = CalibrationCurve::new(10);

        // Well-calibrated predictions using deterministic pattern
        for i in 0..100 {
            // 20% confidence, ~20% success rate
            curve.add_prediction(0.2, i % 5 == 0);
            // 50% confidence, ~50% success rate
            curve.add_prediction(0.5, i % 2 == 0);
            // 80% confidence, ~80% success rate
            curve.add_prediction(0.8, i % 5 != 0);
        }

        assert_eq!(curve.total_predictions, 300);
        // ECE should be low for well-calibrated predictions
        assert!(curve.ece < 0.15);
    }

    #[test]
    fn test_calibration_engine() {
        let mut engine = CalibrationEngine::new(CalibrationEngineConfig::default());

        // Record some predictions for an agent
        for i in 0..100 {
            let predicted = 0.7;
            let outcome = i < 70; // 70% success rate matches prediction
            engine.record("agent-1", predicted, outcome, Some("science"));
        }

        let profile = engine.get_profile("agent-1").unwrap();
        assert_eq!(profile.overall_curve.total_predictions, 100);

        // Should be well-calibrated
        let adjustment = profile.kvector_adjustment();
        assert!(adjustment.k_p_delta >= 0.0);
    }

    #[test]
    fn test_recalibration() {
        let mut engine = CalibrationEngine::new(CalibrationEngineConfig::default());

        // Overconfident agent: always predicts 90% but only 50% success
        for i in 0..100 {
            engine.record("overconfident", 0.9, i < 50, None);
        }

        // Recalibration should lower the prediction
        let recalibrated = engine.recalibrate("overconfident", 0.9, None);
        assert!(recalibrated < 0.9);
    }

    #[test]
    fn test_kvector_adjustment() {
        let mut profile = AgentCalibrationProfile::new("test".to_string());

        // Perfect calibration using deterministic pattern
        for i in 0..100 {
            let p = (i % 10) as f64 / 10.0 + 0.05;
            // Simulate calibrated outcomes based on probability threshold
            let outcome = (i as f64 / 100.0) < p;
            profile.record_prediction(p, outcome, None);
        }

        let adj = profile.kvector_adjustment();
        // Should not have major penalties
        assert!(adj.k_p_delta >= -0.05);
        assert!(adj.k_i_delta >= -0.05);
    }

    #[test]
    fn test_epistemic_calibration_profile() {
        use crate::epistemic::EmpiricalLevel;

        let mut profile = EpistemicCalibrationProfile::default();

        // E0 (speculative) - higher variance acceptable
        for i in 0..50 {
            profile.record(EmpiricalLevel::E0Null, 0.7, i < 30); // 60% when predicting 70%
        }

        // E3 (cryptographic) - should be well calibrated
        for i in 0..50 {
            profile.record(EmpiricalLevel::E3Cryptographic, 0.8, i < 40); // 80% when predicting 80%
        }

        // E0 should be within looser bounds
        assert!(profile.e_level_curves[0].ece < 0.25);

        // E3 should be well calibrated
        assert!(profile.e_level_curves[3].ece < 0.10);
    }

    #[test]
    fn test_systematic_overconfidence() {
        use crate::epistemic::EmpiricalLevel;

        let mut profile = EpistemicCalibrationProfile::default();

        // Systematically overconfident: predict 90%, get 50%
        for i in 0..100 {
            profile.record(EmpiricalLevel::E1Testimonial, 0.9, i < 50);
        }

        let overconfidence = profile.systematic_overconfidence();
        assert!(overconfidence > 0.3); // Clear overconfidence
        assert!(matches!(
            profile.epistemic_quality(),
            EpistemicCalibrationQuality::SystematicallyOverconfident
        ));
    }

    #[test]
    fn test_temporal_calibration_curve() {
        let mut curve = TemporalCalibrationCurve::new(100, 86400); // 1 day half-life

        let base_time = 1000000u64;

        // Old predictions (should get lower weight)
        for i in 0..30 {
            curve.add(0.7, i < 21, base_time - 172800 + i as u64, None, None); // 2 days ago, 70% acc
        }

        // Recent predictions (should get higher weight)
        for i in 0..30 {
            curve.add(0.7, i < 14, base_time + i as u64, None, None); // Now, only ~47% acc
        }

        // Temporal weighting should show degradation
        let trend = curve.trend(base_time);
        assert!(trend > 0.0); // Positive = degrading
    }

    #[test]
    fn test_enhanced_agent_profile() {
        use crate::epistemic::EmpiricalLevel;

        let mut profile = EnhancedAgentCalibrationProfile::new("agent-1".to_string());

        let timestamp = 1000000u64;

        // Record mixed predictions
        for i in 0..100 {
            let predicted = 0.7;
            let outcome = i < 70; // Well-calibrated
            profile.record_full(
                predicted,
                outcome,
                timestamp + i as u64,
                Some(EmpiricalLevel::E2PrivateVerify),
                Some("science"),
            );
        }

        let adj = profile.comprehensive_kvector_adjustment(timestamp + 100);

        // Should be reasonably calibrated
        assert!(adj.k_p_delta >= -0.05);
        assert!(adj.systematic_overconfidence.abs() < 0.1);
    }

    #[test]
    fn test_apply_calibration_to_agent() {
        use super::super::{AgentClass, AgentConstraints, AgentId, AgentStatus, EpistemicStats};
        use crate::matl::KVector;

        let mut agent = InstrumentalActor {
            agent_id: AgentId::generate(),
            sponsor_did: "did:test:sponsor".to_string(),
            agent_class: AgentClass::Supervised,
            kredit_balance: 5000,
            kredit_cap: 10000,
            constraints: AgentConstraints::default(),
            behavior_log: vec![],
            status: AgentStatus::Active,
            created_at: 0,
            last_activity: 0,
            actions_this_hour: 0,
            k_vector: KVector::new(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
            epistemic_stats: EpistemicStats::default(),
            output_history: vec![],
            uncertainty_calibration: UncertaintyCalibration::default(),
            pending_escalations: vec![],
        };

        let adjustment = ComprehensiveCalibrationAdjustment {
            k_p_delta: 0.05,
            k_i_delta: -0.02,
            k_r_delta: 0.03,
            basic_quality: CalibrationQuality::Good,
            epistemic_quality: EpistemicCalibrationQuality::ModeratelyCalibrated,
            trend: -0.01,
            systematic_overconfidence: 0.02,
            temporal_brier: 0.15,
            temporal_ece: 0.04,
        };

        let new_kvector = apply_calibration_to_agent(&mut agent, &adjustment);

        assert!((new_kvector.k_p - 0.55).abs() < 0.01);
        assert!((new_kvector.k_i - 0.48).abs() < 0.01);
        assert!((new_kvector.k_r - 0.53).abs() < 0.01);
    }
}
