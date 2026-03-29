// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Active Inference Bridge for MAGI Loop
//!
//! This module integrates the MAGI Loop (World-Grounded Self Model) with:
//! - **PrincipledSignals**: Converting calibration data to control signals
//! - **PAC (Phase-Amplitude Coupling)**: Tracking prediction-action coupling
//! - **Active Inference**: Connecting calibrated EFE to action selection
//!
//! ## Integration Flow
//!
//! ```text
//! ┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐
//! │   MAGI Loop     │────▶│ Active Inference │────▶│ PrincipledSignals │
//! │ (Calibration)   │     │     Bridge       │     │   (Control)       │
//! └─────────────────┘     └──────────────────┘     └───────────────────┘
//!         │                       │
//!         │                       ▼
//!         │               ┌──────────────────┐
//!         └──────────────▶│  PAC Tracker     │
//!                         │ (Coupling)       │
//!                         └──────────────────┘
//! ```
//!
//! ## Key Mappings
//!
//! | MAGI Component | Active Inference Component | Meaning |
//! |----------------|---------------------------|---------|
//! | Brier Score | Prediction Error | How wrong were we? |
//! | ECE | Uncertainty | How miscalibrated are we? |
//! | Calibrated Confidence | Confidence Signal | How sure should we be? |
//! | Attribution Quality | Coherence | Can we explain errors? |

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use crate::consciousness::pac::PacTracker;
use crate::experience::signals::PrincipledSignals;

use super::{CalibrationQuality, WorldGroundedSelfModel, WorldPrediction};

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for the Active Inference bridge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveInferenceBridgeConfig {
    /// Window size for PAC computation
    pub pac_window_size: usize,

    /// Weight for calibration-derived prediction error
    pub prediction_error_weight: f32,

    /// Weight for ECE-derived uncertainty
    pub uncertainty_weight: f32,

    /// Weight for calibrated confidence
    pub confidence_weight: f32,

    /// Minimum data points before signals are meaningful
    pub min_data_points: usize,

    /// Enable PAC tracking (can be disabled for performance)
    pub enable_pac: bool,
}

impl Default for ActiveInferenceBridgeConfig {
    fn default() -> Self {
        Self {
            pac_window_size: 100,
            prediction_error_weight: 1.0,
            uncertainty_weight: 1.0,
            confidence_weight: 1.0,
            min_data_points: 10,
            enable_pac: true,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ACTIVE INFERENCE BRIDGE
// ═══════════════════════════════════════════════════════════════════════════════

/// Bridge connecting MAGI Loop to Active Inference components
pub struct ActiveInferenceBridge {
    /// Configuration
    config: ActiveInferenceBridgeConfig,

    /// PAC tracker for prediction-outcome coupling
    pac_tracker: Option<PacTracker>,

    /// Recent prediction errors for smoothing
    recent_prediction_errors: VecDeque<f64>,

    /// Recent outcomes for PAC amplitude
    recent_outcomes: VecDeque<f64>,

    /// Phase counter for PAC (oscillates based on prediction rhythm)
    phase_counter: f64,

    /// Total observations
    total_observations: usize,
}

impl ActiveInferenceBridge {
    /// Create a new bridge with configuration
    pub fn new(config: ActiveInferenceBridgeConfig) -> Self {
        let pac_tracker = if config.enable_pac {
            Some(PacTracker::new(
                "Prediction",
                "Outcome",
                config.pac_window_size,
            ))
        } else {
            None
        };

        Self {
            config,
            pac_tracker,
            recent_prediction_errors: VecDeque::with_capacity(100),
            recent_outcomes: VecDeque::with_capacity(100),
            phase_counter: 0.0,
            total_observations: 0,
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(ActiveInferenceBridgeConfig::default())
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SIGNAL GENERATION
    // ═══════════════════════════════════════════════════════════════════════════

    /// Generate PrincipledSignals from MAGI Loop state
    ///
    /// This is the core integration: converting calibration data into
    /// control signals that drive behavior.
    pub fn generate_signals(&self, model: &WorldGroundedSelfModel) -> PrincipledSignals {
        let loop_state = model.loop_state();
        let calibration_summary = model.calibration_summary();

        // Check if we have enough data
        if loop_state.predictions_resolved < self.config.min_data_points {
            return PrincipledSignals::new(); // Neutral signals
        }

        // 1. PREDICTION ERROR: From average Brier score
        // Brier score is [0, 1] where 0 is perfect, 1 is worst
        // Map to prediction_error signal
        let avg_brier = calibration_summary.global_brier;
        let prediction_error =
            (avg_brier as f32 * self.config.prediction_error_weight).clamp(0.0, 1.0);

        // 2. UNCERTAINTY: From ECE (Expected Calibration Error)
        // High ECE = high systematic miscalibration = high uncertainty
        let ece = calibration_summary.global_ece;
        let uncertainty = (ece as f32 * 2.0 * self.config.uncertainty_weight).clamp(0.0, 1.0);

        // 3. CONFIDENCE: Inverse of uncertainty, weighted by data quality
        let data_quality = match loop_state.calibration_quality {
            CalibrationQuality::Insufficient => 0.3,
            CalibrationQuality::Poor => 0.4,
            CalibrationQuality::Moderate => 0.6,
            CalibrationQuality::Good => 0.8,
            CalibrationQuality::Excellent => 0.95,
        };
        let confidence =
            ((1.0 - uncertainty) * data_quality * self.config.confidence_weight).clamp(0.0, 1.0);

        // 4. COHERENCE: From attribution quality (how well can we explain errors?)
        // If we have attributions with testable predictions, coherence is higher
        let coherence = self.compute_attribution_coherence(model);

        // 5. SALIENCE: Based on how active the MAGI loop is
        // More predictions being made = higher salience of world modeling
        let recent_activity = if loop_state.predictions_made > 0 {
            (loop_state.predictions_resolved as f32 / loop_state.predictions_made as f32)
                .clamp(0.0, 1.0)
        } else {
            0.5
        };
        let salience = recent_activity;

        PrincipledSignals {
            prediction_error,
            uncertainty,
            coherence,
            confidence,
            salience,
            phi_monitor: 0.0, // Not derived from MAGI
        }
    }

    /// Compute coherence from attribution quality
    fn compute_attribution_coherence(&self, model: &WorldGroundedSelfModel) -> f32 {
        let loop_state = model.loop_state();

        // If we have attributions for a good fraction of resolved predictions,
        // we're being coherent about our errors
        if loop_state.predictions_resolved == 0 {
            return 0.5; // Neutral
        }

        let attribution_ratio =
            loop_state.attributions_generated as f32 / loop_state.predictions_resolved as f32;

        // Good coherence: we explain most of our errors
        (attribution_ratio.clamp(0.0, 1.0) * 0.5 + 0.3).clamp(0.0, 1.0)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // PAC INTEGRATION
    // ═══════════════════════════════════════════════════════════════════════════

    /// Observe a prediction resolution for PAC tracking
    ///
    /// This feeds the prediction-outcome coupling measurement.
    ///
    /// * `confidence`: The predicted confidence (low-freq phase signal)
    /// * `success`: Whether the prediction was correct (high-freq amplitude)
    pub fn observe_resolution(&mut self, confidence: f64, success: bool) {
        use std::f64::consts::PI;

        self.total_observations += 1;

        // Track prediction error
        let error = if success { 0.0 } else { 1.0 };
        if self.recent_prediction_errors.len() >= 100 {
            self.recent_prediction_errors.pop_front();
        }
        self.recent_prediction_errors.push_back(error);

        // Track outcome
        let outcome = if success { 1.0 } else { 0.0 };
        if self.recent_outcomes.len() >= 100 {
            self.recent_outcomes.pop_front();
        }
        self.recent_outcomes.push_back(outcome);

        // Update PAC tracker if enabled
        if let Some(ref mut pac) = self.pac_tracker {
            // Phase: Convert confidence to phase [0, 2π]
            // Low confidence = early phase, high confidence = late phase
            let phase = confidence * 2.0 * PI;

            // Amplitude: Success magnitude
            // We modulate by confidence to see if confident predictions succeed more
            let amplitude = outcome * (0.5 + confidence * 0.5);

            pac.observe(phase, amplitude);
        }

        // Update phase counter for next observation
        self.phase_counter += PI / 4.0; // Quarter-cycle per observation
        if self.phase_counter > 2.0 * PI {
            self.phase_counter -= 2.0 * PI;
        }
    }

    /// Observe a prediction directly from MAGI Loop
    pub fn observe_prediction(&mut self, prediction: &WorldPrediction) {
        // Only observe if not pending (i.e., resolved)
        if !prediction.resolution.is_pending() {
            // Consider the prediction successful if it was correct
            let success = prediction.resolution.is_correct();
            self.observe_resolution(prediction.confidence, success);
        }
    }

    /// Get the current Phase-Amplitude Coupling (Modulation Index)
    ///
    /// Returns a value in [0, 1] where:
    /// - 0.0 = No coupling (predictions don't inform outcomes)
    /// - 1.0 = Perfect coupling (confidence perfectly predicts success)
    pub fn modulation_index(&self) -> Option<f64> {
        self.pac_tracker
            .as_ref()
            .map(|p| p.compute_modulation_index())
    }

    /// Get the current coupling quality assessment
    pub fn coupling_quality(&self) -> CouplingQuality {
        match self.modulation_index() {
            None => CouplingQuality::NotTracking,
            Some(mi) if self.total_observations < self.config.min_data_points => {
                CouplingQuality::InsufficientData
            }
            Some(mi) if mi < 0.1 => CouplingQuality::NoCoupling,
            Some(mi) if mi < 0.3 => CouplingQuality::WeakCoupling,
            Some(mi) if mi < 0.6 => CouplingQuality::ModerateCoupling,
            Some(_) => CouplingQuality::StrongCoupling,
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ACTION SELECTION INTEGRATION
    // ═══════════════════════════════════════════════════════════════════════════

    /// Get recommended exploration factor based on calibration state
    ///
    /// This integrates with Active Inference action selection:
    /// - Poor calibration → more exploration (higher epistemic value weight)
    /// - Good calibration → more exploitation (higher pragmatic value weight)
    pub fn exploration_factor(&self, model: &WorldGroundedSelfModel) -> f64 {
        match model.loop_state().calibration_quality {
            CalibrationQuality::Insufficient => 0.9, // Explore heavily
            CalibrationQuality::Poor => 0.7,
            CalibrationQuality::Moderate => 0.5,
            CalibrationQuality::Good => 0.3,
            CalibrationQuality::Excellent => 0.2, // Exploit more
        }
    }

    /// Adjust EFE weights based on calibration state
    ///
    /// Returns (pragmatic_weight, epistemic_weight, novelty_weight)
    pub fn adjusted_efe_weights(&self, model: &WorldGroundedSelfModel) -> (f64, f64, f64) {
        let exploration = self.exploration_factor(model);

        let pragmatic = 1.0 - exploration * 0.5; // Reduce pragmatic when exploring
        let epistemic = exploration; // Increase epistemic when exploring
        let novelty = 0.1 + exploration * 0.3; // Increase novelty penalty when exploring

        (pragmatic, epistemic, novelty)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // STATISTICS
    // ═══════════════════════════════════════════════════════════════════════════

    /// Get bridge statistics
    pub fn statistics(&self) -> BridgeStatistics {
        let avg_prediction_error = if self.recent_prediction_errors.is_empty() {
            None
        } else {
            Some(
                self.recent_prediction_errors.iter().sum::<f64>()
                    / self.recent_prediction_errors.len() as f64,
            )
        };

        let avg_outcome = if self.recent_outcomes.is_empty() {
            None
        } else {
            Some(self.recent_outcomes.iter().sum::<f64>() / self.recent_outcomes.len() as f64)
        };

        BridgeStatistics {
            total_observations: self.total_observations,
            modulation_index: self.modulation_index(),
            coupling_quality: self.coupling_quality(),
            average_prediction_error: avg_prediction_error,
            average_outcome: avg_outcome,
            pac_enabled: self.pac_tracker.is_some(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COUPLING QUALITY
// ═══════════════════════════════════════════════════════════════════════════════

/// Quality of prediction-outcome coupling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CouplingQuality {
    /// PAC tracking is disabled
    NotTracking,
    /// Not enough data to assess coupling
    InsufficientData,
    /// No meaningful coupling (MI < 0.1)
    NoCoupling,
    /// Weak coupling (MI 0.1-0.3)
    WeakCoupling,
    /// Moderate coupling (MI 0.3-0.6)
    ModerateCoupling,
    /// Strong coupling (MI > 0.6)
    StrongCoupling,
}

impl CouplingQuality {
    /// Static string matching Debug output — avoids `format!("{:?}")` on hot path.
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NotTracking => "NotTracking",
            Self::InsufficientData => "InsufficientData",
            Self::NoCoupling => "NoCoupling",
            Self::WeakCoupling => "WeakCoupling",
            Self::ModerateCoupling => "ModerateCoupling",
            Self::StrongCoupling => "StrongCoupling",
        }
    }

    /// Is coupling meaningful?
    pub fn is_meaningful(&self) -> bool {
        matches!(
            self,
            Self::WeakCoupling | Self::ModerateCoupling | Self::StrongCoupling
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// STATISTICS
// ═══════════════════════════════════════════════════════════════════════════════

/// Statistics from the Active Inference bridge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeStatistics {
    /// Total observations processed
    pub total_observations: usize,
    /// Current Modulation Index (if PAC enabled)
    pub modulation_index: Option<f64>,
    /// Current coupling quality
    pub coupling_quality: CouplingQuality,
    /// Average prediction error (recent)
    pub average_prediction_error: Option<f64>,
    /// Average outcome (recent)
    pub average_outcome: Option<f64>,
    /// Whether PAC tracking is enabled
    pub pac_enabled: bool,
}

// ═══════════════════════════════════════════════════════════════════════════════
// INTEGRATED CONTROLLER
// ═══════════════════════════════════════════════════════════════════════════════

/// Integrated controller combining MAGI Loop with Active Inference
///
/// This is the main entry point for using the MAGI Loop with the
/// Active Inference framework.
pub struct MagiActiveInferenceController {
    /// The world-grounded self model (MAGI Loop)
    pub model: WorldGroundedSelfModel,

    /// The Active Inference bridge
    pub bridge: ActiveInferenceBridge,
}

impl MagiActiveInferenceController {
    /// Create a new controller with default configurations
    pub fn new() -> Self {
        Self {
            model: WorldGroundedSelfModel::with_defaults(),
            bridge: ActiveInferenceBridge::with_defaults(),
        }
    }

    /// Create with custom configurations
    pub fn with_configs(
        model_config: super::WorldGroundedConfig,
        bridge_config: ActiveInferenceBridgeConfig,
    ) -> Self {
        Self {
            model: WorldGroundedSelfModel::new(model_config),
            bridge: ActiveInferenceBridge::new(bridge_config),
        }
    }

    /// Get current control signals
    pub fn signals(&self) -> PrincipledSignals {
        self.bridge.generate_signals(&self.model)
    }

    /// Check if the system should ask for clarification
    pub fn should_clarify(&self) -> bool {
        self.signals().should_clarify()
    }

    /// Get exploration factor for action selection
    pub fn exploration_factor(&self) -> f64 {
        self.bridge.exploration_factor(&self.model)
    }

    /// Get adjusted EFE weights
    pub fn efe_weights(&self) -> (f64, f64, f64) {
        self.bridge.adjusted_efe_weights(&self.model)
    }

    /// Get bridge statistics
    pub fn statistics(&self) -> BridgeStatistics {
        self.bridge.statistics()
    }
}

impl Default for MagiActiveInferenceController {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_creation() {
        let bridge = ActiveInferenceBridge::with_defaults();
        assert!(bridge.pac_tracker.is_some());
        assert_eq!(bridge.total_observations, 0);
    }

    #[test]
    fn test_signal_generation_insufficient_data() {
        let bridge = ActiveInferenceBridge::with_defaults();
        let model = WorldGroundedSelfModel::with_defaults();

        let signals = bridge.generate_signals(&model);

        // With no data, signals should be neutral
        assert_eq!(signals.prediction_error, 0.5);
        assert_eq!(signals.uncertainty, 0.5);
    }

    #[test]
    fn test_observe_resolution() {
        let mut bridge = ActiveInferenceBridge::with_defaults();

        // Observe some resolutions
        for i in 0..20 {
            let success = i % 3 != 0; // 2/3 success rate
            bridge.observe_resolution(0.7, success);
        }

        assert_eq!(bridge.total_observations, 20);
        assert!(bridge.recent_prediction_errors.len() == 20);

        // Check that we have a modulation index
        let mi = bridge.modulation_index();
        assert!(mi.is_some());
    }

    #[test]
    fn test_coupling_quality() {
        let bridge = ActiveInferenceBridge::with_defaults();

        // Initially insufficient data
        assert_eq!(bridge.coupling_quality(), CouplingQuality::InsufficientData);
    }

    #[test]
    fn test_exploration_factor() {
        let bridge = ActiveInferenceBridge::with_defaults();
        let model = WorldGroundedSelfModel::with_defaults();

        let factor = bridge.exploration_factor(&model);

        // With insufficient data, should explore heavily
        assert!(factor > 0.7);
    }

    #[test]
    fn test_controller_creation() {
        let controller = MagiActiveInferenceController::new();

        let signals = controller.signals();
        assert!(signals.prediction_error >= 0.0 && signals.prediction_error <= 1.0);
    }

    #[test]
    fn test_adjusted_efe_weights() {
        let bridge = ActiveInferenceBridge::with_defaults();
        let model = WorldGroundedSelfModel::with_defaults();

        let (pragmatic, epistemic, _novelty) = bridge.adjusted_efe_weights(&model);

        // With insufficient data, should favor exploration
        assert!(epistemic > 0.7);
        assert!(pragmatic < 0.8);
    }
}
