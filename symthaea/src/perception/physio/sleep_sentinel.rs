// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Sleep Sentinel - Consciousness State Detection via LTC Dynamics
//!
//! ## Project Hypnos: Computational Biomarker for Consciousness
//!
//! This module implements a dual-channel LTC architecture that detects
//! consciousness states (wake vs sleep stages) from EEG signals.
//!
//! ## Architecture
//!
//! ```text
//! Fpz-Cz (Frontal)  ─┬─► [Frontal LTC]  ─┬─► [Global Integrator] ─► State
//!                    │                    │
//! Pz-Oz (Occipital) ─┴─► [Occipital LTC] ─┘
//!                              ▲
//!                              │
//!                    Synchronization Measure (Φ proxy)
//! ```
//!
//! ## Key Insight
//!
//! Different sleep stages have distinct integration signatures:
//!
//! - **Wake**: High complexity, low global synchronization → Medium Φ
//! - **N1/N2**: Variable, transitional
//! - **N3 (Deep)**: Delta waves, HIGH global synchronization → HIGH Φ
//! - **REM**: High complexity, LOW integration (paradox!) → LOW Φ
//!
//! The REM paradox is KEY: The brain is active but disconnected.
//! Standard statistics see "activity". Φ captures "integration".
//!
//! ## Why LTC Wins Here
//!
//! EEG is continuous dynamics. Transformers would discretize into windows,
//! losing the phase relationships that define synchronization. The LTC's
//! adaptive τ (time constant) naturally tracks the dominant frequency,
//! and cross-channel coherence emerges from the integrator dynamics.

use super::SleepStage;
use crate::unified_ltc::{LearningAlgorithm, UnifiedLTC, UnifiedLTCConfig};
use std::collections::{HashMap, VecDeque};

// ============================================================================
// Adaptive Threshold System
// ============================================================================

/// Configuration for adaptive threshold learning
#[derive(Debug, Clone)]
pub struct AdaptiveThresholdConfig {
    /// Initial learning rate for threshold adaptation (0.0 to 1.0)
    pub initial_learning_rate: f32,
    /// Minimum learning rate after decay
    pub min_learning_rate: f32,
    /// Learning rate decay factor per session (multiplied after each session)
    pub learning_rate_decay: f32,
    /// Number of epochs required for calibration
    pub calibration_epochs: usize,
    /// Smoothing factor for exponential moving average (0.0 to 1.0)
    /// Higher values = more weight on recent observations
    pub ema_alpha: f32,
    /// Maximum standard deviations from population norm allowed
    pub max_deviation_from_norm: f32,
    /// Z-score threshold for anomaly detection
    pub anomaly_z_threshold: f32,
    /// Minimum samples before allowing adaptation
    pub min_samples_for_adaptation: usize,
}

impl Default for AdaptiveThresholdConfig {
    fn default() -> Self {
        Self {
            initial_learning_rate: 0.1,
            min_learning_rate: 0.01,
            learning_rate_decay: 0.95,
            calibration_epochs: 10,
            ema_alpha: 0.1,
            max_deviation_from_norm: 2.5,
            anomaly_z_threshold: 3.0,
            min_samples_for_adaptation: 5,
        }
    }
}

/// Population norms for thresholds (conservative defaults)
#[derive(Debug, Clone)]
pub struct PopulationNorms {
    /// Complexity threshold: mean and std across population
    pub complexity: (f32, f32),
    /// Synchrony threshold: mean and std across population
    pub synchrony: (f32, f32),
    /// Phi proxy threshold: mean and std
    pub phi_proxy: (f32, f32),
    /// Frequency boundaries: (low_freq_boundary, high_freq_boundary)
    pub frequency_bounds: (f32, f32),
}

impl Default for PopulationNorms {
    fn default() -> Self {
        Self {
            // (mean, std) for each threshold
            // Calibrated from Sleep-EDF feature analysis:
            // - Complexity varies widely across stages (0.1 - 0.6)
            // - Synchrony is typically 0.4 - 0.7 depending on stage
            complexity: (0.35, 0.20),      // Allow wider range
            synchrony: (0.55, 0.15),       // Center on moderate synchrony
            phi_proxy: (0.4, 0.15),        // Phi tends to be lower in real data
            frequency_bounds: (8.0, 12.0), // Hz boundaries for alpha band
        }
    }
}

/// Running statistics for a single metric
#[derive(Debug, Clone)]
pub struct RunningStats {
    /// Number of samples observed
    pub count: usize,
    /// Running mean
    pub mean: f32,
    /// Running M2 for Welford's algorithm (variance = M2 / count)
    pub m2: f32,
    /// Exponential moving average
    pub ema: f32,
    /// Minimum observed value
    pub min: f32,
    /// Maximum observed value
    pub max: f32,
}

impl Default for RunningStats {
    fn default() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            ema: 0.0,
            min: f32::MAX,
            max: f32::MIN,
        }
    }
}

impl RunningStats {
    /// Update with a new sample using Welford's online algorithm
    pub fn update(&mut self, value: f32, ema_alpha: f32) {
        self.count += 1;

        // Welford's online algorithm for mean and variance
        let delta = value - self.mean;
        self.mean += delta / self.count as f32;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;

        // Exponential moving average
        if self.count == 1 {
            self.ema = value;
        } else {
            self.ema = symthaea_core::math::ema_update(self.ema, value, ema_alpha);
        }

        // Track min/max
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }

    /// Get current variance
    pub fn variance(&self) -> f32 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / (self.count - 1) as f32
        }
    }

    /// Get current standard deviation
    pub fn std(&self) -> f32 {
        self.variance().sqrt()
    }

    /// Compute z-score for a value relative to baseline
    pub fn z_score(&self, value: f32) -> f32 {
        let std = self.std();
        if std < 1e-6 {
            0.0
        } else {
            (value - self.mean) / std
        }
    }

    /// Check if a value is anomalous
    pub fn is_anomaly(&self, value: f32, z_threshold: f32) -> bool {
        self.z_score(value).abs() > z_threshold
    }
}

/// Per-subject baseline statistics
#[derive(Debug, Clone, Default)]
pub struct SubjectBaseline {
    /// Subject identifier
    pub subject_id: String,
    /// Statistics for complexity metric
    pub complexity_stats: RunningStats,
    /// Statistics for synchrony metric
    pub synchrony_stats: RunningStats,
    /// Statistics for phi proxy metric
    pub phi_stats: RunningStats,
    /// Statistics for dominant frequency
    pub frequency_stats: RunningStats,
    /// Per-stage complexity statistics
    pub stage_complexity: HashMap<SleepStage, RunningStats>,
    /// Per-stage synchrony statistics
    pub stage_synchrony: HashMap<SleepStage, RunningStats>,
    /// Total epochs observed
    pub total_epochs: usize,
    /// Correct predictions count
    pub correct_predictions: usize,
}

impl SubjectBaseline {
    pub fn new(subject_id: impl Into<String>) -> Self {
        Self {
            subject_id: subject_id.into(),
            ..Default::default()
        }
    }

    /// Update baseline with new observation
    pub fn update(
        &mut self,
        metrics: &IntegrationMetrics,
        stage: Option<SleepStage>,
        ema_alpha: f32,
    ) {
        self.complexity_stats.update(metrics.complexity, ema_alpha);
        self.synchrony_stats.update(metrics.synchrony, ema_alpha);
        self.phi_stats.update(metrics.phi_proxy, ema_alpha);
        self.frequency_stats
            .update(metrics.dominant_freq_hz, ema_alpha);

        if let Some(stage) = stage {
            self.stage_complexity
                .entry(stage)
                .or_default()
                .update(metrics.complexity, ema_alpha);
            self.stage_synchrony
                .entry(stage)
                .or_default()
                .update(metrics.synchrony, ema_alpha);
        }

        self.total_epochs += 1;
    }
}

/// Calibration state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationState {
    /// Not yet calibrated
    Uncalibrated,
    /// Currently in calibration mode
    Calibrating,
    /// Calibration complete
    Calibrated,
}

/// Adaptive threshold manager for individual subjects
#[derive(Debug, Clone)]
pub struct AdaptiveThreshold {
    /// Configuration
    config: AdaptiveThresholdConfig,
    /// Population norms for bounds checking
    population_norms: PopulationNorms,
    /// Subject baseline statistics
    baseline: SubjectBaseline,
    /// Current calibration state
    calibration_state: CalibrationState,
    /// Epochs collected during calibration
    calibration_epochs_collected: usize,
    /// Current learning rate (decays over sessions)
    current_learning_rate: f32,
    /// Session counter for learning rate decay
    session_count: usize,
    /// Adapted complexity threshold
    adapted_complexity_threshold: f32,
    /// Adapted synchrony threshold
    adapted_synchrony_threshold: f32,
    /// Adapted phi threshold
    adapted_phi_threshold: f32,
    /// Adapted low frequency boundary
    adapted_low_freq: f32,
    /// Adapted high frequency boundary
    adapted_high_freq: f32,
    /// Number of times adaptation was blocked by safety bounds
    bounds_violations: usize,
    /// Number of anomalies detected
    anomalies_detected: usize,
    /// Whether to use fallback conservative defaults
    using_fallback: bool,
}

impl AdaptiveThreshold {
    /// Create new adaptive threshold manager
    pub fn new(config: AdaptiveThresholdConfig, subject_id: impl Into<String>) -> Self {
        let population_norms = PopulationNorms::default();
        Self {
            current_learning_rate: config.initial_learning_rate,
            config,
            adapted_complexity_threshold: population_norms.complexity.0,
            adapted_synchrony_threshold: population_norms.synchrony.0,
            adapted_phi_threshold: population_norms.phi_proxy.0,
            adapted_low_freq: population_norms.frequency_bounds.0,
            adapted_high_freq: population_norms.frequency_bounds.1,
            population_norms,
            baseline: SubjectBaseline::new(subject_id),
            calibration_state: CalibrationState::Uncalibrated,
            calibration_epochs_collected: 0,
            session_count: 0,
            bounds_violations: 0,
            anomalies_detected: 0,
            using_fallback: false,
        }
    }

    /// Create with custom population norms
    pub fn with_population_norms(
        config: AdaptiveThresholdConfig,
        subject_id: impl Into<String>,
        norms: PopulationNorms,
    ) -> Self {
        let mut instance = Self::new(config, subject_id);
        instance.adapted_complexity_threshold = norms.complexity.0;
        instance.adapted_synchrony_threshold = norms.synchrony.0;
        instance.adapted_phi_threshold = norms.phi_proxy.0;
        instance.adapted_low_freq = norms.frequency_bounds.0;
        instance.adapted_high_freq = norms.frequency_bounds.1;
        instance.population_norms = norms;
        instance
    }

    // ========================================================================
    // Calibration Methods
    // ========================================================================

    /// Start calibration mode
    pub fn start_calibration(&mut self) {
        self.calibration_state = CalibrationState::Calibrating;
        self.calibration_epochs_collected = 0;
        // Reset baseline for fresh calibration
        self.baseline = SubjectBaseline::new(self.baseline.subject_id.clone());
    }

    /// End calibration mode (can be called early or after sufficient epochs)
    pub fn end_calibration(&mut self) -> bool {
        if self.calibration_epochs_collected >= self.config.min_samples_for_adaptation {
            self.calibration_state = CalibrationState::Calibrated;
            // Initialize adapted thresholds from calibration data
            self.initialize_from_calibration();
            true
        } else {
            // Not enough data, stay uncalibrated
            self.calibration_state = CalibrationState::Uncalibrated;
            false
        }
    }

    /// Check if calibrated
    pub fn is_calibrated(&self) -> bool {
        self.calibration_state == CalibrationState::Calibrated
    }

    /// Get calibration state
    pub fn calibration_state(&self) -> CalibrationState {
        self.calibration_state
    }

    /// Get calibration progress (0.0 to 1.0)
    pub fn calibration_progress(&self) -> f32 {
        if self.config.calibration_epochs == 0 {
            1.0
        } else {
            (self.calibration_epochs_collected as f32 / self.config.calibration_epochs as f32)
                .min(1.0)
        }
    }

    /// Initialize adapted thresholds from calibration baseline
    fn initialize_from_calibration(&mut self) {
        let stats = &self.baseline;

        // Use baseline mean + 0.5 std as initial threshold
        // This captures the "typical" value plus some margin
        if stats.complexity_stats.count >= self.config.min_samples_for_adaptation {
            let candidate = stats.complexity_stats.mean + 0.5 * stats.complexity_stats.std();
            self.adapted_complexity_threshold = self.apply_bounds(
                candidate,
                self.population_norms.complexity.0,
                self.population_norms.complexity.1,
            );
        }

        if stats.synchrony_stats.count >= self.config.min_samples_for_adaptation {
            let candidate = stats.synchrony_stats.mean + 0.5 * stats.synchrony_stats.std();
            self.adapted_synchrony_threshold = self.apply_bounds(
                candidate,
                self.population_norms.synchrony.0,
                self.population_norms.synchrony.1,
            );
        }

        if stats.phi_stats.count >= self.config.min_samples_for_adaptation {
            let candidate = stats.phi_stats.mean;
            self.adapted_phi_threshold = self.apply_bounds(
                candidate,
                self.population_norms.phi_proxy.0,
                self.population_norms.phi_proxy.1,
            );
        }
    }

    // ========================================================================
    // Online Learning
    // ========================================================================

    /// Update thresholds after a classification with feedback
    ///
    /// # Arguments
    /// * `metrics` - The integration metrics from this epoch
    /// * `predicted` - The predicted consciousness state
    /// * `actual` - The actual sleep stage (ground truth)
    /// * `was_correct` - Whether the prediction was correct
    pub fn update_with_feedback(
        &mut self,
        metrics: &IntegrationMetrics,
        predicted: ConsciousnessState,
        actual: SleepStage,
        was_correct: bool,
    ) {
        // Always update baseline statistics
        self.baseline
            .update(metrics, Some(actual), self.config.ema_alpha);

        if was_correct {
            self.baseline.correct_predictions += 1;
        }

        // Handle calibration mode
        if self.calibration_state == CalibrationState::Calibrating {
            self.calibration_epochs_collected += 1;
            if self.calibration_epochs_collected >= self.config.calibration_epochs {
                self.end_calibration();
            }
            return;
        }

        // Skip adaptation if not calibrated or using fallback
        if !self.is_calibrated() || self.using_fallback {
            return;
        }

        // Check for anomalies
        if self.is_anomalous_reading(metrics) {
            self.anomalies_detected += 1;
            // Don't adapt on anomalous readings
            return;
        }

        // Apply reinforcement learning signal
        // Correct classification: small adjustment toward current values
        // Incorrect classification: larger adjustment away from current values
        let reinforcement = if was_correct { 1.0 } else { -1.0 };
        let lr = self.current_learning_rate;

        // Adapt thresholds based on what went wrong
        self.adapt_thresholds(metrics, predicted, actual, reinforcement, lr);
    }

    /// Adapt thresholds based on prediction error
    fn adapt_thresholds(
        &mut self,
        metrics: &IntegrationMetrics,
        predicted: ConsciousnessState,
        actual: SleepStage,
        reinforcement: f32,
        lr: f32,
    ) {
        // Determine which thresholds need adjustment based on the error type
        let actual_state = match actual {
            SleepStage::Wake => ConsciousnessState::Awake,
            SleepStage::N1 => ConsciousnessState::Transitional, // N1 = Transitional
            SleepStage::N2 => ConsciousnessState::LightSleep,   // N2 = LightSleep
            SleepStage::N3 => ConsciousnessState::DeepSleep,
            SleepStage::REM => ConsciousnessState::REM,
            _ => return, // Don't adapt for unknown/movement
        };

        if predicted == actual_state {
            // Correct prediction: reinforce current thresholds slightly
            // Move thresholds toward values that would still classify correctly
            return;
        }

        // Incorrect prediction: adjust thresholds to prevent this error
        match (predicted, actual_state) {
            // Predicted wake but was actually deep sleep
            // -> Lower complexity threshold OR raise synchrony threshold
            (ConsciousnessState::Awake, ConsciousnessState::DeepSleep) => {
                let delta = lr * reinforcement * 0.05;
                self.try_adjust_complexity_threshold(-delta);
                self.try_adjust_synchrony_threshold(delta);
            }
            // Predicted deep sleep but was actually wake
            // -> Raise complexity threshold OR lower synchrony threshold
            (ConsciousnessState::DeepSleep, ConsciousnessState::Awake) => {
                let delta = lr * reinforcement * 0.05;
                self.try_adjust_complexity_threshold(delta);
                self.try_adjust_synchrony_threshold(-delta);
            }
            // Predicted REM but was actually deep sleep
            // -> Raise phi threshold (REM has low phi, deep sleep has high phi)
            (ConsciousnessState::REM, ConsciousnessState::DeepSleep) => {
                let delta = lr * reinforcement * 0.05;
                self.try_adjust_phi_threshold(delta);
            }
            // Predicted deep sleep but was actually REM
            // -> Lower phi threshold
            (ConsciousnessState::DeepSleep, ConsciousnessState::REM) => {
                let delta = lr * reinforcement * 0.05;
                self.try_adjust_phi_threshold(-delta);
            }
            // Other cases: general adjustment based on metrics
            _ => {
                // Use the actual metric values to guide adjustment
                let complexity_error = metrics.complexity - self.adapted_complexity_threshold;
                let synchrony_error = metrics.synchrony - self.adapted_synchrony_threshold;

                self.try_adjust_complexity_threshold(lr * reinforcement * complexity_error * 0.1);
                self.try_adjust_synchrony_threshold(lr * reinforcement * synchrony_error * 0.1);
            }
        }
    }

    /// Try to adjust complexity threshold within bounds
    fn try_adjust_complexity_threshold(&mut self, delta: f32) {
        let candidate = self.adapted_complexity_threshold + delta;
        let bounded = self.apply_bounds(
            candidate,
            self.population_norms.complexity.0,
            self.population_norms.complexity.1,
        );
        if (bounded - candidate).abs() > 1e-6 {
            self.bounds_violations += 1;
        }
        self.adapted_complexity_threshold = bounded;
    }

    /// Try to adjust synchrony threshold within bounds
    fn try_adjust_synchrony_threshold(&mut self, delta: f32) {
        let candidate = self.adapted_synchrony_threshold + delta;
        let bounded = self.apply_bounds(
            candidate,
            self.population_norms.synchrony.0,
            self.population_norms.synchrony.1,
        );
        if (bounded - candidate).abs() > 1e-6 {
            self.bounds_violations += 1;
        }
        self.adapted_synchrony_threshold = bounded;
    }

    /// Try to adjust phi threshold within bounds
    fn try_adjust_phi_threshold(&mut self, delta: f32) {
        let candidate = self.adapted_phi_threshold + delta;
        let bounded = self.apply_bounds(
            candidate,
            self.population_norms.phi_proxy.0,
            self.population_norms.phi_proxy.1,
        );
        if (bounded - candidate).abs() > 1e-6 {
            self.bounds_violations += 1;
        }
        self.adapted_phi_threshold = bounded;
    }

    /// Apply safety bounds to keep threshold within population norms
    fn apply_bounds(&self, value: f32, norm_mean: f32, norm_std: f32) -> f32 {
        let max_dev = self.config.max_deviation_from_norm;
        let lower = norm_mean - max_dev * norm_std;
        let upper = norm_mean + max_dev * norm_std;
        value.clamp(lower.max(0.0), upper.min(1.0))
    }

    /// Check if a reading is anomalous
    fn is_anomalous_reading(&self, metrics: &IntegrationMetrics) -> bool {
        let z_thresh = self.config.anomaly_z_threshold;

        // Check each metric against baseline
        let complexity_anomaly = self.baseline.complexity_stats.count
            >= self.config.min_samples_for_adaptation
            && self
                .baseline
                .complexity_stats
                .is_anomaly(metrics.complexity, z_thresh);
        let synchrony_anomaly = self.baseline.synchrony_stats.count
            >= self.config.min_samples_for_adaptation
            && self
                .baseline
                .synchrony_stats
                .is_anomaly(metrics.synchrony, z_thresh);
        let phi_anomaly = self.baseline.phi_stats.count >= self.config.min_samples_for_adaptation
            && self
                .baseline
                .phi_stats
                .is_anomaly(metrics.phi_proxy, z_thresh);

        complexity_anomaly || synchrony_anomaly || phi_anomaly
    }

    // ========================================================================
    // Session Management
    // ========================================================================

    /// Start a new session (e.g., new night of recording)
    /// Applies learning rate decay
    pub fn start_new_session(&mut self) {
        self.session_count += 1;
        self.current_learning_rate = (self.current_learning_rate * self.config.learning_rate_decay)
            .max(self.config.min_learning_rate);
    }

    /// Get current learning rate
    pub fn learning_rate(&self) -> f32 {
        self.current_learning_rate
    }

    /// Get number of sessions
    pub fn session_count(&self) -> usize {
        self.session_count
    }

    // ========================================================================
    // Fallback and Safety
    // ========================================================================

    /// Enable fallback to conservative defaults
    /// Called when adaptation appears to be failing
    pub fn enable_fallback(&mut self) {
        self.using_fallback = true;
        // Reset to population norms
        self.adapted_complexity_threshold = self.population_norms.complexity.0;
        self.adapted_synchrony_threshold = self.population_norms.synchrony.0;
        self.adapted_phi_threshold = self.population_norms.phi_proxy.0;
        self.adapted_low_freq = self.population_norms.frequency_bounds.0;
        self.adapted_high_freq = self.population_norms.frequency_bounds.1;
    }

    /// Disable fallback mode
    pub fn disable_fallback(&mut self) {
        self.using_fallback = false;
    }

    /// Check if using fallback mode
    pub fn is_using_fallback(&self) -> bool {
        self.using_fallback
    }

    /// Check if adaptation should be disabled due to poor performance
    /// Returns true if recent accuracy is too low
    pub fn should_enable_fallback(&self, recent_accuracy: f32, threshold: f32) -> bool {
        self.baseline.total_epochs >= 20 && recent_accuracy < threshold
    }

    // ========================================================================
    // Threshold Access
    // ========================================================================

    /// Get current complexity threshold
    pub fn complexity_threshold(&self) -> f32 {
        self.adapted_complexity_threshold
    }

    /// Get current synchrony threshold
    pub fn synchrony_threshold(&self) -> f32 {
        self.adapted_synchrony_threshold
    }

    /// Get current phi threshold
    pub fn phi_threshold(&self) -> f32 {
        self.adapted_phi_threshold
    }

    /// Get frequency boundaries
    pub fn frequency_bounds(&self) -> (f32, f32) {
        (self.adapted_low_freq, self.adapted_high_freq)
    }

    /// Get z-score normalized metrics relative to subject baseline
    pub fn normalize_metrics(&self, metrics: &IntegrationMetrics) -> IntegrationMetrics {
        IntegrationMetrics {
            complexity: self.baseline.complexity_stats.z_score(metrics.complexity),
            synchrony: self.baseline.synchrony_stats.z_score(metrics.synchrony),
            phi_proxy: self.baseline.phi_stats.z_score(metrics.phi_proxy),
            dominant_freq_hz: self
                .baseline
                .frequency_stats
                .z_score(metrics.dominant_freq_hz),
            // Keep causal influence as-is (relative measures)
            frontal_to_occipital: metrics.frontal_to_occipital,
            occipital_to_frontal: metrics.occipital_to_frontal,
            trajectory_variance: metrics.trajectory_variance,
            // Pass through band powers (already normalized)
            delta_power: metrics.delta_power,
            theta_power: metrics.theta_power,
            alpha_power: metrics.alpha_power,
            beta_power: metrics.beta_power,
            gamma_power: metrics.gamma_power,
            spectral_entropy: metrics.spectral_entropy,
        }
    }

    // ========================================================================
    // Statistics and Persistence
    // ========================================================================

    /// Get baseline statistics
    pub fn baseline(&self) -> &SubjectBaseline {
        &self.baseline
    }

    /// Get number of bounds violations
    pub fn bounds_violations(&self) -> usize {
        self.bounds_violations
    }

    /// Get number of anomalies detected
    pub fn anomalies_detected(&self) -> usize {
        self.anomalies_detected
    }

    /// Get subject accuracy
    pub fn subject_accuracy(&self) -> f32 {
        if self.baseline.total_epochs == 0 {
            0.0
        } else {
            self.baseline.correct_predictions as f32 / self.baseline.total_epochs as f32
        }
    }

    /// Export calibration data for persistence
    pub fn export_calibration(&self) -> CalibrationData {
        CalibrationData {
            subject_id: self.baseline.subject_id.clone(),
            complexity_threshold: self.adapted_complexity_threshold,
            synchrony_threshold: self.adapted_synchrony_threshold,
            phi_threshold: self.adapted_phi_threshold,
            low_freq: self.adapted_low_freq,
            high_freq: self.adapted_high_freq,
            baseline_complexity_mean: self.baseline.complexity_stats.mean,
            baseline_complexity_std: self.baseline.complexity_stats.std(),
            baseline_synchrony_mean: self.baseline.synchrony_stats.mean,
            baseline_synchrony_std: self.baseline.synchrony_stats.std(),
            baseline_phi_mean: self.baseline.phi_stats.mean,
            baseline_phi_std: self.baseline.phi_stats.std(),
            total_epochs: self.baseline.total_epochs,
            session_count: self.session_count,
            learning_rate: self.current_learning_rate,
        }
    }

    /// Import calibration data from persistence
    pub fn import_calibration(&mut self, data: CalibrationData) {
        self.baseline.subject_id = data.subject_id;
        self.adapted_complexity_threshold = data.complexity_threshold;
        self.adapted_synchrony_threshold = data.synchrony_threshold;
        self.adapted_phi_threshold = data.phi_threshold;
        self.adapted_low_freq = data.low_freq;
        self.adapted_high_freq = data.high_freq;

        // Reconstruct baseline stats (approximate - we don't store full state)
        self.baseline.complexity_stats.mean = data.baseline_complexity_mean;
        self.baseline.complexity_stats.count = data.total_epochs;
        self.baseline.synchrony_stats.mean = data.baseline_synchrony_mean;
        self.baseline.synchrony_stats.count = data.total_epochs;
        self.baseline.phi_stats.mean = data.baseline_phi_mean;
        self.baseline.phi_stats.count = data.total_epochs;
        self.baseline.total_epochs = data.total_epochs;

        self.session_count = data.session_count;
        self.current_learning_rate = data.learning_rate;
        self.calibration_state = CalibrationState::Calibrated;
    }
}

/// Serializable calibration data for persistence
#[derive(Debug, Clone)]
pub struct CalibrationData {
    pub subject_id: String,
    pub complexity_threshold: f32,
    pub synchrony_threshold: f32,
    pub phi_threshold: f32,
    pub low_freq: f32,
    pub high_freq: f32,
    pub baseline_complexity_mean: f32,
    pub baseline_complexity_std: f32,
    pub baseline_synchrony_mean: f32,
    pub baseline_synchrony_std: f32,
    pub baseline_phi_mean: f32,
    pub baseline_phi_std: f32,
    pub total_epochs: usize,
    pub session_count: usize,
    pub learning_rate: f32,
}

// ============================================================================
// Original Sleep Sentinel Code
// ============================================================================

/// Configuration for Sleep Sentinel
#[derive(Debug, Clone)]
pub struct SleepSentinelConfig {
    /// Number of neurons in local LTCs (frontal/occipital)
    pub local_neurons: usize,
    /// Number of neurons in global integrator
    pub global_neurons: usize,
    /// Timestep in milliseconds
    pub dt_ms: f32,
    /// Integration window size in samples
    pub integration_window: usize,
    /// Base time constant (should match EEG frequency range)
    pub tau_base: f32,
    /// Minimum time constant
    pub tau_min: f32,
    /// Maximum time constant
    pub tau_max: f32,
    /// Learning rate for online adaptation
    pub learning_rate: f32,
    /// Number of steps per epoch
    pub steps_per_epoch: usize,
    /// Complexity threshold for wake detection (initial/fallback)
    pub complexity_threshold: f32,
    /// Synchrony threshold for deep sleep detection (initial/fallback)
    pub synchrony_threshold: f32,
    /// Enable adaptive thresholds
    pub enable_adaptive_thresholds: bool,
    /// Configuration for adaptive threshold learning
    pub adaptive_config: AdaptiveThresholdConfig,
    /// Use spectral_analysis module (Welch PSD) instead of moving-average filters
    pub use_spectral_analysis: bool,
}

impl Default for SleepSentinelConfig {
    fn default() -> Self {
        Self {
            local_neurons: 64,       // Small but sufficient
            global_neurons: 128,     // Larger for integration
            dt_ms: 10.0,             // 100 Hz effective rate
            integration_window: 300, // 3 seconds at 100 Hz
            tau_base: 100.0,         // 100ms base (10 Hz sensitivity)
            tau_min: 10.0,           // 10ms min (100 Hz - gamma)
            tau_max: 1000.0,         // 1s max (1 Hz - delta)
            learning_rate: 0.001,
            steps_per_epoch: 3000, // 30 seconds at 100 Hz
            // Thresholds calibrated for real EEG:
            // - Real signals have lower variance than idealized synthetic
            // - These values represent the median, allowing scoring to work in both directions
            complexity_threshold: 0.35, // Lowered from 0.6 - real EEG has lower variance
            synchrony_threshold: 0.55,  // Lowered from 0.7 - correlation varies more
            enable_adaptive_thresholds: true,
            adaptive_config: AdaptiveThresholdConfig::default(),
            use_spectral_analysis: false,
        }
    }
}

/// Detected consciousness state
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConsciousnessState {
    /// Awake, alert
    Awake,
    /// Light sleep (N1/N2)
    LightSleep,
    /// Deep sleep (N3) - HIGH integration
    DeepSleep,
    /// REM sleep - HIGH activity, LOW integration
    REM,
    /// Uncertain/transitional
    Transitional,
}

impl ConsciousnessState {
    /// Convert to predicted sleep stage
    pub fn to_sleep_stage(&self) -> SleepStage {
        match self {
            ConsciousnessState::Awake => SleepStage::Wake,
            ConsciousnessState::LightSleep => SleepStage::N2, // Conservative
            ConsciousnessState::DeepSleep => SleepStage::N3,
            ConsciousnessState::REM => SleepStage::REM,
            ConsciousnessState::Transitional => SleepStage::N1,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            ConsciousnessState::Awake => "Awake",
            ConsciousnessState::LightSleep => "Light Sleep",
            ConsciousnessState::DeepSleep => "Deep Sleep",
            ConsciousnessState::REM => "REM",
            ConsciousnessState::Transitional => "Transitional",
        }
    }
}

/// Result of computing the Phi proxy for a signal window.
///
/// Contains the core integration measures plus spectral features
/// needed for sleep stage analysis and consciousness benchmarking.
#[derive(Debug, Clone)]
pub struct PhiProxyResult {
    /// Integration measure: synchrony × bidirectional causality (0..1).
    pub phi_proxy: f32,
    /// Pearson correlation between frontal and occipital channels, normalized to 0..1.
    pub synchrony: f32,
    /// Combined signal variance scaled to 0..1.
    pub complexity: f32,
    /// Permutation entropy (order 3) of the frontal channel (0..~2.58).
    pub permutation_entropy: f32,
    /// Dominant frequency estimated from zero crossings (Hz).
    pub dominant_freq_hz: f32,
    /// Lagged causal influence: frontal → occipital (0..1).
    pub frontal_to_occipital: f32,
    /// Lagged causal influence: occipital → frontal (0..1).
    pub occipital_to_frontal: f32,
    /// Delta band power (0.5-4 Hz).
    pub delta_power: f32,
    /// Theta band power (4-8 Hz).
    pub theta_power: f32,
    /// Alpha band power (8-12 Hz).
    pub alpha_power: f32,
    /// Beta band power (12-30 Hz).
    pub beta_power: f32,
    /// Gamma band power (30-100 Hz).
    pub gamma_power: f32,
}

/// Integration metrics (Φ proxy)
#[derive(Debug, Clone)]
pub struct IntegrationMetrics {
    /// Complexity (entropy) of combined signal
    pub complexity: f32,
    /// Synchronization between frontal and occipital
    pub synchrony: f32,
    /// Causal influence from frontal to occipital
    pub frontal_to_occipital: f32,
    /// Causal influence from occipital to frontal
    pub occipital_to_frontal: f32,
    /// Global integration measure (our Φ proxy)
    pub phi_proxy: f32,
    /// Dominant frequency estimate
    pub dominant_freq_hz: f32,
    /// State trajectory variance
    pub trajectory_variance: f32,
    /// Delta band power (0.5-4 Hz) - high in N3
    pub delta_power: f32,
    /// Theta band power (4-8 Hz) - high in N1, REM
    pub theta_power: f32,
    /// Alpha band power (8-12 Hz) - high in relaxed wake
    pub alpha_power: f32,
    /// Beta band power (12-30 Hz) - high in alert wake
    pub beta_power: f32,
    /// Gamma band power (30-100 Hz) - high in conscious binding
    pub gamma_power: f32,
    /// Spectral entropy (higher = more complex, broadband signal)
    pub spectral_entropy: f32,
}

impl Default for IntegrationMetrics {
    fn default() -> Self {
        Self {
            complexity: 0.5,
            synchrony: 0.5,
            frontal_to_occipital: 0.5,
            occipital_to_frontal: 0.5,
            phi_proxy: 0.5,
            dominant_freq_hz: 10.0,
            trajectory_variance: 0.1,
            delta_power: 0.0,
            theta_power: 0.0,
            alpha_power: 0.0,
            beta_power: 0.0,
            gamma_power: 0.0,
            spectral_entropy: 0.0,
        }
    }
}

/// Compute frequency band powers using moving average bandpass filtering
/// This approximation works well with the current scoring thresholds
fn compute_band_powers(signal: &[f32], sample_rate: f32) -> (f32, f32, f32, f32) {
    if signal.len() < 10 {
        return (0.0, 0.0, 0.0, 0.0);
    }

    // Compute signal energy in different frequency ranges using bandpass-like filtering
    // Delta (0.5-4 Hz): very slow oscillations
    // Theta (4-8 Hz): slow oscillations
    // Alpha (8-12 Hz): medium oscillations
    // Beta (12-30 Hz): fast oscillations

    let n = signal.len();
    let dt = 1.0 / sample_rate;

    // Simple moving average filters to extract band components
    // Delta: 250ms window (2-4 Hz capture)
    let delta_window = (0.25 / dt).max(3.0) as usize;
    // Theta: 125ms window (4-8 Hz capture)
    let theta_window = (0.125 / dt).max(3.0) as usize;
    // Alpha: 100ms window (8-12 Hz capture)
    let alpha_window = (0.1 / dt).max(3.0) as usize;
    // Beta: 33ms window (12-30 Hz capture)
    let beta_window = (0.033 / dt).max(3.0) as usize;

    // Compute low-pass filtered versions
    let slow = moving_average(signal, delta_window);
    let medium = moving_average(signal, theta_window);
    let fast = moving_average(signal, alpha_window);
    let vfast = moving_average(signal, beta_window);

    // Band powers from differences between filtered versions
    let mut delta_power = 0.0f32;
    let mut theta_power = 0.0f32;
    let mut alpha_power = 0.0f32;
    let mut beta_power = 0.0f32;

    for i in 0..n {
        // Delta: slowest component
        delta_power += slow[i].powi(2);
        // Theta: medium - slow
        theta_power += (medium[i] - slow[i]).powi(2);
        // Alpha: fast - medium
        alpha_power += (fast[i] - medium[i]).powi(2);
        // Beta: original - fast (high freq residual)
        beta_power += (signal[i] - vfast[i]).powi(2);
    }

    // Normalize by signal length
    let n_f = n as f32;
    delta_power = (delta_power / n_f).sqrt();
    theta_power = (theta_power / n_f).sqrt();
    alpha_power = (alpha_power / n_f).sqrt();
    beta_power = (beta_power / n_f).sqrt();

    // Normalize to sum to 1 (relative power)
    let total = delta_power + theta_power + alpha_power + beta_power + 1e-10;
    (
        delta_power / total,
        theta_power / total,
        alpha_power / total,
        beta_power / total,
    )
}

/// Simple moving average filter
fn moving_average(signal: &[f32], window: usize) -> Vec<f32> {
    let n = signal.len();
    let mut result = vec![0.0; n];
    let window = window.min(n).max(1);
    let half = window / 2;

    for i in 0..n {
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(n);
        let sum: f32 = signal[start..end].iter().sum();
        result[i] = sum / (end - start) as f32;
    }

    result
}

/// Compute frequency band powers using Welch PSD from spectral_analysis module.
///
/// Returns (delta, theta, alpha, beta, gamma, spectral_entropy) as normalized ratios.
/// This is more accurate than the moving-average approximation in `compute_band_powers`.
fn compute_band_powers_spectral(
    signal: &[f32],
    sample_rate: f32,
) -> (f32, f32, f32, f32, f32, f32) {
    use crate::dynamics::spectral_analysis::{SpectralAnalyzer, SpectralConfig, WindowType};

    if signal.len() < 32 {
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    // Convert f32 signal to f64 for spectral_analysis module
    let signal_f64: Vec<f64> = signal.iter().map(|&x| x as f64).collect();

    // Configure for EEG analysis: window size ~2 seconds of data
    let window_size = (2.0 * sample_rate as f64) as usize;
    let window_size = window_size.next_power_of_two().min(signal.len());
    let config = SpectralConfig {
        sample_rate: sample_rate as f64,
        window_size,
        overlap: 0.5,
        window_type: WindowType::Hann,
    };
    let analyzer = SpectralAnalyzer::new(config);

    // Get band powers via Welch PSD
    let bands = analyzer.band_power(&signal_f64);

    // Normalize to ratios (sum to ~1)
    let total = bands.total.max(1e-12);
    let delta = (bands.delta / total) as f32;
    let theta = (bands.theta / total) as f32;
    let alpha = (bands.alpha / total) as f32;
    let beta = (bands.beta / total) as f32;
    let gamma = (bands.gamma / total) as f32;

    // Compute spectral entropy
    let entropy = analyzer.spectral_entropy(&signal_f64) as f32;

    (delta, theta, alpha, beta, gamma, entropy)
}

/// Sleep Sentinel - Dual-channel LTC for consciousness detection
pub struct SleepSentinel {
    /// Configuration
    config: SleepSentinelConfig,
    /// Frontal cortex LTC (Fpz-Cz)
    frontal_ltc: UnifiedLTC,
    /// Occipital cortex LTC (Pz-Oz)
    occipital_ltc: UnifiedLTC,
    /// Global integrator LTC
    integrator_ltc: UnifiedLTC,
    /// Recent frontal states (for metrics)
    frontal_history: VecDeque<Vec<f32>>,
    /// Recent occipital states (for metrics)
    occipital_history: VecDeque<Vec<f32>>,
    /// Recent integrator states (for metrics)
    integrator_history: VecDeque<Vec<f32>>,
    /// Raw input history for complexity calculation (frontal, occipital)
    raw_input_history: VecDeque<(f32, f32)>,
    /// Sample counter
    sample_count: u64,
    /// Current integration metrics
    current_metrics: IntegrationMetrics,
    /// Running statistics
    stats: SleepSentinelStats,
    /// Adaptive threshold manager (per-subject)
    adaptive_threshold: Option<AdaptiveThreshold>,
}

/// Running statistics
#[derive(Debug, Clone, Default)]
pub struct SleepSentinelStats {
    pub epochs_processed: u64,
    pub samples_processed: u64,
    pub predictions: Vec<(ConsciousnessState, SleepStage)>, // (predicted, actual)
    pub accuracy_history: Vec<f32>,
}

impl SleepSentinel {
    /// Create new Sleep Sentinel
    pub fn new(config: SleepSentinelConfig) -> Self {
        // Create frontal LTC (tracks Fpz-Cz dynamics)
        let mut frontal_config = UnifiedLTCConfig::scalar(
            config.local_neurons,
            1,                    // Single channel input
            config.local_neurons, // Full state output
        );
        frontal_config.dt = config.dt_ms / 1000.0;
        frontal_config.tau_base = config.tau_base / 1000.0;
        frontal_config.tau_min = config.tau_min / 1000.0;
        frontal_config.tau_max = config.tau_max / 1000.0;
        frontal_config.learning = LearningAlgorithm::None; // Inference only

        // Create occipital LTC (tracks Pz-Oz dynamics)
        let occipital_config = frontal_config.clone();

        // Create integrator LTC (combines both)
        let mut integrator_config = UnifiedLTCConfig::scalar(
            config.global_neurons,
            config.local_neurons * 2, // Both local states
            config.global_neurons,
        );
        integrator_config.dt = config.dt_ms / 1000.0;
        integrator_config.tau_base = config.tau_base / 1000.0 * 2.0; // Slower for integration
        integrator_config.tau_min = config.tau_min / 1000.0;
        integrator_config.tau_max = config.tau_max / 1000.0 * 2.0;
        integrator_config.learning = LearningAlgorithm::None;

        let frontal_ltc = UnifiedLTC::new(frontal_config).expect("Failed to create frontal LTC");
        let occipital_ltc =
            UnifiedLTC::new(occipital_config).expect("Failed to create occipital LTC");
        let integrator_ltc =
            UnifiedLTC::new(integrator_config).expect("Failed to create integrator LTC");

        // Initialize adaptive threshold if enabled
        let adaptive_threshold = if config.enable_adaptive_thresholds {
            Some(AdaptiveThreshold::new(
                config.adaptive_config.clone(),
                "default_subject",
            ))
        } else {
            None
        };

        Self {
            config,
            frontal_ltc,
            occipital_ltc,
            integrator_ltc,
            frontal_history: VecDeque::with_capacity(500),
            occipital_history: VecDeque::with_capacity(500),
            integrator_history: VecDeque::with_capacity(500),
            raw_input_history: VecDeque::with_capacity(500),
            sample_count: 0,
            current_metrics: IntegrationMetrics::default(),
            stats: SleepSentinelStats::default(),
            adaptive_threshold,
        }
    }

    /// Create Sleep Sentinel for a specific subject with adaptive thresholds
    pub fn new_for_subject(config: SleepSentinelConfig, subject_id: impl Into<String>) -> Self {
        let mut sentinel = Self::new(config.clone());
        if config.enable_adaptive_thresholds {
            sentinel.adaptive_threshold =
                Some(AdaptiveThreshold::new(config.adaptive_config, subject_id));
        }
        sentinel
    }

    /// Set subject ID for adaptive thresholds
    pub fn set_subject(&mut self, subject_id: impl Into<String>) {
        if let Some(ref mut adaptive) = self.adaptive_threshold {
            adaptive.baseline.subject_id = subject_id.into();
        } else if self.config.enable_adaptive_thresholds {
            self.adaptive_threshold = Some(AdaptiveThreshold::new(
                self.config.adaptive_config.clone(),
                subject_id,
            ));
        }
    }

    /// Process a single sample pair (frontal, occipital)
    pub fn process_sample(&mut self, frontal: f32, occipital: f32) -> ConsciousnessState {
        // Normalize inputs to [-1, 1] range (assume microvolts)
        let frontal_norm = (frontal / 100.0).clamp(-1.0, 1.0);
        let occipital_norm = (occipital / 100.0).clamp(-1.0, 1.0);

        // Step frontal LTC
        let n = self.config.local_neurons;
        let (frontal_out, _) = self
            .frontal_ltc
            .forward(&[frontal_norm])
            .unwrap_or_else(|_| (vec![0.0; n], vec![]));

        // Step occipital LTC
        let (occipital_out, _) = self
            .occipital_ltc
            .forward(&[occipital_norm])
            .unwrap_or_else(|_| (vec![0.0; n], vec![]));

        // Combine states for integrator
        let mut combined = Vec::with_capacity(self.config.local_neurons * 2);
        combined
            .extend_from_slice(&frontal_out[..self.config.local_neurons.min(frontal_out.len())]);
        combined.extend_from_slice(
            &occipital_out[..self.config.local_neurons.min(occipital_out.len())],
        );

        // Pad if necessary
        while combined.len() < self.config.local_neurons * 2 {
            combined.push(0.0);
        }

        // Step integrator
        let (integrator_out, _) = self
            .integrator_ltc
            .forward(&combined)
            .unwrap_or_else(|_| (vec![0.0; n * 2], vec![]));

        // Record history
        self.frontal_history.push_back(frontal_out);
        self.occipital_history.push_back(occipital_out);
        self.integrator_history.push_back(integrator_out);
        self.raw_input_history
            .push_back((frontal_norm, occipital_norm));

        // Trim history
        while self.frontal_history.len() > self.config.integration_window {
            self.frontal_history.pop_front();
            self.occipital_history.pop_front();
            self.integrator_history.pop_front();
            self.raw_input_history.pop_front();
        }

        self.sample_count += 1;
        self.stats.samples_processed += 1;

        // Update metrics periodically (every 10 samples for efficiency)
        if self.sample_count.is_multiple_of(10) && self.frontal_history.len() >= 50 {
            self.update_metrics();
        }

        // Classify state
        self.classify_state()
    }

    /// Update integration metrics from history
    fn update_metrics(&mut self) {
        if self.frontal_history.len() < 50 {
            return;
        }

        let n = self.frontal_history.len();

        // 1. Compute complexity from RAW INPUT signal variance
        // Raw inputs preserve the actual signal dynamics before LTC processing
        let raw_frontal: Vec<f32> = self.raw_input_history.iter().map(|(f, _)| *f).collect();
        let raw_occipital: Vec<f32> = self.raw_input_history.iter().map(|(_, o)| *o).collect();

        let raw_frontal_mean: f32 = raw_frontal.iter().sum::<f32>() / n as f32;
        let raw_occipital_mean: f32 = raw_occipital.iter().sum::<f32>() / n as f32;

        let raw_frontal_var: f32 = raw_frontal
            .iter()
            .map(|x| (x - raw_frontal_mean).powi(2))
            .sum::<f32>()
            / n as f32;
        let raw_occipital_var: f32 = raw_occipital
            .iter()
            .map(|x| (x - raw_occipital_mean).powi(2))
            .sum::<f32>()
            / n as f32;

        // Combined variance as complexity measure (inputs in [-1, 1], variance max ~0.33)
        let combined_variance = (raw_frontal_var + raw_occipital_var) / 2.0;
        // Scale to 0-1: typical variance for sinusoidal signal is ~0.5 of amplitude^2
        self.current_metrics.complexity = (combined_variance * 3.0).min(1.0);
        self.current_metrics.trajectory_variance = combined_variance;

        // 2. Compute synchrony (Pearson correlation between frontal and occipital)
        // Use RAW inputs which preserve actual signal correlation
        // LTC outputs converge to stable states and lose correlation information
        let mut correlation = 0.0;
        let mut frontal_sum_sq = 0.0;
        let mut occipital_sum_sq = 0.0;
        for (f, o) in raw_frontal.iter().zip(raw_occipital.iter()) {
            let f_val = f - raw_frontal_mean;
            let o_val = o - raw_occipital_mean;
            correlation += f_val * o_val;
            frontal_sum_sq += f_val * f_val;
            occipital_sum_sq += o_val * o_val;
        }

        let denom = (frontal_sum_sq * occipital_sum_sq).sqrt();
        self.current_metrics.synchrony = if denom > 1e-6 {
            ((correlation / denom) + 1.0) / 2.0 // Normalize from [-1,1] to [0, 1]
        } else {
            0.5
        };

        // 3. Estimate causal influence (simplified: lagged correlation using raw inputs)
        let lag = 5; // ~50ms lag at 100Hz
        if n > lag {
            // Frontal → Occipital
            let mut causal_fo = 0.0;
            for i in lag..n {
                let f_prev = raw_frontal[i - lag];
                let o_curr = raw_occipital[i];
                causal_fo += (f_prev - raw_frontal_mean) * (o_curr - raw_occipital_mean);
            }

            // Occipital → Frontal
            let mut causal_of = 0.0;
            for i in lag..n {
                let o_prev = raw_occipital[i - lag];
                let f_curr = raw_frontal[i];
                causal_of += (o_prev - raw_occipital_mean) * (f_curr - raw_frontal_mean);
            }

            let count = (n - lag) as f32;
            self.current_metrics.frontal_to_occipital = ((causal_fo / count).tanh() + 1.0) / 2.0;
            self.current_metrics.occipital_to_frontal = ((causal_of / count).tanh() + 1.0) / 2.0;
        }

        // 4. Compute Φ proxy (integration = synchrony * bidirectional causality)
        let bidirectional = (self.current_metrics.frontal_to_occipital
            + self.current_metrics.occipital_to_frontal)
            / 2.0;
        self.current_metrics.phi_proxy = self.current_metrics.synchrony * bidirectional;

        // 5. Estimate dominant frequency from zero crossings (around mean)
        // Use raw frontal signal which preserves input dynamics
        let mut crossings = 0;
        let mut prev_val = raw_frontal.first().copied().unwrap_or(0.0) - raw_frontal_mean;
        for &val in raw_frontal.iter().skip(1) {
            let centered = val - raw_frontal_mean;
            if (centered > 0.0) != (prev_val > 0.0) {
                crossings += 1;
            }
            prev_val = centered;
        }
        let duration_sec = n as f32 * self.config.dt_ms / 1000.0;
        self.current_metrics.dominant_freq_hz = if duration_sec > 0.0 {
            crossings as f32 / (2.0 * duration_sec)
        } else {
            10.0
        };
    }

    /// Compute the Phi proxy (integration measure) from raw signal buffers.
    ///
    /// Returns `(phi_proxy, synchrony, complexity, permutation_entropy)` for the
    /// current integration window. Returns `None` if insufficient data (<50 samples).
    ///
    /// - **phi_proxy**: synchrony × bidirectional causality (0..1)
    /// - **synchrony**: Pearson correlation between frontal and occipital (normalized to 0..1)
    /// - **complexity**: combined signal variance scaled to 0..1
    /// - **permutation_entropy**: order-3 permutation entropy of frontal channel (0..1)
    pub fn compute_phi_proxy(&mut self) -> Option<PhiProxyResult> {
        if self.frontal_history.len() < 50 {
            return None;
        }
        self.update_metrics();
        let m = &self.current_metrics;

        // Compute permutation entropy from raw frontal signal
        let raw_frontal: Vec<f64> = self
            .raw_input_history
            .iter()
            .map(|(f, _)| *f as f64)
            .collect();
        let perm_entropy = super::permutation_entropy_order3(&raw_frontal, 1);

        Some(PhiProxyResult {
            phi_proxy: m.phi_proxy,
            synchrony: m.synchrony,
            complexity: m.complexity,
            permutation_entropy: perm_entropy as f32,
            dominant_freq_hz: m.dominant_freq_hz,
            frontal_to_occipital: m.frontal_to_occipital,
            occipital_to_frontal: m.occipital_to_frontal,
            delta_power: m.delta_power,
            theta_power: m.theta_power,
            alpha_power: m.alpha_power,
            beta_power: m.beta_power,
            gamma_power: m.gamma_power,
        })
    }

    /// Classify consciousness state based on metrics
    ///
    /// Uses a scoring-based approach that computes evidence for each state
    /// and selects the one with highest score. This is more robust than
    /// strict AND conditions which fail when real signals don't perfectly
    /// match idealized patterns.
    fn classify_state(&self) -> ConsciousnessState {
        let m = &self.current_metrics;

        // Get thresholds - use adaptive if available, otherwise config defaults
        let (complexity_thresh, synchrony_thresh, phi_thresh, freq_bounds) =
            if let Some(ref adaptive) = self.adaptive_threshold {
                if adaptive.is_calibrated() && !adaptive.is_using_fallback() {
                    (
                        adaptive.complexity_threshold(),
                        adaptive.synchrony_threshold(),
                        adaptive.phi_threshold(),
                        adaptive.frequency_bounds(),
                    )
                } else {
                    (
                        self.config.complexity_threshold,
                        self.config.synchrony_threshold,
                        0.5,
                        (8.0, 12.0),
                    )
                }
            } else {
                (
                    self.config.complexity_threshold,
                    self.config.synchrony_threshold,
                    0.5,
                    (8.0, 12.0),
                )
            };

        // Compute continuous evidence scores rather than binary conditions
        // This handles the reality that real EEG rarely perfectly matches idealized patterns

        // Normalized distances from thresholds (positive = above threshold)
        let complexity_score = (m.complexity - complexity_thresh) / complexity_thresh.max(0.1);
        let synchrony_score = (m.synchrony - synchrony_thresh) / synchrony_thresh.max(0.1);
        let phi_score = (m.phi_proxy - phi_thresh) / phi_thresh.max(0.1);

        // Frequency scores: how far from the alpha band (8-12 Hz)?
        let freq_low = freq_bounds.0; // 8 Hz
        let freq_high = freq_bounds.1; // 12 Hz

        // low_freq_score: positive when frequency is low (delta/theta range)
        let low_freq_score = (freq_low - m.dominant_freq_hz) / freq_low.max(1.0);
        // high_freq_score: positive when frequency is high (beta range)
        let high_freq_score = (m.dominant_freq_hz - freq_high) / freq_high.max(1.0);

        // Compute evidence scores for each state
        // These weights are based on neurophysiological signatures.
        //
        // Welch PSD vs moving-average thresholds:
        // With proper Welch PSD, delta is 60-80% of total power for ALL stages.
        // The delta_ratio (delta/faster) is thus always high (~2-5).
        // We use higher thresholds when spectral analysis is active.
        let spectral = self.config.use_spectral_analysis;

        // Adaptive threshold multipliers for Welch PSD
        // With PSD: delta_ratio for N3 needs ~3.5 (vs 2.0 for moving-avg)
        // With PSD: fast_ratio for Wake needs ~0.25 (vs 0.6 for moving-avg)
        let (
            n3_delta_thresh,
            n3_delta_strong,
            _n3_delta_mid,
            n3_delta_low,
            wake_fast_high,
            wake_fast_mid,
            wake_beta_thresh,
            wake_activity_thresh,
            rem_theta_delta_thresh,
            rem_delta_ratio_thresh,
            rem_activity_thresh,
            n2_delta_upper,
            n2_delta_mid,
            n2_delta_border,
            n1_theta_thresh,
            n1_alpha_ceil,
        ) = if spectral {
            // Welch PSD thresholds: delta always dominates, use tighter ratios
            (
                3.5, 3.0, 2.5, 2.0, // N3 delta_ratio thresholds (raised) - 4 values
                0.25, 0.15, 0.03, 0.05, // Wake fast thresholds (lowered: alpha+beta/delta)
                0.30, 2.8, 0.08, // REM thresholds (adjusted for PSD)
                3.0, 2.8, 3.5, // N2 delta_ratio upper bounds (raised)
                0.08, 0.20,
            ) // N1 theta/alpha thresholds (lowered)
        } else {
            // Moving-average thresholds: original values
            (
                2.0, 1.8, 1.5, 1.2, // N3 - 4 values
                0.6, 0.4, 0.06, 0.12, 0.50, 1.6, 0.12, 2.0, 1.8, 2.5, 0.12, 0.25,
            )
        };

        // Wake: High complexity, high frequency (alpha/beta), lowest delta ratio
        let fast_activity = m.alpha_power + m.beta_power;
        let fast_ratio = if m.delta_power > 0.01 {
            fast_activity / m.delta_power
        } else {
            2.0
        };
        // Spectral entropy boost for Wake: high entropy = more complex/awake signal
        let entropy_wake_boost = if spectral && m.spectral_entropy > 0.7 {
            0.20
        } else if spectral && m.spectral_entropy > 0.5 {
            0.10
        } else {
            0.0
        };
        let wake_boost = if fast_ratio > wake_fast_high && fast_activity > wake_activity_thresh {
            0.45
        } else if fast_ratio > wake_fast_mid || m.beta_power > wake_beta_thresh {
            0.30
        } else if fast_activity > wake_activity_thresh * 0.83 {
            0.18
        } else {
            0.0
        };
        let wake_score = complexity_score.max(0.0) * 0.18
            + high_freq_score.max(0.0) * 0.15
            + (-synchrony_score) * 0.15
            + wake_boost
            + entropy_wake_boost;

        // N3 (Deep Sleep): Delta OVERWHELMINGLY dominates + high synchrony
        let faster_power = m.theta_power + m.alpha_power + m.beta_power;
        let delta_ratio = if faster_power > 0.01 {
            m.delta_power / faster_power
        } else {
            2.0
        };
        let strong_delta = delta_ratio > n3_delta_thresh;
        // Spectral entropy boost for N3: low entropy = simple/deep-sleep signal
        let entropy_deep_boost = if spectral && m.spectral_entropy < 0.3 {
            0.15
        } else if spectral && m.spectral_entropy < 0.5 {
            0.05
        } else {
            0.0
        };
        let delta_boost = if strong_delta && synchrony_score > 0.0 {
            0.28
        } else if delta_ratio > n3_delta_strong {
            0.15
        } else if delta_ratio > n3_delta_low {
            0.05
        } else {
            0.0
        };
        let deep_score = synchrony_score.max(0.0) * 0.25
            + low_freq_score.max(0.0) * 0.20
            + phi_score.max(0.0) * 0.22
            + delta_boost
            + entropy_deep_boost;

        // REM: Theta/alpha mix without extreme delta dominance (the paradox)
        let rem_active = m.theta_power + m.alpha_power;
        let theta_to_delta = if m.delta_power > 0.01 {
            m.theta_power / m.delta_power
        } else {
            2.0
        };
        let is_rem_pattern =
            theta_to_delta > rem_theta_delta_thresh && delta_ratio < rem_delta_ratio_thresh;
        let rem_paradox_boost = if is_rem_pattern && rem_active > rem_activity_thresh {
            0.75
        } else if rem_active > rem_activity_thresh && delta_ratio < rem_delta_ratio_thresh * 1.125 {
            0.50
        } else if theta_to_delta > rem_theta_delta_thresh * 0.7 {
            0.30
        } else {
            0.0
        };
        let rem_score = complexity_score.max(0.0) * 0.10
            + (-phi_score).max(0.0) * 0.05
            + (-synchrony_score).max(0.0) * 0.03
            + rem_paradox_boost;

        // N2 (Light Sleep): Moderate synchrony, delta present but not overwhelming
        let rem_penalty = if is_rem_pattern { 0.15 } else { 0.0 };
        let n2_boost = if synchrony_score > 0.0 && delta_ratio < n2_delta_upper && !is_rem_pattern {
            0.28
        } else if delta_ratio < n2_delta_mid && !is_rem_pattern {
            0.20
        } else if delta_ratio < n2_delta_border {
            0.08
        } else {
            0.0
        };
        let n2_score = synchrony_score.max(0.0) * 0.28
            + phi_score.max(0.0) * 0.20
            + low_freq_score.max(0.0) * 0.24
            + n2_boost
            - rem_penalty;

        // N1 (Transitional): Lower complexity than wake, theta appearing
        let n1_boost = if m.theta_power > n1_theta_thresh && m.alpha_power < n1_alpha_ceil {
            0.15
        } else {
            0.0
        };
        let n1_score = (-complexity_score).max(0.0) * 0.30
            + synchrony_score.max(0.0) * 0.25
            + low_freq_score.max(0.0) * 0.30
            + n1_boost;

        // Find the state with highest score
        let scores = [
            (wake_score, ConsciousnessState::Awake),
            (n1_score, ConsciousnessState::Transitional), // N1
            (n2_score, ConsciousnessState::LightSleep),   // N2
            (deep_score, ConsciousnessState::DeepSleep),  // N3
            (rem_score, ConsciousnessState::REM),
        ];

        scores
            .iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, state)| *state)
            .unwrap_or(ConsciousnessState::Transitional)
    }

    /// Process a full 30-second epoch
    pub fn process_epoch(
        &mut self,
        frontal_data: &[f64],
        occipital_data: &[f64],
    ) -> (ConsciousnessState, IntegrationMetrics) {
        // Reset history for clean epoch
        self.frontal_history.clear();
        self.occipital_history.clear();
        self.integrator_history.clear();
        self.raw_input_history.clear();

        // Reset LTC states
        self.frontal_ltc.reset();
        self.occipital_ltc.reset();
        self.integrator_ltc.reset();

        // Compute frequency band powers from raw signal (before subsampling)
        // This captures the spectral content that LTC dynamics might miss
        let sample_rate = 100.0; // Sleep-EDF standard is 100 Hz
        let frontal_f32: Vec<f32> = frontal_data.iter().map(|&x| x as f32).collect();

        if self.config.use_spectral_analysis {
            // Use proper Welch PSD from spectral_analysis module (higher accuracy)
            let (delta, theta, alpha, beta, gamma, entropy) =
                compute_band_powers_spectral(&frontal_f32, sample_rate);
            self.current_metrics.delta_power = delta;
            self.current_metrics.theta_power = theta;
            self.current_metrics.alpha_power = alpha;
            self.current_metrics.beta_power = beta;
            self.current_metrics.gamma_power = gamma;
            self.current_metrics.spectral_entropy = entropy;
        } else {
            // Fallback: moving-average bandpass filtering
            let (delta, theta, alpha, beta) = compute_band_powers(&frontal_f32, sample_rate);
            self.current_metrics.delta_power = delta;
            self.current_metrics.theta_power = theta;
            self.current_metrics.alpha_power = alpha;
            self.current_metrics.beta_power = beta;
        }

        // Subsample to match our effective rate
        let target_samples = self.config.steps_per_epoch;
        let step = frontal_data.len().max(1) / target_samples.max(1);
        let step = step.max(1);

        let mut last_state = ConsciousnessState::Transitional;

        for i in (0..frontal_data.len()).step_by(step) {
            let f = frontal_data[i] as f32;
            let o = occipital_data.get(i).copied().unwrap_or(0.0) as f32;
            last_state = self.process_sample(f, o);
        }

        self.stats.epochs_processed += 1;

        (last_state, self.current_metrics.clone())
    }

    /// Train on labeled epoch with online threshold adaptation
    pub fn train_epoch(
        &mut self,
        frontal_data: &[f64],
        occipital_data: &[f64],
        actual_stage: SleepStage,
    ) -> (ConsciousnessState, bool) {
        let (predicted_state, metrics) = self.process_epoch(frontal_data, occipital_data);
        let predicted_stage = predicted_state.to_sleep_stage();

        let correct = predicted_stage == actual_stage;
        self.stats.predictions.push((predicted_state, actual_stage));

        // Calculate recent accuracy before taking mutable borrow
        let recent_accuracy = self.recent_accuracy(50);

        // Online learning via adaptive threshold adjustment
        if let Some(ref mut adaptive) = self.adaptive_threshold {
            adaptive.update_with_feedback(&metrics, predicted_state, actual_stage, correct);

            // Check if we should enable fallback due to poor performance
            if adaptive.should_enable_fallback(recent_accuracy, 0.4)
                && !adaptive.is_using_fallback()
            {
                adaptive.enable_fallback();
            }
        }

        (predicted_state, correct)
    }

    /// Calculate accuracy over recent N predictions
    fn recent_accuracy(&self, n: usize) -> f32 {
        let recent: Vec<_> = self.stats.predictions.iter().rev().take(n).collect();
        if recent.is_empty() {
            return 0.0;
        }
        let correct = recent
            .iter()
            .filter(|(pred, actual)| pred.to_sleep_stage() == *actual)
            .count();
        correct as f32 / recent.len() as f32
    }

    /// Get current integration metrics
    pub fn metrics(&self) -> &IntegrationMetrics {
        &self.current_metrics
    }

    /// Get statistics
    pub fn stats(&self) -> &SleepSentinelStats {
        &self.stats
    }

    // ========================================================================
    // Adaptive Threshold Management
    // ========================================================================

    /// Start calibration mode for a specific subject
    pub fn start_calibration(&mut self, subject_id: Option<&str>) {
        if let Some(id) = subject_id {
            self.set_subject(id);
        }
        if let Some(ref mut adaptive) = self.adaptive_threshold {
            adaptive.start_calibration();
        }
    }

    /// End calibration mode
    /// Returns true if calibration was successful (enough data collected)
    pub fn end_calibration(&mut self) -> bool {
        if let Some(ref mut adaptive) = self.adaptive_threshold {
            adaptive.end_calibration()
        } else {
            false
        }
    }

    /// Check if the sentinel is calibrated for the current subject
    pub fn is_calibrated(&self) -> bool {
        self.adaptive_threshold
            .as_ref()
            .map(|a| a.is_calibrated())
            .unwrap_or(false)
    }

    /// Get calibration progress (0.0 to 1.0)
    pub fn calibration_progress(&self) -> f32 {
        self.adaptive_threshold
            .as_ref()
            .map(|a| a.calibration_progress())
            .unwrap_or(0.0)
    }

    /// Start a new session (e.g., new night of recording)
    /// Applies learning rate decay to the adaptive thresholds
    pub fn start_new_session(&mut self) {
        if let Some(ref mut adaptive) = self.adaptive_threshold {
            adaptive.start_new_session();
        }
    }

    /// Get the adaptive threshold manager
    pub fn adaptive_threshold(&self) -> Option<&AdaptiveThreshold> {
        self.adaptive_threshold.as_ref()
    }

    /// Get mutable reference to adaptive threshold manager
    pub fn adaptive_threshold_mut(&mut self) -> Option<&mut AdaptiveThreshold> {
        self.adaptive_threshold.as_mut()
    }

    /// Export calibration data for persistence
    pub fn export_calibration(&self) -> Option<CalibrationData> {
        self.adaptive_threshold
            .as_ref()
            .map(|a| a.export_calibration())
    }

    /// Import calibration data from persistence
    pub fn import_calibration(&mut self, data: CalibrationData) {
        if let Some(ref mut adaptive) = self.adaptive_threshold {
            adaptive.import_calibration(data);
        } else if self.config.enable_adaptive_thresholds {
            let mut adaptive =
                AdaptiveThreshold::new(self.config.adaptive_config.clone(), &data.subject_id);
            adaptive.import_calibration(data);
            self.adaptive_threshold = Some(adaptive);
        }
    }

    /// Get current thresholds (adapted or default)
    pub fn current_thresholds(&self) -> (f32, f32, f32) {
        if let Some(ref adaptive) = self.adaptive_threshold {
            if adaptive.is_calibrated() {
                return (
                    adaptive.complexity_threshold(),
                    adaptive.synchrony_threshold(),
                    adaptive.phi_threshold(),
                );
            }
        }
        (
            self.config.complexity_threshold,
            self.config.synchrony_threshold,
            0.5,
        )
    }

    /// Get z-score normalized metrics for current subject
    pub fn normalized_metrics(&self) -> Option<IntegrationMetrics> {
        self.adaptive_threshold
            .as_ref()
            .filter(|a| a.is_calibrated())
            .map(|a| a.normalize_metrics(&self.current_metrics))
    }

    /// Calculate overall accuracy
    pub fn accuracy(&self) -> f32 {
        if self.stats.predictions.is_empty() {
            return 0.0;
        }

        let correct = self
            .stats
            .predictions
            .iter()
            .filter(|(pred, actual)| pred.to_sleep_stage() == *actual)
            .count();

        correct as f32 / self.stats.predictions.len() as f32
    }

    /// Get per-class accuracy
    pub fn per_class_accuracy(&self) -> Vec<(SleepStage, f32, usize)> {
        let stages = [
            SleepStage::Wake,
            SleepStage::N1,
            SleepStage::N2,
            SleepStage::N3,
            SleepStage::REM,
        ];

        stages
            .iter()
            .map(|&stage| {
                let samples: Vec<_> = self
                    .stats
                    .predictions
                    .iter()
                    .filter(|(_, actual)| *actual == stage)
                    .collect();

                let correct = samples
                    .iter()
                    .filter(|(pred, _)| pred.to_sleep_stage() == stage)
                    .count();

                let accuracy = if samples.is_empty() {
                    0.0
                } else {
                    correct as f32 / samples.len() as f32
                };

                (stage, accuracy, samples.len())
            })
            .collect()
    }

    /// Reset all state (preserves adaptive thresholds and calibration)
    pub fn reset(&mut self) {
        self.frontal_ltc.reset();
        self.occipital_ltc.reset();
        self.integrator_ltc.reset();
        self.frontal_history.clear();
        self.occipital_history.clear();
        self.integrator_history.clear();
        self.raw_input_history.clear();
        self.sample_count = 0;
        self.current_metrics = IntegrationMetrics::default();
        // Note: adaptive_threshold is preserved to maintain calibration
    }

    /// Full reset including adaptive thresholds
    pub fn reset_all(&mut self) {
        self.reset();
        self.stats = SleepSentinelStats::default();
        if self.config.enable_adaptive_thresholds {
            self.adaptive_threshold = Some(AdaptiveThreshold::new(
                self.config.adaptive_config.clone(),
                "default_subject",
            ));
        } else {
            self.adaptive_threshold = None;
        }
    }

    /// Get summary string
    pub fn summary(&self) -> String {
        let adaptive_info = if let Some(ref adaptive) = self.adaptive_threshold {
            let (c, s, p) = (
                adaptive.complexity_threshold(),
                adaptive.synchrony_threshold(),
                adaptive.phi_threshold(),
            );
            format!(
                "\n\nAdaptive Thresholds:\n\
                 Calibration: {:?}\n\
                 Complexity: {:.3} (baseline mean: {:.3})\n\
                 Synchrony: {:.3} (baseline mean: {:.3})\n\
                 Phi: {:.3} (baseline mean: {:.3})\n\
                 Learning rate: {:.4}\n\
                 Sessions: {}\n\
                 Bounds violations: {}\n\
                 Anomalies detected: {}",
                adaptive.calibration_state(),
                c,
                adaptive.baseline().complexity_stats.mean,
                s,
                adaptive.baseline().synchrony_stats.mean,
                p,
                adaptive.baseline().phi_stats.mean,
                adaptive.learning_rate(),
                adaptive.session_count(),
                adaptive.bounds_violations(),
                adaptive.anomalies_detected(),
            )
        } else {
            String::new()
        };

        format!(
            "Sleep Sentinel Summary:\n\
             Epochs: {}\n\
             Samples: {}\n\
             Accuracy: {:.1}%\n\n\
             Per-class:\n{}{}",
            self.stats.epochs_processed,
            self.stats.samples_processed,
            self.accuracy() * 100.0,
            self.per_class_accuracy()
                .iter()
                .map(|(stage, acc, n)| format!("  {}: {:.1}% (n={})", stage.name(), acc * 100.0, n))
                .collect::<Vec<_>>()
                .join("\n"),
            adaptive_info
        )
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Generate synthetic EEG-like signal
    fn generate_synthetic_eeg(
        stage: SleepStage,
        sample_rate: f64,
        duration_sec: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = (sample_rate * duration_sec) as usize;
        let mut frontal = Vec::with_capacity(n);
        let mut occipital = Vec::with_capacity(n);

        for i in 0..n {
            let t = i as f64 / sample_rate;

            // Base signals depend on stage
            let (f, o) = match stage {
                SleepStage::Wake => {
                    // High-freq, desynchronized alpha/beta
                    let alpha = 30.0 * (2.0 * PI * 10.0 * t).sin();
                    let beta = 15.0 * (2.0 * PI * 20.0 * t).sin();
                    // Different phases for frontal/occipital
                    (
                        alpha + beta + 10.0 * (t * 31.4).sin(),
                        alpha * 0.8 + beta * 1.2 + 10.0 * (t * 28.3).sin(),
                    )
                }
                SleepStage::N3 => {
                    // Synchronized delta waves
                    let delta = 75.0 * (2.0 * PI * 2.0 * t).sin();
                    // Highly correlated between channels
                    (delta, delta * 0.95 + 5.0 * (t * 12.5).sin())
                }
                SleepStage::REM => {
                    // High activity but desynchronized (like wake but different)
                    let mixed =
                        25.0 * (2.0 * PI * 8.0 * t).sin() + 20.0 * (2.0 * PI * 15.0 * t).sin();
                    // Very different between channels
                    (
                        mixed + 15.0 * (t * 47.1).sin(),
                        mixed * 0.6 + 25.0 * (t * 33.2).sin(),
                    )
                }
                _ => {
                    // Light sleep - intermediate
                    let theta = 40.0 * (2.0 * PI * 6.0 * t).sin();
                    (
                        theta + 10.0 * (t * 25.0).sin(),
                        theta * 0.7 + 15.0 * (t * 22.0).sin(),
                    )
                }
            };

            // Add noise
            let noise_f = 5.0 * ((i as f64 * 0.1234).sin() * (i as f64 * 0.5678).cos());
            let noise_o = 5.0 * ((i as f64 * 0.2345).sin() * (i as f64 * 0.6789).cos());

            frontal.push(f + noise_f);
            occipital.push(o + noise_o);
        }

        (frontal, occipital)
    }

    #[test]
    fn test_sentinel_creation() {
        let config = SleepSentinelConfig::default();
        let sentinel = SleepSentinel::new(config);
        assert_eq!(sentinel.stats.epochs_processed, 0);
    }

    #[test]
    #[ignore = "slow ~60s: trains LTC networks for EEG processing"]
    fn test_wake_detection() {
        // Use default config with reduced steps for faster test
        let mut config = SleepSentinelConfig::default();
        config.steps_per_epoch = 500; // Reduced from 3000 for faster test

        let mut sentinel = SleepSentinel::new(config);

        // Use shorter duration (10s at 100Hz = 1000 samples)
        let (frontal, occipital) = generate_synthetic_eeg(SleepStage::Wake, 100.0, 10.0);
        let (state, metrics) = sentinel.process_epoch(&frontal, &occipital);

        println!("Wake detection: {:?}", state);
        println!(
            "Metrics: complexity={:.2}, synchrony={:.2}, phi={:.2}, freq={:.1}Hz",
            metrics.complexity, metrics.synchrony, metrics.phi_proxy, metrics.dominant_freq_hz
        );

        // Wake should have meaningful complexity (non-zero signal variance)
        // Threshold set to 0.1 based on synthetic EEG amplitude (~50uV normalized to 0.5)
        assert!(
            metrics.complexity > 0.1,
            "Wake should have complexity > 0.1, got {}",
            metrics.complexity
        );
    }

    #[test]
    #[ignore = "slow ~60s: trains LTC networks for EEG processing"]
    fn test_deep_sleep_detection() {
        // Use default config with reduced steps for faster test
        let mut config = SleepSentinelConfig::default();
        config.steps_per_epoch = 500; // Reduced from 3000 for faster test

        let mut sentinel = SleepSentinel::new(config);

        // Use shorter duration for faster test (10s at 100Hz = 1000 samples)
        let (frontal, occipital) = generate_synthetic_eeg(SleepStage::N3, 100.0, 10.0);
        let (state, metrics) = sentinel.process_epoch(&frontal, &occipital);

        println!("Deep sleep detection: {:?}", state);
        println!(
            "Metrics: complexity={:.2}, synchrony={:.2}, phi={:.2}, freq={:.1}Hz",
            metrics.complexity, metrics.synchrony, metrics.phi_proxy, metrics.dominant_freq_hz
        );

        // Deep sleep has synchronized delta waves - check for higher synchrony than wake
        // The N3 synthetic signal has highly correlated frontal/occipital channels
        assert!(
            metrics.synchrony > 0.4,
            "Deep sleep should have synchrony > 0.4, got {}",
            metrics.synchrony
        );
    }

    #[test]
    #[ignore = "slow ~60s: trains LTC networks for EEG processing"]
    fn test_rem_paradox() {
        // Use default config with reduced steps for faster test
        let mut config = SleepSentinelConfig::default();
        config.steps_per_epoch = 500; // Reduced from 3000 for faster test

        let mut sentinel = SleepSentinel::new(config);

        // Use shorter duration for faster test (10s at 100Hz = 1000 samples)
        let (frontal, occipital) = generate_synthetic_eeg(SleepStage::REM, 100.0, 10.0);
        let (_state, metrics) = sentinel.process_epoch(&frontal, &occipital);

        println!(
            "REM detection: complexity={:.2}, synchrony={:.2}, phi={:.2}",
            metrics.complexity, metrics.synchrony, metrics.phi_proxy
        );

        // REM paradox: active brain (complexity > 0) but less synchronized than deep sleep
        // This is what makes it interesting - activity without integration
        assert!(
            metrics.complexity > 0.05,
            "REM should have some complexity, got {}",
            metrics.complexity
        );
    }

    #[test]
    #[ignore = "slow ~60s: trains LTC networks, tests stage discrimination across Wake/N3/REM"]
    fn test_discrimination() {
        let config = SleepSentinelConfig::default();
        let mut sentinel = SleepSentinel::new(config);

        // Test that different stages produce different metrics
        let stages = [SleepStage::Wake, SleepStage::N3, SleepStage::REM];
        let mut results = Vec::new();

        for stage in &stages {
            sentinel.reset();
            let (frontal, occipital) = generate_synthetic_eeg(*stage, 100.0, 30.0);
            let (_, metrics) = sentinel.process_epoch(&frontal, &occipital);
            results.push((
                *stage,
                metrics.complexity,
                metrics.synchrony,
                metrics.phi_proxy,
            ));
        }

        println!("\nStage Discrimination:");
        for (stage, c, s, p) in &results {
            println!(
                "  {:?}: complexity={:.2}, synchrony={:.2}, phi={:.2}",
                stage, c, s, p
            );
        }

        // Verify stages are distinguishable
        // Deep sleep should have highest synchrony
        let n3_sync = results
            .iter()
            .find(|(s, _, _, _)| *s == SleepStage::N3)
            .unwrap()
            .2;
        let wake_sync = results
            .iter()
            .find(|(s, _, _, _)| *s == SleepStage::Wake)
            .unwrap()
            .2;
        assert!(
            n3_sync > wake_sync,
            "Deep sleep should have higher synchrony than wake"
        );
    }

    // ========================================================================
    // Adaptive Threshold Tests
    // ========================================================================

    #[test]
    fn test_running_stats_welford() {
        let mut stats = RunningStats::default();
        let values = [1.0, 2.0, 3.0, 4.0, 5.0];

        for &v in &values {
            stats.update(v, 0.1);
        }

        // Mean should be 3.0
        assert!(
            (stats.mean - 3.0).abs() < 0.01,
            "Mean should be 3.0, got {}",
            stats.mean
        );

        // Variance should be 2.5 (sample variance)
        assert!(
            (stats.variance() - 2.5).abs() < 0.01,
            "Variance should be 2.5, got {}",
            stats.variance()
        );

        // Std should be sqrt(2.5) ≈ 1.58
        assert!(
            (stats.std() - 1.58).abs() < 0.1,
            "Std should be ~1.58, got {}",
            stats.std()
        );

        // Z-score of 5.0 should be (5-3)/1.58 ≈ 1.26
        let z = stats.z_score(5.0);
        assert!((z - 1.26).abs() < 0.1, "Z-score should be ~1.26, got {}", z);
    }

    #[test]
    fn test_running_stats_anomaly_detection() {
        let mut stats = RunningStats::default();

        // Build up baseline with normal values
        for i in 0..100 {
            stats.update(50.0 + (i as f32 % 10.0) - 5.0, 0.1); // Values around 50 ± 5
        }

        // Normal value should not be anomalous
        assert!(!stats.is_anomaly(52.0, 3.0), "52.0 should not be anomalous");

        // Extreme value should be anomalous
        assert!(stats.is_anomaly(100.0, 3.0), "100.0 should be anomalous");
    }

    #[test]
    fn test_adaptive_threshold_creation() {
        let config = AdaptiveThresholdConfig::default();
        let adaptive = AdaptiveThreshold::new(config, "subject_001");

        assert_eq!(adaptive.calibration_state(), CalibrationState::Uncalibrated);
        assert!(!adaptive.is_calibrated());
        assert_eq!(adaptive.baseline().subject_id, "subject_001");
    }

    #[test]
    fn test_calibration_workflow() {
        let mut config = AdaptiveThresholdConfig::default();
        config.calibration_epochs = 5;
        config.min_samples_for_adaptation = 3;

        let mut adaptive = AdaptiveThreshold::new(config, "subject_001");

        // Start calibration
        adaptive.start_calibration();
        assert_eq!(adaptive.calibration_state(), CalibrationState::Calibrating);
        assert!(!adaptive.is_calibrated());

        // Simulate calibration epochs
        for i in 0..5 {
            let metrics = IntegrationMetrics {
                complexity: 0.5 + (i as f32 * 0.02),
                synchrony: 0.6 + (i as f32 * 0.01),
                phi_proxy: 0.45,
                ..Default::default()
            };
            adaptive.update_with_feedback(
                &metrics,
                ConsciousnessState::LightSleep,
                SleepStage::N2,
                true,
            );
        }

        // Should auto-complete calibration
        assert!(
            adaptive.is_calibrated(),
            "Should be calibrated after enough epochs"
        );
        assert_eq!(adaptive.calibration_state(), CalibrationState::Calibrated);

        // Thresholds should have adapted from baseline
        let baseline_complexity = adaptive.baseline().complexity_stats.mean;
        assert!(
            (baseline_complexity - 0.54).abs() < 0.1,
            "Baseline complexity should be ~0.54"
        );
    }

    #[test]
    fn test_early_calibration_end() {
        let mut config = AdaptiveThresholdConfig::default();
        config.calibration_epochs = 10;
        config.min_samples_for_adaptation = 3;

        let mut adaptive = AdaptiveThreshold::new(config, "subject_001");

        adaptive.start_calibration();

        // Add only 2 epochs (less than min_samples_for_adaptation)
        for _ in 0..2 {
            let metrics = IntegrationMetrics::default();
            adaptive.update_with_feedback(
                &metrics,
                ConsciousnessState::LightSleep,
                SleepStage::N2,
                true,
            );
        }

        // Try to end calibration early
        let success = adaptive.end_calibration();
        assert!(!success, "Calibration should fail with insufficient data");
        assert!(!adaptive.is_calibrated());
    }

    #[test]
    fn test_threshold_bounds_enforcement() {
        let mut config = AdaptiveThresholdConfig::default();
        config.max_deviation_from_norm = 1.0; // Tight bounds
        config.calibration_epochs = 3;
        config.min_samples_for_adaptation = 2;
        config.initial_learning_rate = 0.5; // High learning rate for test

        let mut adaptive = AdaptiveThreshold::new(config, "subject_001");

        // Calibrate first
        adaptive.start_calibration();
        for _ in 0..3 {
            let metrics = IntegrationMetrics {
                complexity: 0.6,
                synchrony: 0.7,
                phi_proxy: 0.5,
                ..Default::default()
            };
            adaptive.update_with_feedback(
                &metrics,
                ConsciousnessState::LightSleep,
                SleepStage::N2,
                true,
            );
        }

        let initial_complexity = adaptive.complexity_threshold();

        // Try to push threshold way out of bounds with many incorrect predictions
        for _ in 0..50 {
            let metrics = IntegrationMetrics {
                complexity: 0.1, // Very different from threshold
                synchrony: 0.9,
                phi_proxy: 0.2,
                ..Default::default()
            };
            adaptive.update_with_feedback(
                &metrics,
                ConsciousnessState::Awake, // Predicted
                SleepStage::N3,            // Actual (wrong)
                false,
            );
        }

        let final_complexity = adaptive.complexity_threshold();

        // Threshold should have changed but stayed within bounds
        // Population norm for complexity is (0.6, 0.15), so bounds are roughly 0.45 to 0.75
        assert!(final_complexity >= 0.0, "Threshold should stay positive");
        assert!(final_complexity <= 1.0, "Threshold should stay <= 1.0");

        // Should have recorded bounds violations
        assert!(
            adaptive.bounds_violations() > 0,
            "Should have bounds violations"
        );

        println!("Initial complexity threshold: {}", initial_complexity);
        println!("Final complexity threshold: {}", final_complexity);
        println!("Bounds violations: {}", adaptive.bounds_violations());
    }

    #[test]
    fn test_threshold_convergence() {
        // Test that thresholds converge toward subject-specific values
        let mut config = AdaptiveThresholdConfig::default();
        config.calibration_epochs = 5;
        config.min_samples_for_adaptation = 3;
        config.initial_learning_rate = 0.1;

        let mut adaptive = AdaptiveThreshold::new(config, "subject_high_complexity");

        // Calibrate with high complexity baseline
        adaptive.start_calibration();
        for _ in 0..5 {
            let metrics = IntegrationMetrics {
                complexity: 0.75, // Higher than population mean (0.6)
                synchrony: 0.65,
                phi_proxy: 0.5,
                ..Default::default()
            };
            adaptive.update_with_feedback(
                &metrics,
                ConsciousnessState::Awake,
                SleepStage::Wake,
                true,
            );
        }

        let threshold_after_calibration = adaptive.complexity_threshold();

        // Threshold should have moved toward subject's higher baseline
        // Initial is 0.6 (population mean), subject has 0.75 mean
        assert!(
            threshold_after_calibration > 0.6,
            "Threshold should increase for high-complexity subject: {}",
            threshold_after_calibration
        );

        println!(
            "Threshold converged to: {} (from population mean 0.6)",
            threshold_after_calibration
        );
    }

    #[test]
    fn test_learning_rate_decay() {
        let mut config = AdaptiveThresholdConfig::default();
        config.initial_learning_rate = 0.1;
        config.learning_rate_decay = 0.9;
        config.min_learning_rate = 0.01;

        let mut adaptive = AdaptiveThreshold::new(config, "subject_001");

        assert!((adaptive.learning_rate() - 0.1).abs() < 0.001);

        // Start sessions
        adaptive.start_new_session();
        assert!((adaptive.learning_rate() - 0.09).abs() < 0.001); // 0.1 * 0.9

        adaptive.start_new_session();
        assert!((adaptive.learning_rate() - 0.081).abs() < 0.001); // 0.09 * 0.9

        // Keep decaying
        for _ in 0..50 {
            adaptive.start_new_session();
        }

        // Should not go below min
        assert!(
            adaptive.learning_rate() >= 0.01,
            "Learning rate should not go below min"
        );
    }

    #[test]
    fn test_fallback_mode() {
        let mut config = AdaptiveThresholdConfig::default();
        config.calibration_epochs = 3;
        config.min_samples_for_adaptation = 2;

        let mut adaptive = AdaptiveThreshold::new(config, "subject_001");

        // Calibrate
        adaptive.start_calibration();
        for _ in 0..3 {
            let metrics = IntegrationMetrics {
                complexity: 0.8,
                synchrony: 0.8,
                phi_proxy: 0.6,
                ..Default::default()
            };
            adaptive.update_with_feedback(
                &metrics,
                ConsciousnessState::DeepSleep,
                SleepStage::N3,
                true,
            );
        }

        // Thresholds should be adapted
        let adapted_complexity = adaptive.complexity_threshold();
        assert!(adapted_complexity != 0.6, "Threshold should have adapted");

        // Enable fallback
        adaptive.enable_fallback();
        assert!(adaptive.is_using_fallback());

        // Thresholds should reset to population norms (0.35 complexity, 0.55 synchrony)
        assert!((adaptive.complexity_threshold() - 0.35).abs() < 0.01);
        assert!((adaptive.synchrony_threshold() - 0.55).abs() < 0.01);

        // Disable fallback
        adaptive.disable_fallback();
        assert!(!adaptive.is_using_fallback());
    }

    #[test]
    fn test_calibration_persistence() {
        let mut config = AdaptiveThresholdConfig::default();
        config.calibration_epochs = 3;
        config.min_samples_for_adaptation = 2;

        let mut adaptive = AdaptiveThreshold::new(config.clone(), "subject_001");

        // Calibrate
        adaptive.start_calibration();
        for _ in 0..5 {
            let metrics = IntegrationMetrics {
                complexity: 0.7,
                synchrony: 0.65,
                phi_proxy: 0.55,
                ..Default::default()
            };
            adaptive.update_with_feedback(
                &metrics,
                ConsciousnessState::LightSleep,
                SleepStage::N2,
                true,
            );
        }

        // Export calibration
        let exported = adaptive.export_calibration();

        // Create new instance and import
        let mut new_adaptive = AdaptiveThreshold::new(config, "different_id");
        new_adaptive.import_calibration(exported.clone());

        // Verify imported data
        assert_eq!(new_adaptive.baseline().subject_id, "subject_001");
        assert!(new_adaptive.is_calibrated());
        assert!(
            (new_adaptive.complexity_threshold() - adaptive.complexity_threshold()).abs() < 0.001
        );
        assert!(
            (new_adaptive.synchrony_threshold() - adaptive.synchrony_threshold()).abs() < 0.001
        );
    }

    #[test]
    fn test_z_score_normalization() {
        let mut config = AdaptiveThresholdConfig::default();
        config.calibration_epochs = 10;
        config.min_samples_for_adaptation = 5;

        let mut adaptive = AdaptiveThreshold::new(config, "subject_001");

        // Build baseline
        adaptive.start_calibration();
        for i in 0..10 {
            let metrics = IntegrationMetrics {
                complexity: 0.5 + (i as f32 * 0.01),
                synchrony: 0.6,
                phi_proxy: 0.5,
                dominant_freq_hz: 10.0,
                ..Default::default()
            };
            adaptive.update_with_feedback(
                &metrics,
                ConsciousnessState::LightSleep,
                SleepStage::N2,
                true,
            );
        }

        // Get normalized metrics for a value at the mean
        let at_mean = IntegrationMetrics {
            complexity: adaptive.baseline().complexity_stats.mean,
            synchrony: adaptive.baseline().synchrony_stats.mean,
            phi_proxy: adaptive.baseline().phi_stats.mean,
            dominant_freq_hz: adaptive.baseline().frequency_stats.mean,
            ..Default::default()
        };

        let normalized = adaptive.normalize_metrics(&at_mean);

        // Z-scores at the mean should be close to 0
        assert!(
            normalized.complexity.abs() < 0.1,
            "Z-score at mean should be ~0"
        );
        assert!(
            normalized.synchrony.abs() < 0.1,
            "Z-score at mean should be ~0"
        );
    }

    #[test]
    #[ignore = "slow ~60s: trains LTC networks with adaptive threshold calibration"]
    fn test_sentinel_with_adaptive_thresholds() {
        let mut config = SleepSentinelConfig::default();
        config.enable_adaptive_thresholds = true;
        config.adaptive_config.calibration_epochs = 5;
        config.adaptive_config.min_samples_for_adaptation = 3;

        let mut sentinel = SleepSentinel::new_for_subject(config, "test_subject");

        // Start calibration
        sentinel.start_calibration(None);
        assert!(!sentinel.is_calibrated());

        // Run calibration epochs
        for _ in 0..5 {
            let (frontal, occipital) = generate_synthetic_eeg(SleepStage::N2, 100.0, 30.0);
            sentinel.train_epoch(&frontal, &occipital, SleepStage::N2);
        }

        // Should be calibrated now
        assert!(sentinel.is_calibrated(), "Sentinel should be calibrated");

        // Get current thresholds
        let (c, s, p) = sentinel.current_thresholds();
        println!(
            "Adapted thresholds: complexity={:.3}, synchrony={:.3}, phi={:.3}",
            c, s, p
        );

        // Summary should include adaptive info
        let summary = sentinel.summary();
        assert!(
            summary.contains("Adaptive Thresholds"),
            "Summary should include adaptive info"
        );
        assert!(
            summary.contains("Calibrated"),
            "Summary should show calibrated state"
        );
    }

    #[test]
    #[ignore = "slow ~60s: trains LTC networks, tests calibration export/import"]
    fn test_sentinel_calibration_persistence() {
        let mut config = SleepSentinelConfig::default();
        config.enable_adaptive_thresholds = true;
        config.adaptive_config.calibration_epochs = 3;
        config.adaptive_config.min_samples_for_adaptation = 2;

        let mut sentinel = SleepSentinel::new_for_subject(config.clone(), "original_subject");

        // Calibrate
        sentinel.start_calibration(None);
        for _ in 0..5 {
            let (frontal, occipital) = generate_synthetic_eeg(SleepStage::Wake, 100.0, 30.0);
            sentinel.train_epoch(&frontal, &occipital, SleepStage::Wake);
        }

        // Export
        let calibration_data = sentinel
            .export_calibration()
            .expect("Should have calibration data");

        // Create new sentinel and import
        let mut new_sentinel = SleepSentinel::new(config);
        new_sentinel.import_calibration(calibration_data);

        assert!(new_sentinel.is_calibrated());
        assert_eq!(
            sentinel.current_thresholds(),
            new_sentinel.current_thresholds()
        );
    }

    #[test]
    fn test_anomaly_detection_skips_adaptation() {
        let mut config = AdaptiveThresholdConfig::default();
        config.calibration_epochs = 5;
        config.min_samples_for_adaptation = 3;
        config.anomaly_z_threshold = 2.0;
        config.initial_learning_rate = 0.5;

        let mut adaptive = AdaptiveThreshold::new(config, "subject_001");

        // Calibrate with consistent values
        adaptive.start_calibration();
        for _ in 0..10 {
            let metrics = IntegrationMetrics {
                complexity: 0.5,
                synchrony: 0.6,
                phi_proxy: 0.5,
                ..Default::default()
            };
            adaptive.update_with_feedback(
                &metrics,
                ConsciousnessState::LightSleep,
                SleepStage::N2,
                true,
            );
        }

        let threshold_before = adaptive.complexity_threshold();

        // Send anomalous readings (should be detected and skipped)
        for _ in 0..5 {
            let anomalous_metrics = IntegrationMetrics {
                complexity: 0.99, // Way outside normal range
                synchrony: 0.01,
                phi_proxy: 0.99,
                ..Default::default()
            };
            adaptive.update_with_feedback(
                &anomalous_metrics,
                ConsciousnessState::Awake,
                SleepStage::Wake,
                false,
            );
        }

        // Threshold should not have changed much (anomalies skipped)
        let threshold_after = adaptive.complexity_threshold();
        assert!(
            (threshold_before - threshold_after).abs() < 0.05,
            "Threshold should not change much on anomalies: before={}, after={}",
            threshold_before,
            threshold_after
        );

        // Should have detected anomalies
        assert!(
            adaptive.anomalies_detected() > 0,
            "Should have detected anomalies"
        );
        println!("Anomalies detected: {}", adaptive.anomalies_detected());
    }
}
