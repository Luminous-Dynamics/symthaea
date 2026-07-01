// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Phi Feedback Controller
//!
//! Closes the loop between Φ measurement and consciousness processing.
//! Instead of Φ being a passive metric, the feedback controller uses Φ trends
//! to dynamically modulate pipeline parameters: binding thresholds, workspace
//! capacity, attention gain, and consciousness threshold.
//!
//! ## Feedback Mechanisms
//!
//! 1. **Adaptive Binding**: Rising Φ → tighter binding threshold (more selective),
//!    falling Φ → looser binding (more exploratory)
//! 2. **Workspace Gating**: Φ above target → expand workspace capacity,
//!    below → contract to focus resources
//! 3. **Attention Gain**: Φ trend modulates attention precision
//!    (high Φ = confident exploitation, low Φ = broad exploration)
//! 4. **Consciousness Threshold**: Adapts based on running Φ average to prevent
//!    the system from being perpetually conscious or unconscious

use serde::{Deserialize, Serialize};

/// Configuration for the Phi feedback controller
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiFeedbackConfig {
    /// Target Φ value the controller tries to maintain
    pub phi_target: f64,
    /// Learning rate for parameter adaptation (0.0-1.0)
    pub learning_rate: f64,
    /// Size of the sliding window for trend estimation
    pub window_size: usize,
    /// Minimum binding threshold (floor)
    pub min_binding_threshold: f64,
    /// Maximum binding threshold (ceiling)
    pub max_binding_threshold: f64,
    /// Minimum workspace capacity
    pub min_workspace_capacity: usize,
    /// Maximum workspace capacity
    pub max_workspace_capacity: usize,
    /// Enable consciousness threshold adaptation
    pub adapt_consciousness_threshold: bool,
    /// Smoothing factor for exponential moving average (0.0-1.0)
    pub ema_alpha: f64,
}

impl Default for PhiFeedbackConfig {
    fn default() -> Self {
        Self {
            phi_target: 0.5,
            learning_rate: 0.1,
            window_size: 20,
            min_binding_threshold: 0.3,
            max_binding_threshold: 0.9,
            min_workspace_capacity: 1,
            max_workspace_capacity: 10,
            adapt_consciousness_threshold: true,
            ema_alpha: 0.2,
        }
    }
}

/// Modulation signals produced by the feedback controller
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackModulation {
    /// Adjusted binding threshold
    pub binding_threshold: f64,
    /// Adjusted workspace capacity
    pub workspace_capacity: usize,
    /// Attention gain multiplier (1.0 = neutral, >1 = sharpen, <1 = broaden)
    pub attention_gain: f64,
    /// Adjusted consciousness threshold
    pub consciousness_threshold: f64,
    /// Current Φ trend (positive = rising, negative = falling)
    pub phi_trend: f64,
    /// Error signal: current Φ - target Φ
    pub phi_error: f64,
    /// Controller confidence (based on history length / window_size)
    pub confidence: f64,
}

/// Phi Feedback Controller
///
/// Uses a proportional-derivative (PD) control strategy to modulate
/// consciousness pipeline parameters based on Φ measurements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiFeedbackController {
    config: PhiFeedbackConfig,
    /// History of Φ measurements
    phi_history: Vec<f64>,
    /// Exponential moving average of Φ
    phi_ema: f64,
    /// Previous Φ error (for derivative term)
    prev_error: f64,
    /// Accumulated modulation state
    binding_threshold: f64,
    workspace_capacity_f: f64,
    consciousness_threshold: f64,
    /// Total steps processed
    step_count: u64,
}

impl PhiFeedbackController {
    /// Create a new feedback controller with default config
    pub fn new() -> Self {
        Self::with_config(PhiFeedbackConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: PhiFeedbackConfig) -> Self {
        Self {
            binding_threshold: 0.7,    // Default starting point
            workspace_capacity_f: 3.0, // Default starting point
            consciousness_threshold: 0.5,
            phi_ema: 0.0,
            prev_error: 0.0,
            phi_history: Vec::with_capacity(config.window_size),
            step_count: 0,
            config,
        }
    }

    /// Feed a new Φ measurement and get modulation signals
    ///
    /// This is the core feedback loop. Call this after each `process()` cycle
    /// with the resulting Φ value.
    pub fn step(&mut self, phi: f64) -> FeedbackModulation {
        // Record history
        self.phi_history.push(phi);
        if self.phi_history.len() > self.config.window_size {
            self.phi_history.remove(0);
        }
        self.step_count += 1;

        // Update EMA
        if self.step_count == 1 {
            self.phi_ema = phi;
        } else {
            self.phi_ema =
                self.config.ema_alpha * phi + (1.0 - self.config.ema_alpha) * self.phi_ema;
        }

        // Compute error and trend
        let error = phi - self.config.phi_target;
        let trend = self.compute_trend();
        let derivative = error - self.prev_error;
        self.prev_error = error;

        // Confidence based on how much history we have
        let confidence = (self.phi_history.len() as f64 / self.config.window_size as f64).min(1.0);

        // PD control: modulation = learning_rate * (P * error + D * derivative)
        let p_gain = 1.0;
        let d_gain = 0.5;
        let modulation =
            self.config.learning_rate * confidence * (p_gain * error + d_gain * derivative);

        // === 1. Adaptive Binding Threshold ===
        // Rising Φ (positive error) → tighter binding (higher threshold, more selective)
        // Falling Φ (negative error) → looser binding (lower threshold, more exploratory)
        self.binding_threshold = (self.binding_threshold + modulation * 0.5).clamp(
            self.config.min_binding_threshold,
            self.config.max_binding_threshold,
        );

        // === 2. Workspace Capacity ===
        // High Φ → expand workspace (system can integrate more)
        // Low Φ → contract workspace (focus on fewer items)
        self.workspace_capacity_f = (self.workspace_capacity_f + modulation * 2.0).clamp(
            self.config.min_workspace_capacity as f64,
            self.config.max_workspace_capacity as f64,
        );

        // === 3. Attention Gain ===
        // Positive trend → exploit (sharpen attention)
        // Negative trend → explore (broaden attention)
        let attention_gain = (1.0 + trend * 0.5).clamp(0.5, 2.0);

        // === 4. Consciousness Threshold ===
        // Adapts to keep the system near its natural operating point
        if self.config.adapt_consciousness_threshold {
            // Move threshold toward current EMA to avoid system being stuck
            let threshold_error = self.phi_ema - self.consciousness_threshold;
            self.consciousness_threshold = (self.consciousness_threshold
                + threshold_error * self.config.learning_rate * 0.3)
                .clamp(0.2, 0.8);
        }

        FeedbackModulation {
            binding_threshold: self.binding_threshold,
            workspace_capacity: self.workspace_capacity_f.round() as usize,
            attention_gain,
            consciousness_threshold: self.consciousness_threshold,
            phi_trend: trend,
            phi_error: error,
            confidence,
        }
    }

    /// Compute Φ trend from recent history
    fn compute_trend(&self) -> f64 {
        if self.phi_history.len() < 2 {
            return 0.0;
        }

        let n = self.phi_history.len();
        let half = n / 2;
        if half == 0 {
            return 0.0;
        }

        let first_half_avg: f64 = self.phi_history[..half].iter().sum::<f64>() / half as f64;
        let second_half_avg: f64 = self.phi_history[half..].iter().sum::<f64>() / (n - half) as f64;

        second_half_avg - first_half_avg
    }

    /// Get the current Φ EMA
    pub fn phi_ema(&self) -> f64 {
        self.phi_ema
    }

    /// Get the current Φ trend
    pub fn phi_trend(&self) -> f64 {
        self.compute_trend()
    }

    /// Get the number of steps processed
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Get the Φ history
    pub fn phi_history(&self) -> &[f64] {
        &self.phi_history
    }

    /// Get the current config
    pub fn config(&self) -> &PhiFeedbackConfig {
        &self.config
    }

    /// Get average Φ over the window
    pub fn average_phi(&self) -> f64 {
        if self.phi_history.is_empty() {
            return 0.0;
        }
        self.phi_history.iter().sum::<f64>() / self.phi_history.len() as f64
    }

    /// Get peak Φ observed
    pub fn peak_phi(&self) -> f64 {
        self.phi_history.iter().cloned().fold(0.0_f64, f64::max)
    }

    /// Get the Φ variance over the window
    pub fn phi_variance(&self) -> f64 {
        if self.phi_history.len() < 2 {
            return 0.0;
        }
        let mean = self.average_phi();
        let sum_sq: f64 = self.phi_history.iter().map(|p| (p - mean).powi(2)).sum();
        sum_sq / (self.phi_history.len() - 1) as f64
    }

    /// Check if Φ has stabilized (variance below threshold)
    pub fn is_stable(&self, variance_threshold: f64) -> bool {
        self.phi_history.len() >= self.config.window_size
            && self.phi_variance() < variance_threshold
    }

    /// Reset the controller state while keeping configuration
    pub fn reset(&mut self) {
        self.phi_history.clear();
        self.phi_ema = 0.0;
        self.prev_error = 0.0;
        self.binding_threshold = 0.7;
        self.workspace_capacity_f = 3.0;
        self.consciousness_threshold = 0.5;
        self.step_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_controller_creation() {
        let ctrl = PhiFeedbackController::new();
        assert_eq!(ctrl.step_count(), 0);
        assert_eq!(ctrl.phi_ema(), 0.0);
        assert!(ctrl.phi_history().is_empty());
    }

    #[test]
    fn test_controller_with_config() {
        let config = PhiFeedbackConfig {
            phi_target: 0.7,
            learning_rate: 0.2,
            window_size: 10,
            ..Default::default()
        };
        let ctrl = PhiFeedbackController::with_config(config);
        assert_eq!(ctrl.config().phi_target, 0.7);
        assert_eq!(ctrl.config().learning_rate, 0.2);
        assert_eq!(ctrl.config().window_size, 10);
    }

    #[test]
    fn test_single_step() {
        let mut ctrl = PhiFeedbackController::new();
        let modulation = ctrl.step(0.6);

        assert_eq!(ctrl.step_count(), 1);
        assert!((ctrl.phi_ema() - 0.6).abs() < 1e-10);
        assert!(modulation.confidence > 0.0);
        assert!(modulation.binding_threshold > 0.0);
        assert!(modulation.workspace_capacity >= 1);
    }

    #[test]
    fn test_rising_phi_tightens_binding() {
        let mut ctrl = PhiFeedbackController::with_config(PhiFeedbackConfig {
            phi_target: 0.3,
            learning_rate: 0.5,
            window_size: 5,
            ..Default::default()
        });

        // Feed rising Φ values (above target)
        let mut last_binding = 0.7;
        for i in 0..10 {
            let phi = 0.5 + i as f64 * 0.03;
            let m = ctrl.step(phi);
            last_binding = m.binding_threshold;
        }

        // Binding threshold should have risen (more selective)
        assert!(
            last_binding > 0.7,
            "Rising Φ above target should tighten binding, got {}",
            last_binding
        );
    }

    #[test]
    fn test_falling_phi_loosens_binding() {
        let mut ctrl = PhiFeedbackController::with_config(PhiFeedbackConfig {
            phi_target: 0.7,
            learning_rate: 0.5,
            window_size: 5,
            ..Default::default()
        });

        // Feed falling Φ values (below target)
        let mut last_binding = 0.7;
        for i in 0..10 {
            let phi = 0.4 - i as f64 * 0.02;
            let m = ctrl.step(phi);
            last_binding = m.binding_threshold;
        }

        // Binding threshold should have dropped (more exploratory)
        assert!(
            last_binding < 0.7,
            "Falling Φ below target should loosen binding, got {}",
            last_binding
        );
    }

    #[test]
    fn test_workspace_expansion_with_high_phi() {
        let mut ctrl = PhiFeedbackController::with_config(PhiFeedbackConfig {
            phi_target: 0.3,
            learning_rate: 0.3,
            window_size: 5,
            ..Default::default()
        });

        // Consistently high Φ
        let mut last_capacity = 3;
        for _ in 0..20 {
            let m = ctrl.step(0.8);
            last_capacity = m.workspace_capacity;
        }

        assert!(
            last_capacity > 3,
            "High Φ should expand workspace, got {}",
            last_capacity
        );
    }

    #[test]
    fn test_workspace_contraction_with_low_phi() {
        let mut ctrl = PhiFeedbackController::with_config(PhiFeedbackConfig {
            phi_target: 0.7,
            learning_rate: 0.3,
            window_size: 5,
            ..Default::default()
        });

        // Consistently low Φ
        let mut last_capacity = 3;
        for _ in 0..20 {
            let m = ctrl.step(0.1);
            last_capacity = m.workspace_capacity;
        }

        assert!(
            last_capacity < 3,
            "Low Φ should contract workspace, got {}",
            last_capacity
        );
    }

    #[test]
    fn test_attention_gain_positive_trend() {
        let mut ctrl = PhiFeedbackController::with_config(PhiFeedbackConfig {
            window_size: 6,
            ..Default::default()
        });

        // Rising sequence
        for i in 0..6 {
            ctrl.step(0.3 + i as f64 * 0.1);
        }
        let m = ctrl.step(0.95);

        assert!(
            m.attention_gain > 1.0,
            "Positive trend should sharpen attention, got {}",
            m.attention_gain
        );
    }

    #[test]
    fn test_attention_gain_negative_trend() {
        let mut ctrl = PhiFeedbackController::with_config(PhiFeedbackConfig {
            window_size: 6,
            ..Default::default()
        });

        // Falling sequence
        for i in 0..6 {
            ctrl.step(0.9 - i as f64 * 0.1);
        }
        let m = ctrl.step(0.1);

        assert!(
            m.attention_gain < 1.0,
            "Negative trend should broaden attention, got {}",
            m.attention_gain
        );
    }

    #[test]
    fn test_consciousness_threshold_adapts() {
        let mut ctrl = PhiFeedbackController::with_config(PhiFeedbackConfig {
            phi_target: 0.5,
            learning_rate: 0.3,
            adapt_consciousness_threshold: true,
            window_size: 10,
            ..Default::default()
        });

        // High Φ should raise consciousness threshold
        for _ in 0..30 {
            ctrl.step(0.8);
        }
        let m = ctrl.step(0.8);

        assert!(
            m.consciousness_threshold > 0.5,
            "High Φ should raise consciousness threshold, got {}",
            m.consciousness_threshold
        );
    }

    #[test]
    fn test_thresholds_stay_in_bounds() {
        let mut ctrl = PhiFeedbackController::new();

        // Extreme high Φ
        for _ in 0..100 {
            let m = ctrl.step(1.0);
            assert!(m.binding_threshold <= ctrl.config().max_binding_threshold);
            assert!(m.binding_threshold >= ctrl.config().min_binding_threshold);
            assert!(m.workspace_capacity <= ctrl.config().max_workspace_capacity);
            assert!(m.workspace_capacity >= ctrl.config().min_workspace_capacity);
            assert!(m.consciousness_threshold >= 0.2);
            assert!(m.consciousness_threshold <= 0.8);
            assert!(m.attention_gain >= 0.5);
            assert!(m.attention_gain <= 2.0);
        }

        ctrl.reset();

        // Extreme low Φ
        for _ in 0..100 {
            let m = ctrl.step(0.0);
            assert!(m.binding_threshold <= ctrl.config().max_binding_threshold);
            assert!(m.binding_threshold >= ctrl.config().min_binding_threshold);
            assert!(m.workspace_capacity <= ctrl.config().max_workspace_capacity);
            assert!(m.workspace_capacity >= ctrl.config().min_workspace_capacity);
        }
    }

    #[test]
    fn test_phi_statistics() {
        let mut ctrl = PhiFeedbackController::new();

        for i in 0..10 {
            ctrl.step(0.3 + i as f64 * 0.05);
        }

        assert!(ctrl.average_phi() > 0.3);
        assert!(ctrl.peak_phi() > 0.7);
        assert!(ctrl.phi_variance() > 0.0);
    }

    #[test]
    fn test_stability_detection() {
        let mut ctrl = PhiFeedbackController::with_config(PhiFeedbackConfig {
            window_size: 5,
            ..Default::default()
        });

        // Not stable yet (not enough data)
        assert!(!ctrl.is_stable(0.01));

        // Feed constant values
        for _ in 0..5 {
            ctrl.step(0.5);
        }

        assert!(ctrl.is_stable(0.01), "Constant Φ should be stable");
    }

    #[test]
    fn test_reset() {
        let mut ctrl = PhiFeedbackController::new();

        for _ in 0..10 {
            ctrl.step(0.7);
        }
        assert!(ctrl.step_count() > 0);

        ctrl.reset();
        assert_eq!(ctrl.step_count(), 0);
        assert!(ctrl.phi_history().is_empty());
        assert_eq!(ctrl.phi_ema(), 0.0);
    }

    #[test]
    fn test_confidence_grows_with_history() {
        let mut ctrl = PhiFeedbackController::with_config(PhiFeedbackConfig {
            window_size: 10,
            ..Default::default()
        });

        let m1 = ctrl.step(0.5);
        assert!(m1.confidence < 0.2, "First step should have low confidence");

        for _ in 0..9 {
            ctrl.step(0.5);
        }
        let m10 = ctrl.step(0.5);
        // At 11 steps with window 10, confidence should be 1.0 (window is full)
        assert!(
            (m10.confidence - 1.0).abs() < 1e-10,
            "Full window should have confidence 1.0, got {}",
            m10.confidence
        );
    }

    #[test]
    fn test_trend_computation() {
        let mut ctrl = PhiFeedbackController::with_config(PhiFeedbackConfig {
            window_size: 10,
            ..Default::default()
        });

        // Rising: first half low, second half high
        for _ in 0..5 {
            ctrl.step(0.3);
        }
        for _ in 0..5 {
            ctrl.step(0.7);
        }

        assert!(
            ctrl.phi_trend() > 0.0,
            "Rising sequence should have positive trend"
        );

        ctrl.reset();

        // Falling: first half high, second half low
        for _ in 0..5 {
            ctrl.step(0.7);
        }
        for _ in 0..5 {
            ctrl.step(0.3);
        }

        assert!(
            ctrl.phi_trend() < 0.0,
            "Falling sequence should have negative trend"
        );
    }
}
