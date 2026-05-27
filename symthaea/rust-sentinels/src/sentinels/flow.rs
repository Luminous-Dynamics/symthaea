// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! FlowSentinel - Proof of Flow
//!
//! Detects optimal performance states where challenge matches skill.
//!
//! # Neuroscience Basis
//! - **Alpha Synchronization**: Transient alpha increases during "in the zone" states
//! - **Hypofrontality**: Reduced frontal activity during flow (less self-monitoring)
//! - **Theta/Beta Balance**: Optimal ratio for peak performance
//! - **Sensorimotor Rhythm (SMR)**: 12-15 Hz associated with calm focus

use crate::signal::{band_power, welch_psd};

/// Flow state classification
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum FlowState {
    /// Anxiety - challenge exceeds skill
    Anxiety,
    /// Boredom - skill exceeds challenge
    Boredom,
    /// Apathy - low skill, low challenge
    Apathy,
    /// Control - high skill, moderate challenge
    Control,
    /// Flow - optimal balance of challenge and skill
    Flow,
}

/// Flow analysis result
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FlowScore {
    /// Depth of immersion/engagement (0-1)
    pub immersion: f32,
    /// Perceived ease of action (0-1)
    pub effortlessness: f32,
    /// Temporal perception shift indicator (0-1)
    pub time_dilation: f32,
    /// Execution quality estimate (0-1)
    pub performance: f32,
    /// Overall flow index (0-1)
    pub flow_index: f32,
    /// Classified flow state
    pub state: FlowState,
    /// Confidence in classification (0-1)
    pub confidence: f32,
}

impl FlowScore {
    /// Check if currently in flow state
    pub fn in_flow(&self) -> bool {
        self.state == FlowState::Flow
    }

    /// Get a human-readable description
    pub fn description(&self) -> &'static str {
        match self.state {
            FlowState::Anxiety => "Challenge exceeds current skill level",
            FlowState::Boredom => "Task too easy for current skill level",
            FlowState::Apathy => "Low engagement, disinterested",
            FlowState::Control => "Comfortable performance, room for more challenge",
            FlowState::Flow => "Optimal state - fully immersed in the zone",
        }
    }
}

/// Configuration for flow detection
#[derive(Debug, Clone)]
pub struct FlowConfig {
    /// Optimal alpha power level (peak of gaussian)
    pub optimal_alpha: f32,
    /// Alpha tolerance (width of gaussian)
    pub alpha_tolerance: f32,
    /// Maximum frontal beta for effortlessness
    pub max_frontal_beta: f32,
    /// SMR band (12-15 Hz) weight
    pub smr_weight: f32,
    /// Flow threshold
    pub flow_threshold: f32,
}

impl Default for FlowConfig {
    fn default() -> Self {
        Self {
            optimal_alpha: 0.35,
            alpha_tolerance: 0.15,
            max_frontal_beta: 0.20,
            smr_weight: 0.3,
            flow_threshold: 0.6,
        }
    }
}

/// FlowSentinel - Detects optimal performance flow states
pub struct FlowSentinel {
    config: FlowConfig,
    baseline_alpha: f32,
    baseline_beta: f32,
    calibrated: bool,
}

impl FlowSentinel {
    /// Create a new FlowSentinel with default configuration
    pub fn new() -> Self {
        Self {
            config: FlowConfig::default(),
            baseline_alpha: 0.30,
            baseline_beta: 0.20,
            calibrated: false,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: FlowConfig) -> Self {
        Self {
            config,
            baseline_alpha: 0.30,
            baseline_beta: 0.20,
            calibrated: false,
        }
    }

    /// Calibrate baseline from resting state data
    pub fn calibrate(&mut self, data: &[f32], sample_rate: f32) {
        let powers = self.extract_powers(data, sample_rate);
        self.baseline_alpha = powers.1;
        self.baseline_beta = powers.2;
        self.calibrated = true;
    }

    /// Analyze flow state from EEG data
    pub fn analyze(&self, data: &[f32], sample_rate: f32) -> FlowScore {
        let (theta_rel, alpha_rel, beta_rel, smr_rel) = self.extract_powers(data, sample_rate);

        // Calculate alpha optimality using gaussian function
        // Flow requires moderate alpha (not too high = drowsy, not too low = stressed)
        let alpha_diff = (alpha_rel - self.config.optimal_alpha).abs();
        let alpha_optimal =
            (-alpha_diff.powi(2) / (2.0 * self.config.alpha_tolerance.powi(2))).exp();

        // Calculate effortlessness from reduced frontal beta
        // Lower beta = less self-monitoring, more automatic action
        let effortlessness = (1.0 - beta_rel / self.config.max_frontal_beta)
            .max(0.0)
            .min(1.0);

        // Calculate immersion from theta/alpha balance
        let immersion = ((alpha_rel + theta_rel * 0.5) * 1.5).min(1.0);

        // Time dilation correlates with theta activity and reduced beta
        let time_dilation = (theta_rel * 2.0 * effortlessness).min(1.0);

        // Performance estimate from SMR (sensorimotor rhythm 12-15 Hz)
        let performance = (smr_rel * 4.0).min(1.0);

        // Calculate overall flow index
        // Geometric mean to require all components
        let flow_index =
            (alpha_optimal * effortlessness * immersion * (0.5 + performance * 0.5)).powf(0.5);

        // Classify state based on various indicators
        let state = self.classify_state(flow_index, alpha_rel, beta_rel, theta_rel);

        // Calculate confidence
        let confidence = self.calculate_confidence(alpha_rel, beta_rel);

        FlowScore {
            immersion,
            effortlessness,
            time_dilation,
            performance,
            flow_index,
            state,
            confidence,
        }
    }

    /// Extract relative band powers including SMR
    fn extract_powers(&self, data: &[f32], sample_rate: f32) -> (f32, f32, f32, f32) {
        let nperseg = (sample_rate as usize).min(data.len());
        let (freqs, psd) = welch_psd(data, sample_rate, nperseg);

        let delta = band_power(&freqs, &psd, 0.5, 4.0);
        let theta = band_power(&freqs, &psd, 4.0, 8.0);
        let alpha = band_power(&freqs, &psd, 8.0, 13.0);
        let smr = band_power(&freqs, &psd, 12.0, 15.0); // Sensorimotor rhythm
        let beta = band_power(&freqs, &psd, 13.0, 30.0);
        let gamma = band_power(&freqs, &psd, 30.0, 50.0);

        let total = delta + theta + alpha + beta + gamma + 1e-10;

        (theta / total, alpha / total, beta / total, smr / total)
    }

    /// Classify flow state
    fn classify_state(&self, flow_index: f32, alpha: f32, beta: f32, theta: f32) -> FlowState {
        // High flow index = Flow state
        if flow_index >= self.config.flow_threshold {
            return FlowState::Flow;
        }

        // High beta + low alpha = Anxiety (overthinking)
        if beta > 0.30 && alpha < 0.25 {
            return FlowState::Anxiety;
        }

        // Very high alpha + very low beta = Boredom (under-stimulated)
        if alpha > 0.45 && beta < 0.10 {
            return FlowState::Boredom;
        }

        // Low everything = Apathy
        if alpha < 0.20 && theta < 0.15 && beta < 0.15 {
            return FlowState::Apathy;
        }

        // Good alpha but not quite flow = Control
        FlowState::Control
    }

    /// Calculate confidence
    fn calculate_confidence(&self, alpha: f32, beta: f32) -> f32 {
        // Higher confidence when signals are in expected ranges
        let alpha_conf = if alpha > 0.15 && alpha < 0.50 {
            1.0
        } else {
            0.7
        };
        let beta_conf = if beta > 0.05 && beta < 0.35 { 1.0 } else { 0.7 };

        (alpha_conf + beta_conf) / 2.0
    }
}

impl Default for FlowSentinel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_flow_signal(n_samples: usize, sample_rate: f32, flow_level: f32) -> Vec<f32> {
        use std::f32::consts::PI;
        (0..n_samples)
            .map(|i| {
                let t = i as f32 / sample_rate;
                // Flow state: moderate alpha, low beta, some theta
                let alpha_weight = 0.30 + flow_level * 0.1; // Optimal around 0.35
                let beta_weight = 0.25 - flow_level * 0.15; // Lower in flow
                let theta_weight = 0.15 + flow_level * 0.1;
                let smr_weight = 0.1 + flow_level * 0.1; // Higher in flow

                theta_weight * (2.0 * PI * 6.0 * t).sin()
                    + alpha_weight * (2.0 * PI * 10.0 * t).sin()
                    + smr_weight * (2.0 * PI * 13.5 * t).sin()
                    + beta_weight * (2.0 * PI * 20.0 * t).sin()
            })
            .collect()
    }

    #[test]
    fn test_flow_detection() {
        let sentinel = FlowSentinel::new();
        let sample_rate = 256.0;

        // Non-flow signal (high beta, anxious)
        let anxious = generate_flow_signal(1280, sample_rate, 0.1);
        let score_anxious = sentinel.analyze(&anxious, sample_rate);

        // Flow signal
        let flow = generate_flow_signal(1280, sample_rate, 0.9);
        let score_flow = sentinel.analyze(&flow, sample_rate);

        assert!(
            score_flow.flow_index > score_anxious.flow_index,
            "Flow signal should have higher flow index"
        );
    }

    #[test]
    fn test_flow_state_classification() {
        let sentinel = FlowSentinel::new();

        // Flow state
        assert_eq!(
            sentinel.classify_state(0.7, 0.35, 0.15, 0.20),
            FlowState::Flow
        );

        // Anxiety state
        assert_eq!(
            sentinel.classify_state(0.3, 0.20, 0.35, 0.15),
            FlowState::Anxiety
        );
    }

    #[test]
    fn test_effortlessness_calculation() {
        let sentinel = FlowSentinel::new();
        let sample_rate = 256.0;

        // High beta = low effortlessness
        let high_beta = generate_flow_signal(1280, sample_rate, 0.1);
        let score_high_beta = sentinel.analyze(&high_beta, sample_rate);

        // Low beta = high effortlessness
        let low_beta = generate_flow_signal(1280, sample_rate, 0.9);
        let score_low_beta = sentinel.analyze(&low_beta, sample_rate);

        assert!(
            score_low_beta.effortlessness > score_high_beta.effortlessness,
            "Low beta should have higher effortlessness"
        );
    }
}