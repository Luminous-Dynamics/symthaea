// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! AttentionSentinel - Proof of Attention
//!
//! Detects sustained attention and cognitive engagement from EEG patterns.
//!
//! # Neuroscience Basis
//! - **Frontal Theta**: Sustained attention correlates with increased frontal midline theta (4-8 Hz)
//! - **Alpha Suppression**: Task engagement reduces posterior alpha
//! - **Beta Synchronization**: Frontal beta increases during focused attention
//! - **P300 Component**: Event-related potential indicating attention to rare stimuli

use crate::signal::{band_power, welch_psd};

/// Attention state classification
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum AttentionState {
    /// Distracted - low attention, mind wandering
    Distracted,
    /// Normal - baseline attention level
    Normal,
    /// Focused - heightened attention
    Focused,
    /// DeepFocus - sustained deep concentration
    DeepFocus,
}

/// Attention analysis result
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AttentionScore {
    /// Sustained attention capacity (0-1)
    pub sustained: f32,
    /// Ability to filter distractions (0-1)
    pub selective: f32,
    /// General arousal/alertness level (0-1)
    pub alertness: f32,
    /// Mental workload estimate (0-1)
    pub cognitive_load: f32,
    /// Overall attention index (0-3)
    pub attention_index: f32,
    /// Classified attention state
    pub state: AttentionState,
    /// Confidence in classification (0-1)
    pub confidence: f32,
}

impl AttentionScore {
    /// Get a human-readable description of the attention state
    pub fn description(&self) -> &'static str {
        match self.state {
            AttentionState::Distracted => "Mind wandering, low engagement",
            AttentionState::Normal => "Baseline attention level",
            AttentionState::Focused => "Heightened attention and engagement",
            AttentionState::DeepFocus => "Sustained deep concentration",
        }
    }
}

/// Configuration for attention detection thresholds
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Frontal theta threshold for sustained attention
    pub theta_threshold: f32,
    /// Alpha suppression threshold for selective attention
    pub alpha_suppression_threshold: f32,
    /// Beta threshold for alertness
    pub beta_threshold: f32,
    /// Attention index thresholds [distracted, normal, focused]
    pub index_thresholds: [f32; 3],
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            theta_threshold: 0.20,             // 20% relative theta power
            alpha_suppression_threshold: 0.30, // 30% alpha = baseline
            beta_threshold: 0.15,              // 15% beta for alertness
            index_thresholds: [0.5, 1.0, 2.0], // Distracted < 0.5, Normal < 1.0, Focused < 2.0, DeepFocus >= 2.0
        }
    }
}

/// AttentionSentinel - Detects attention and cognitive engagement
pub struct AttentionSentinel {
    config: AttentionConfig,
    baseline_theta: f32,
    baseline_alpha: f32,
    baseline_beta: f32,
    calibrated: bool,
}

impl AttentionSentinel {
    /// Create a new AttentionSentinel with default configuration
    pub fn new() -> Self {
        Self {
            config: AttentionConfig::default(),
            baseline_theta: 0.20,
            baseline_alpha: 0.35,
            baseline_beta: 0.15,
            calibrated: false,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: AttentionConfig) -> Self {
        Self {
            config,
            baseline_theta: 0.20,
            baseline_alpha: 0.35,
            baseline_beta: 0.15,
            calibrated: false,
        }
    }

    /// Calibrate baseline from resting state data
    pub fn calibrate(&mut self, data: &[f32], sample_rate: f32) {
        let powers = self.extract_powers(data, sample_rate);
        self.baseline_theta = powers.0;
        self.baseline_alpha = powers.1;
        self.baseline_beta = powers.2;
        self.calibrated = true;
    }

    /// Analyze attention state from EEG data
    pub fn analyze(&self, data: &[f32], sample_rate: f32) -> AttentionScore {
        let (theta_rel, alpha_rel, beta_rel, gamma_rel) = self.extract_powers(data, sample_rate);

        // Calculate sustained attention from frontal theta
        // Higher theta relative to baseline = more sustained attention
        let theta_ratio = theta_rel / self.baseline_theta;
        let sustained = (theta_ratio - 0.8).max(0.0).min(1.0) * 2.5;
        let sustained = sustained.min(1.0);

        // Calculate selective attention from alpha suppression
        // Lower alpha relative to baseline = more selective attention (task engagement)
        let alpha_ratio = alpha_rel / self.baseline_alpha;
        let selective = (1.0 - alpha_ratio).max(0.0).min(1.0);

        // Calculate alertness from beta power
        let beta_ratio = beta_rel / self.baseline_beta;
        let alertness = ((beta_ratio - 0.5) * 2.0).max(0.0).min(1.0);

        // Estimate cognitive load from theta + gamma
        let cognitive_load = ((theta_rel + gamma_rel) * 2.0).min(1.0);

        // Calculate overall attention index
        // Formula: (sustained + selective + alertness) / 1.5
        // Range: 0-2 typical, can go to 3 for extreme focus
        let attention_index = sustained * 1.2 + selective * 1.0 + alertness * 0.8;

        // Classify state
        let state = self.classify_state(attention_index);

        // Calculate confidence based on signal consistency
        let confidence = self.calculate_confidence(theta_rel, alpha_rel, beta_rel);

        AttentionScore {
            sustained,
            selective,
            alertness,
            cognitive_load,
            attention_index,
            state,
            confidence,
        }
    }

    /// Extract relative band powers
    fn extract_powers(&self, data: &[f32], sample_rate: f32) -> (f32, f32, f32, f32) {
        let nperseg = (sample_rate as usize).min(data.len());
        let (freqs, psd) = welch_psd(data, sample_rate, nperseg);

        let delta = band_power(&freqs, &psd, 0.5, 4.0);
        let theta = band_power(&freqs, &psd, 4.0, 8.0);
        let alpha = band_power(&freqs, &psd, 8.0, 13.0);
        let beta = band_power(&freqs, &psd, 13.0, 30.0);
        let gamma = band_power(&freqs, &psd, 30.0, 50.0);

        let total = delta + theta + alpha + beta + gamma + 1e-10;

        (theta / total, alpha / total, beta / total, gamma / total)
    }

    /// Classify attention state from index
    fn classify_state(&self, index: f32) -> AttentionState {
        let thresholds = &self.config.index_thresholds;
        if index < thresholds[0] {
            AttentionState::Distracted
        } else if index < thresholds[1] {
            AttentionState::Normal
        } else if index < thresholds[2] {
            AttentionState::Focused
        } else {
            AttentionState::DeepFocus
        }
    }

    /// Calculate confidence based on signal characteristics
    fn calculate_confidence(&self, theta: f32, alpha: f32, beta: f32) -> f32 {
        // Higher confidence when band powers are in expected ranges
        let theta_conf = if theta > 0.1 && theta < 0.4 { 1.0 } else { 0.7 };
        let alpha_conf = if alpha > 0.1 && alpha < 0.5 { 1.0 } else { 0.7 };
        let beta_conf = if beta > 0.05 && beta < 0.4 { 1.0 } else { 0.7 };

        (theta_conf + alpha_conf + beta_conf) / 3.0
    }
}

impl Default for AttentionSentinel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_attention_signal(
        n_samples: usize,
        sample_rate: f32,
        attention_level: f32,
    ) -> Vec<f32> {
        use std::f32::consts::PI;
        (0..n_samples)
            .map(|i| {
                let t = i as f32 / sample_rate;
                // High attention = more theta, less alpha
                let theta_weight = 0.2 + attention_level * 0.3;
                let alpha_weight = 0.4 - attention_level * 0.2;
                let beta_weight = 0.15 + attention_level * 0.1;

                theta_weight * (2.0 * PI * 6.0 * t).sin()
                    + alpha_weight * (2.0 * PI * 10.0 * t).sin()
                    + beta_weight * (2.0 * PI * 20.0 * t).sin()
                    + 0.05 * (2.0 * PI * 40.0 * t).sin()
            })
            .collect()
    }

    #[test]
    fn test_attention_detection() {
        let sentinel = AttentionSentinel::new();
        let sample_rate = 256.0;

        // Low attention signal
        let low_attention = generate_attention_signal(1280, sample_rate, 0.2);
        let score_low = sentinel.analyze(&low_attention, sample_rate);

        // High attention signal
        let high_attention = generate_attention_signal(1280, sample_rate, 0.8);
        let score_high = sentinel.analyze(&high_attention, sample_rate);

        // High attention should have higher index
        assert!(
            score_high.attention_index > score_low.attention_index,
            "High attention signal should have higher attention index"
        );
    }

    #[test]
    fn test_calibration() {
        let mut sentinel = AttentionSentinel::new();
        let sample_rate = 256.0;

        // Calibrate with baseline
        let baseline = generate_attention_signal(1280, sample_rate, 0.5);
        sentinel.calibrate(&baseline, sample_rate);

        assert!(sentinel.calibrated);
    }

    #[test]
    fn test_state_classification() {
        let sentinel = AttentionSentinel::new();

        assert_eq!(sentinel.classify_state(0.3), AttentionState::Distracted);
        assert_eq!(sentinel.classify_state(0.7), AttentionState::Normal);
        assert_eq!(sentinel.classify_state(1.5), AttentionState::Focused);
        assert_eq!(sentinel.classify_state(2.5), AttentionState::DeepFocus);
    }
}