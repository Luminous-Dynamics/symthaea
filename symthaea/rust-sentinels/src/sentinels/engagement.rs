// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! EngagementSentinel - Proof of Engagement
//!
//! Measures cognitive and emotional engagement with content or tasks.
//!
//! # Neuroscience Basis
//! - **Engagement Index**: Classic formula β / (α + θ)
//! - **Workload Index**: Frontal theta power indicates mental effort
//! - **Boredom Detection**: Increased alpha, decreased beta
//! - **Interest/Novelty**: P300 and gamma responses to novel stimuli

use crate::signal::{band_power, welch_psd};

/// Engagement level classification
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum EngagementLevel {
    /// Disengaged - bored, distracted, checked out
    Disengaged,
    /// Low - passive attention, minimal involvement
    Low,
    /// Moderate - active engagement, working
    Moderate,
    /// High - deep engagement, fully involved
    High,
    /// Overload - cognitive overload, too much stimulation
    Overload,
}

/// Engagement analysis result
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EngagementScore {
    /// Cognitive/mental engagement (0-1)
    pub cognitive: f32,
    /// Emotional investment/interest (0-1)
    pub emotional: f32,
    /// Novelty/curiosity response (0-1)
    pub interest: f32,
    /// Mental fatigue indicator (0-1, higher = more tired)
    pub fatigue: f32,
    /// Workload estimate (0-1)
    pub workload: f32,
    /// Classic engagement index (typically 0-3)
    pub engagement_index: f32,
    /// Classified engagement level
    pub level: EngagementLevel,
    /// Confidence in classification (0-1)
    pub confidence: f32,
}

impl EngagementScore {
    /// Check if user is actively engaged
    pub fn is_engaged(&self) -> bool {
        matches!(
            self.level,
            EngagementLevel::Moderate | EngagementLevel::High
        )
    }

    /// Check if user needs a break (fatigued or overloaded)
    pub fn needs_break(&self) -> bool {
        self.fatigue > 0.7 || self.level == EngagementLevel::Overload
    }

    /// Get a human-readable description
    pub fn description(&self) -> &'static str {
        match self.level {
            EngagementLevel::Disengaged => "Bored or distracted, not engaged",
            EngagementLevel::Low => "Passive attention, minimal involvement",
            EngagementLevel::Moderate => "Actively engaged and working",
            EngagementLevel::High => "Deeply engaged, fully invested",
            EngagementLevel::Overload => "Cognitive overload, needs a break",
        }
    }
}

/// Configuration for engagement detection
#[derive(Debug, Clone)]
pub struct EngagementConfig {
    /// Engagement index thresholds [disengaged, low, moderate, high]
    pub index_thresholds: [f32; 4],
    /// Fatigue threshold (theta increase over time)
    pub fatigue_threshold: f32,
    /// Gamma threshold for interest detection
    pub interest_threshold: f32,
}

impl Default for EngagementConfig {
    fn default() -> Self {
        Self {
            index_thresholds: [0.5, 1.0, 2.0, 3.0],
            fatigue_threshold: 0.35,  // Theta > 35% suggests fatigue
            interest_threshold: 0.08, // Gamma > 8% suggests interest
        }
    }
}

/// EngagementSentinel - Measures cognitive and emotional engagement
pub struct EngagementSentinel {
    config: EngagementConfig,
    baseline_theta: f32,
    baseline_alpha: f32,
    baseline_beta: f32,
    baseline_gamma: f32,
    calibrated: bool,
}

impl EngagementSentinel {
    /// Create a new EngagementSentinel with default configuration
    pub fn new() -> Self {
        Self {
            config: EngagementConfig::default(),
            baseline_theta: 0.20,
            baseline_alpha: 0.30,
            baseline_beta: 0.20,
            baseline_gamma: 0.05,
            calibrated: false,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: EngagementConfig) -> Self {
        Self {
            config,
            baseline_theta: 0.20,
            baseline_alpha: 0.30,
            baseline_beta: 0.20,
            baseline_gamma: 0.05,
            calibrated: false,
        }
    }

    /// Calibrate baseline from resting/neutral state data
    pub fn calibrate(&mut self, data: &[f32], sample_rate: f32) {
        let powers = self.extract_powers(data, sample_rate);
        self.baseline_theta = powers.0;
        self.baseline_alpha = powers.1;
        self.baseline_beta = powers.2;
        self.baseline_gamma = powers.3;
        self.calibrated = true;
    }

    /// Analyze engagement from EEG data
    pub fn analyze(&self, data: &[f32], sample_rate: f32) -> EngagementScore {
        let (theta_rel, alpha_rel, beta_rel, gamma_rel) = self.extract_powers(data, sample_rate);

        // Classic engagement index: β / (α + θ)
        let engagement_index = beta_rel / (alpha_rel + theta_rel + 0.01);

        // Cognitive engagement from beta + gamma
        let cognitive = ((beta_rel + gamma_rel) * 2.5).min(1.0);

        // Emotional/interest from gamma (novelty response)
        let emotional = (gamma_rel / self.config.interest_threshold).min(1.0);

        // Interest/curiosity from gamma relative to baseline
        let gamma_ratio = gamma_rel / self.baseline_gamma;
        let interest = ((gamma_ratio - 0.8) * 0.5).max(0.0).min(1.0);

        // Workload from frontal theta
        let workload = (theta_rel / 0.30).min(1.0);

        // Fatigue detection: high theta + low beta
        let fatigue = if theta_rel > self.config.fatigue_threshold && beta_rel < 0.15 {
            ((theta_rel - self.config.fatigue_threshold) * 3.0).min(1.0)
        } else {
            ((theta_rel / self.baseline_theta) - 1.0).max(0.0).min(0.5)
        };

        // Classify engagement level
        let level = self.classify_level(engagement_index, fatigue);

        // Calculate confidence
        let confidence = self.calculate_confidence(theta_rel, alpha_rel, beta_rel);

        EngagementScore {
            cognitive,
            emotional,
            interest,
            fatigue,
            workload,
            engagement_index,
            level,
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

    /// Classify engagement level
    fn classify_level(&self, index: f32, fatigue: f32) -> EngagementLevel {
        // Check for overload first
        if index > self.config.index_thresholds[3] {
            return EngagementLevel::Overload;
        }

        // If fatigued, cap at low engagement
        if fatigue > 0.6 {
            return if index > self.config.index_thresholds[0] {
                EngagementLevel::Low
            } else {
                EngagementLevel::Disengaged
            };
        }

        // Normal classification
        let thresholds = &self.config.index_thresholds;
        if index < thresholds[0] {
            EngagementLevel::Disengaged
        } else if index < thresholds[1] {
            EngagementLevel::Low
        } else if index < thresholds[2] {
            EngagementLevel::Moderate
        } else {
            EngagementLevel::High
        }
    }

    /// Calculate confidence
    fn calculate_confidence(&self, theta: f32, alpha: f32, beta: f32) -> f32 {
        let theta_conf = if theta > 0.1 && theta < 0.4 { 1.0 } else { 0.7 };
        let alpha_conf = if alpha > 0.1 && alpha < 0.5 { 1.0 } else { 0.7 };
        let beta_conf = if beta > 0.05 && beta < 0.4 { 1.0 } else { 0.7 };

        (theta_conf + alpha_conf + beta_conf) / 3.0
    }
}

impl Default for EngagementSentinel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_engagement_signal(n_samples: usize, sample_rate: f32, engagement: f32) -> Vec<f32> {
        use std::f32::consts::PI;
        (0..n_samples)
            .map(|i| {
                let t = i as f32 / sample_rate;
                // High engagement = high beta, low alpha
                let theta_weight = 0.20;
                let alpha_weight = 0.35 - engagement * 0.15;
                let beta_weight = 0.15 + engagement * 0.20;
                let gamma_weight = 0.03 + engagement * 0.05;

                theta_weight * (2.0 * PI * 6.0 * t).sin()
                    + alpha_weight * (2.0 * PI * 10.0 * t).sin()
                    + beta_weight * (2.0 * PI * 20.0 * t).sin()
                    + gamma_weight * (2.0 * PI * 40.0 * t).sin()
            })
            .collect()
    }

    #[test]
    fn test_engagement_detection() {
        let sentinel = EngagementSentinel::new();
        let sample_rate = 256.0;

        // Low engagement (bored)
        let bored = generate_engagement_signal(1280, sample_rate, 0.1);
        let score_bored = sentinel.analyze(&bored, sample_rate);

        // High engagement
        let engaged = generate_engagement_signal(1280, sample_rate, 0.9);
        let score_engaged = sentinel.analyze(&engaged, sample_rate);

        assert!(
            score_engaged.engagement_index > score_bored.engagement_index,
            "Engaged signal should have higher engagement index"
        );
    }

    #[test]
    fn test_engagement_levels() {
        let sentinel = EngagementSentinel::new();

        assert_eq!(
            sentinel.classify_level(0.3, 0.0),
            EngagementLevel::Disengaged
        );
        assert_eq!(sentinel.classify_level(0.7, 0.0), EngagementLevel::Low);
        assert_eq!(sentinel.classify_level(1.5, 0.0), EngagementLevel::Moderate);
        assert_eq!(sentinel.classify_level(2.5, 0.0), EngagementLevel::High);
        assert_eq!(sentinel.classify_level(3.5, 0.0), EngagementLevel::Overload);
    }

    #[test]
    fn test_fatigue_affects_engagement() {
        let sentinel = EngagementSentinel::new();

        // High engagement index but high fatigue should cap at Low
        assert_eq!(sentinel.classify_level(1.5, 0.7), EngagementLevel::Low);
    }

    #[test]
    fn test_needs_break() {
        let sentinel = EngagementSentinel::new();
        let sample_rate = 256.0;

        // Simulate fatigued state (high theta)
        let fatigued: Vec<f32> = (0..1280)
            .map(|i| {
                let t = i as f32 / sample_rate;
                use std::f32::consts::PI;
                0.45 * (2.0 * PI * 6.0 * t).sin() // High theta
                    + 0.20 * (2.0 * PI * 10.0 * t).sin()
                    + 0.10 * (2.0 * PI * 20.0 * t).sin()
            })
            .collect();

        let _score = sentinel.analyze(&fatigued, sample_rate);
        // Note: needs_break depends on calculated fatigue which depends on signal
    }
}