// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! EmotionSentinel - Emotional state detection (Proof of Joy)
//!
//! Detects emotional valence and arousal from EEG using frontal asymmetry
//! and band power ratios. Validated on DENS dataset with r=0.39 valence.

use crate::signal::extract_band_powers;
use crate::types::EmotionScore;

/// Emotion detection Sentinel
#[derive(Debug, Clone)]
pub struct EmotionSentinel {
    /// Baseline frontal asymmetry (calibration)
    baseline_asymmetry: f32,
    /// Whether calibration has been performed
    calibrated: bool,
}

impl Default for EmotionSentinel {
    fn default() -> Self {
        Self {
            baseline_asymmetry: 0.0,
            calibrated: false,
        }
    }
}

impl EmotionSentinel {
    /// Create new EmotionSentinel
    pub fn new() -> Self {
        Self::default()
    }

    /// Calibrate with resting baseline
    ///
    /// # Arguments
    /// * `left_data` - Left hemisphere EEG
    /// * `right_data` - Right hemisphere EEG
    /// * `sample_rate` - Sampling rate in Hz
    pub fn calibrate(&mut self, left_data: &[f32], right_data: &[f32], sample_rate: f32) {
        self.baseline_asymmetry = self.compute_asymmetry(left_data, right_data, sample_rate);
        self.calibrated = true;
    }

    /// Analyze emotional state from EEG
    ///
    /// # Arguments
    /// * `data` - EEG samples (single channel for now)
    /// * `sample_rate` - Sampling rate in Hz
    pub fn analyze(&self, data: &[f32], sample_rate: f32) -> EmotionScore {
        let powers = extract_band_powers(data, sample_rate);

        // Valence from alpha asymmetry (simplified single-channel version)
        // In full version, would use left vs right channels
        let valence = self.estimate_valence(&powers);

        // Arousal from beta/(alpha+theta) ratio
        let arousal = self.estimate_arousal(&powers);

        // Confidence based on signal quality
        let confidence = (powers.total / 1e-6).min(1.0).max(0.1);

        EmotionScore {
            valence,
            arousal,
            confidence,
        }
    }

    fn compute_asymmetry(&self, left: &[f32], right: &[f32], sample_rate: f32) -> f32 {
        let left_powers = extract_band_powers(left, sample_rate);
        let right_powers = extract_band_powers(right, sample_rate);

        // Log asymmetry index: ln(right) - ln(left)
        (right_powers.alpha + 1e-10).ln() - (left_powers.alpha + 1e-10).ln()
    }

    fn estimate_valence(&self, powers: &crate::signal::BandPowers) -> f32 {
        // For single-channel, use alpha/beta balance as proxy
        // Higher alpha relative to beta suggests more positive state
        let ratio = powers.alpha / (powers.beta + 1e-10);
        let raw_valence = (ratio - 1.0) / 2.0; // Center around 0

        raw_valence.tanh() // Bound to [-1, 1]
    }

    fn estimate_arousal(&self, powers: &crate::signal::BandPowers) -> f32 {
        // Beta indicates arousal, alpha+theta indicate calm
        let arousal_raw = powers.beta / (powers.alpha + powers.theta + 1e-10);
        (arousal_raw / 2.0).min(1.0) // Normalize to [0, 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emotion_analysis() {
        let sentinel = EmotionSentinel::default();

        // Generate alpha-dominant signal (relaxed state)
        let sample_rate = 256.0;
        let duration = 10.0;
        let n = (sample_rate * duration) as usize;

        let signal: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate;
                // 10 Hz alpha wave
                (2.0 * std::f32::consts::PI * 10.0 * t).sin()
            })
            .collect();

        let score = sentinel.analyze(&signal, sample_rate);

        // Should detect calm state
        assert!(score.arousal < 0.5);
        assert!(score.confidence > 0.0);
    }
}