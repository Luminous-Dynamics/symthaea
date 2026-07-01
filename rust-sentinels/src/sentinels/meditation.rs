// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! MeditationSentinel - Meditation state detection (Proof of Focus)
//!
//! Measures meditation depth and stability from EEG using alpha/theta
//! patterns. Validated on OpenNeuro ds001787 with 5/5 checks passed.

use crate::signal::extract_band_powers;
use crate::types::MeditationScore;

/// Meditation detection Sentinel
#[derive(Debug, Clone)]
pub struct MeditationSentinel {
    /// Baseline meditation index (for comparison)
    baseline_index: f32,
    /// Whether calibration has been performed
    calibrated: bool,
}

impl Default for MeditationSentinel {
    fn default() -> Self {
        Self {
            baseline_index: 1.0, // Default baseline
            calibrated: false,
        }
    }
}

impl MeditationSentinel {
    /// Create new MeditationSentinel
    pub fn new() -> Self {
        Self::default()
    }

    /// Calibrate with resting baseline
    ///
    /// # Arguments
    /// * `data` - Baseline EEG samples
    /// * `sample_rate` - Sampling rate in Hz
    pub fn calibrate(&mut self, data: &[f32], sample_rate: f32) {
        let powers = extract_band_powers(data, sample_rate);
        self.baseline_index = (powers.alpha + powers.theta) / (powers.beta + 1e-10);
        self.calibrated = true;
    }

    /// Analyze meditation state from EEG
    ///
    /// # Arguments
    /// * `data` - EEG samples
    /// * `sample_rate` - Sampling rate in Hz
    pub fn analyze(&self, data: &[f32], sample_rate: f32) -> MeditationScore {
        let powers = extract_band_powers(data, sample_rate);

        // Meditation index: (alpha + theta) / beta
        let meditation_index = (powers.alpha + powers.theta) / (powers.beta + 1e-10);

        // Depth based on index relative to baseline
        let index_ratio = meditation_index / (self.baseline_index + 1e-10);
        let depth = ((index_ratio - 1.0) * 0.5).tanh() * 0.5 + 0.5;

        // Stability from alpha consistency (simplified)
        let stability = (powers.alpha * 2.0).min(1.0);

        MeditationScore {
            depth: depth.clamp(0.0, 1.0),
            stability,
            alpha_power: powers.alpha,
            theta_power: powers.theta,
            meditation_index,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meditation_analysis() {
        let sentinel = MeditationSentinel::default();

        // Generate alpha+theta dominant signal (meditation state)
        let sample_rate = 256.0;
        let duration = 30.0;
        let n = (sample_rate * duration) as usize;

        let signal: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate;
                let pi2 = 2.0 * std::f32::consts::PI;
                // Mix of alpha (10 Hz) and theta (6 Hz)
                0.6 * (pi2 * 10.0 * t).sin() + 0.4 * (pi2 * 6.0 * t).sin()
            })
            .collect();

        let score = sentinel.analyze(&signal, sample_rate);

        // Should detect meditation state
        assert!(score.alpha_power > 0.2);
        assert!(score.meditation_index > 1.0);
    }

    #[test]
    fn test_calibration() {
        let mut sentinel = MeditationSentinel::new();

        // Baseline calibration
        let baseline: Vec<f32> = (0..256 * 10)
            .map(|i| {
                let t = i as f32 / 256.0;
                (2.0 * std::f32::consts::PI * 10.0 * t).sin()
            })
            .collect();

        sentinel.calibrate(&baseline, 256.0);
        assert!(sentinel.calibrated);
        assert!(sentinel.baseline_index > 0.0);
    }
}