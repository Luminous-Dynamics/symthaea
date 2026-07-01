// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Signal processing utilities for EEG analysis

mod welch;

pub use welch::{band_power, welch_psd};

/// Standard EEG frequency bands
pub mod bands {
    /// Delta band (0.5-4 Hz) - Deep sleep
    pub const DELTA: (f32, f32) = (0.5, 4.0);
    /// Theta band (4-8 Hz) - Drowsiness, light sleep
    pub const THETA: (f32, f32) = (4.0, 8.0);
    /// Alpha band (8-13 Hz) - Relaxed wakefulness
    pub const ALPHA: (f32, f32) = (8.0, 13.0);
    /// Beta band (13-30 Hz) - Active thinking
    pub const BETA: (f32, f32) = (13.0, 30.0);
    /// Gamma band (30-45 Hz) - Higher cognition
    pub const GAMMA: (f32, f32) = (30.0, 45.0);
}

/// Extract all standard band powers from signal
pub fn extract_band_powers(data: &[f32], sample_rate: f32) -> BandPowers {
    let nperseg = (4.0 * sample_rate).min(data.len() as f32) as usize;
    let (freqs, psd) = welch_psd(data, sample_rate, nperseg);

    let delta = band_power(&freqs, &psd, bands::DELTA.0, bands::DELTA.1);
    let theta = band_power(&freqs, &psd, bands::THETA.0, bands::THETA.1);
    let alpha = band_power(&freqs, &psd, bands::ALPHA.0, bands::ALPHA.1);
    let beta = band_power(&freqs, &psd, bands::BETA.0, bands::BETA.1);
    let gamma = band_power(&freqs, &psd, bands::GAMMA.0, bands::GAMMA.1);

    let total = delta + theta + alpha + beta + gamma + 1e-10;

    BandPowers {
        delta: delta / total,
        theta: theta / total,
        alpha: alpha / total,
        beta: beta / total,
        gamma: gamma / total,
        total,
    }
}

/// Relative power in each frequency band
#[derive(Debug, Clone)]
pub struct BandPowers {
    /// Delta (0.5-4 Hz) relative power
    pub delta: f32,
    /// Theta (4-8 Hz) relative power
    pub theta: f32,
    /// Alpha (8-13 Hz) relative power
    pub alpha: f32,
    /// Beta (13-30 Hz) relative power
    pub beta: f32,
    /// Gamma (30-45 Hz) relative power
    pub gamma: f32,
    /// Total power (for denormalization)
    pub total: f32,
}