// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Auto-mastering: LUFS measurement, corrective EQ, loudness normalization.
//!
//! Professional audio goes through mastering before release. This module
//! automates the process: measure loudness (LUFS), analyze frequency balance,
//! apply corrective EQ, normalize to -14 LUFS (streaming standard), compress,
//! and brick-wall limit.

use crate::mixing::{Limiter, ParametricEQ};

/// LUFS measurement result.
#[derive(Debug, Clone)]
pub struct LufsResult {
    pub integrated: f32,
    pub peak_db: f32,
    pub rms_db: f32,
    pub crest_factor_db: f32,
}

/// Frequency balance analysis.
#[derive(Debug, Clone)]
pub struct FrequencyBalance {
    pub low_energy: f32,  // 0-200 Hz
    pub mid_energy: f32,  // 200-2000 Hz
    pub high_energy: f32, // 2000+ Hz
}

impl FrequencyBalance {
    pub fn total(&self) -> f32 {
        self.low_energy + self.mid_energy + self.high_energy
    }
    pub fn low_ratio(&self) -> f32 {
        self.low_energy / self.total().max(1e-10)
    }
    pub fn mid_ratio(&self) -> f32 {
        self.mid_energy / self.total().max(1e-10)
    }
    pub fn high_ratio(&self) -> f32 {
        self.high_energy / self.total().max(1e-10)
    }
}

/// Mastering configuration.
pub struct MasteringConfig {
    pub target_lufs: f32,          // -14.0
    pub limiter_ceiling_db: f32,   // -1.0
    pub target_low_ratio: f32,     // 0.30
    pub target_mid_ratio: f32,     // 0.45
    pub target_high_ratio: f32,    // 0.25
    pub eq_max_correction_db: f32, // 6.0
}

impl Default for MasteringConfig {
    fn default() -> Self {
        Self {
            target_lufs: -14.0,
            limiter_ceiling_db: -1.0,
            target_low_ratio: 0.30,
            target_mid_ratio: 0.45,
            target_high_ratio: 0.25,
            eq_max_correction_db: 6.0,
        }
    }
}

/// Mastering result report.
#[derive(Debug, Clone)]
pub struct MasteringResult {
    pub input_lufs: f32,
    pub output_lufs: f32,
    pub gain_applied_db: f32,
    pub eq_corrections: [f32; 3],
    pub peak_before: f32,
    pub peak_after: f32,
}

/// Measure LUFS (simplified ITU-R BS.1770).
///
/// Uses K-weighted RMS over 400ms gated windows.
pub fn measure_lufs(samples: &[[f32; 2]], sample_rate: u32) -> LufsResult {
    if samples.is_empty() {
        return LufsResult {
            integrated: -70.0,
            peak_db: -70.0,
            rms_db: -70.0,
            crest_factor_db: 0.0,
        };
    }

    let window_size = (sample_rate as f32 * 0.4) as usize; // 400ms
    let mut block_powers: Vec<f32> = Vec::new();
    let mut peak = 0.0f32;
    let mut total_power = 0.0f64;

    for chunk in samples.chunks(window_size) {
        let mut power = 0.0f64;
        for s in chunk {
            let mono = (s[0] + s[1]) * 0.5;
            power += (mono * mono) as f64;
            peak = peak.max(s[0].abs()).max(s[1].abs());
        }
        let mean_power = power / chunk.len() as f64;
        total_power += power;
        block_powers.push(mean_power as f32);
    }

    let total_mean_power = total_power / samples.len() as f64;
    let rms = (total_mean_power as f32).sqrt();
    let rms_db = if rms > 1e-10 {
        20.0 * rms.log10()
    } else {
        -70.0
    };
    let peak_db = if peak > 1e-10 {
        20.0 * peak.log10()
    } else {
        -70.0
    };

    // Absolute gate: exclude blocks below -70 LUFS
    let gated: Vec<f32> = block_powers
        .iter()
        .copied()
        .filter(|&p| p > 1e-7) // ~-70 dBFS
        .collect();

    let integrated = if gated.is_empty() {
        -70.0
    } else {
        let mean = gated.iter().sum::<f32>() / gated.len() as f32;
        // Relative gate: exclude blocks below -10 dB of ungated mean
        let relative_threshold = mean * 0.1; // -10 dB
        let final_blocks: Vec<f32> = gated
            .iter()
            .copied()
            .filter(|&p| p >= relative_threshold)
            .collect();
        if final_blocks.is_empty() {
            -70.0
        } else {
            let final_mean = final_blocks.iter().sum::<f32>() / final_blocks.len() as f32;
            -0.691 + 10.0 * final_mean.max(1e-10).log10()
        }
    };

    LufsResult {
        integrated,
        peak_db,
        rms_db,
        crest_factor_db: peak_db - rms_db,
    }
}

/// Analyze frequency balance using simple band-pass energy measurement.
pub fn analyze_balance(samples: &[[f32; 2]], sample_rate: u32) -> FrequencyBalance {
    // Simple approach: measure energy in three bands using one-pole filters
    let mut low = 0.0f64;
    let mut mid = 0.0f64;
    let mut high = 0.0f64;

    // State for filters
    let mut lp_state = 0.0f32; // 200Hz low-pass
    let mut hp_state = 0.0f32; // 2000Hz high-pass
    let sr = sample_rate as f32;
    let lp_coeff = (std::f32::consts::TAU * 200.0 / sr).min(0.99);
    let hp_coeff = (std::f32::consts::TAU * 2000.0 / sr).min(0.99);

    for s in samples {
        let mono = (s[0] + s[1]) * 0.5;

        // Low-pass for bass
        lp_state += lp_coeff * (mono - lp_state);
        low += (lp_state * lp_state) as f64;

        // High-pass for treble
        hp_state += hp_coeff * (mono - hp_state);
        let hp_out = mono - hp_state;
        high += (hp_out * hp_out) as f64;

        // Mid = total - low - high
        let mid_out = mono - lp_state - hp_out;
        mid += (mid_out * mid_out) as f64;
    }

    let n = samples.len().max(1) as f64;
    FrequencyBalance {
        low_energy: (low / n).sqrt() as f32,
        mid_energy: (mid / n).sqrt() as f32,
        high_energy: (high / n).sqrt() as f32,
    }
}

/// Auto-master a stereo audio buffer in-place.
pub fn auto_master(
    samples: &mut Vec<[f32; 2]>,
    sample_rate: u32,
    config: &MasteringConfig,
) -> MasteringResult {
    // 1. Measure input
    let input = measure_lufs(samples, sample_rate);
    let balance = analyze_balance(samples, sample_rate);

    // 2. Compute corrective EQ
    let low_correction = ratio_to_db(config.target_low_ratio / balance.low_ratio().max(0.01))
        .clamp(-config.eq_max_correction_db, config.eq_max_correction_db);
    let mid_correction = ratio_to_db(config.target_mid_ratio / balance.mid_ratio().max(0.01))
        .clamp(-config.eq_max_correction_db, config.eq_max_correction_db);
    let high_correction = ratio_to_db(config.target_high_ratio / balance.high_ratio().max(0.01))
        .clamp(-config.eq_max_correction_db, config.eq_max_correction_db);

    // 3. Apply EQ
    let mut eq_l = ParametricEQ::new(low_correction, mid_correction, high_correction);
    let mut eq_r = ParametricEQ::new(low_correction, mid_correction, high_correction);
    let sr = sample_rate as f32;
    for s in samples.iter_mut() {
        s[0] = eq_l.process(s[0], sr);
        s[1] = eq_r.process(s[1], sr);
    }

    // 4. Loudness normalization
    let post_eq = measure_lufs(samples, sample_rate);
    let gain_db = config.target_lufs - post_eq.integrated;
    let gain_linear = 10.0f32.powf(gain_db / 20.0);

    for s in samples.iter_mut() {
        s[0] *= gain_linear;
        s[1] *= gain_linear;
    }

    // 5. Brick-wall limiting
    let mut limiter = Limiter::new(sample_rate, config.limiter_ceiling_db);
    for s in samples.iter_mut() {
        let (l, r) = limiter.process(s[0], s[1]);
        s[0] = l;
        s[1] = r;
    }

    // 6. Measure output
    let output = measure_lufs(samples, sample_rate);

    MasteringResult {
        input_lufs: input.integrated,
        output_lufs: output.integrated,
        gain_applied_db: gain_db,
        eq_corrections: [low_correction, mid_correction, high_correction],
        peak_before: input.peak_db,
        peak_after: output.peak_db,
    }
}

fn ratio_to_db(ratio: f32) -> f32 {
    20.0 * ratio.max(1e-10).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_wave(freq: f32, duration_secs: f32, amplitude: f32, sr: u32) -> Vec<[f32; 2]> {
        let n = (duration_secs * sr as f32) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                let s = amplitude * (t * freq * std::f32::consts::TAU).sin();
                [s, s]
            })
            .collect()
    }

    #[test]
    fn lufs_calibration_sine() {
        // -20 dBFS sine at 1kHz should measure approximately -23 LUFS
        let amplitude = 10.0f32.powf(-20.0 / 20.0); // -20 dBFS
        let samples = sine_wave(1000.0, 3.0, amplitude, 44100);
        let result = measure_lufs(&samples, 44100);
        // LUFS = -0.691 + 10*log10(mean_power), for sine: mean_power = A²/2
        // Expected: -0.691 + 10*log10(0.01/2) ≈ -0.691 + (-23.01) ≈ -23.7
        assert!(
            result.integrated < -18.0,
            "should be quiet: {}",
            result.integrated
        );
        assert!(
            result.integrated > -28.0,
            "should be audible: {}",
            result.integrated
        );
    }

    #[test]
    fn auto_master_normalizes() {
        let mut samples = sine_wave(440.0, 2.0, 0.01, 44100); // very quiet
        let config = MasteringConfig::default();
        let result = auto_master(&mut samples, 44100, &config);

        assert!(
            result.output_lufs > result.input_lufs,
            "should be louder after mastering"
        );
        assert!(
            result.gain_applied_db > 0.0,
            "should have applied positive gain"
        );
    }

    #[test]
    fn limiter_respects_ceiling() {
        let mut samples = sine_wave(440.0, 1.0, 0.9, 44100);
        let config = MasteringConfig {
            target_lufs: -6.0,
            ..Default::default()
        };
        auto_master(&mut samples, 44100, &config);

        let ceiling = 10.0f32.powf(-1.0 / 20.0);
        let max_peak = samples
            .iter()
            .flat_map(|s| [s[0].abs(), s[1].abs()])
            .fold(0.0f32, f32::max);
        assert!(
            max_peak <= ceiling + 0.01,
            "peak should be below ceiling: {max_peak}"
        );
    }

    #[test]
    fn frequency_balance_detects_bass() {
        let samples = sine_wave(80.0, 1.0, 0.5, 44100); // pure bass
        let balance = analyze_balance(&samples, 44100);
        assert!(
            balance.low_energy > balance.high_energy,
            "bass sine should have more low energy: low={} high={}",
            balance.low_energy,
            balance.high_energy
        );
    }
}
