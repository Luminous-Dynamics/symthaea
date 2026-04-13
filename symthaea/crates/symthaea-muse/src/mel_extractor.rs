// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Mel-spectrogram extractor for HDC audio training.
//!
//! Converts raw PCM audio into log-mel spectrogram frames that can be paired
//! with HDC MusicalState vectors to train a neural audio decoder.
//!
//! # Pipeline
//!
//! ```text
//! PCM samples → Hann window → rustfft → magnitude spectrum →
//! mel filterbank → log → mel frames
//! ```
//!
//! Output: `Vec<Vec<f32>>` — one frame per hop_size samples, each frame is
//! `n_mels` dimensional log-mel energy.
//!
//! # Standard parameters for music (from MusicGen/AudioLDM conventions):
//! - sample_rate: 44100 Hz
//! - n_fft: 2048
//! - hop_length: 512 (11.6ms at 44.1kHz)
//! - n_mels: 128
//! - f_min: 20 Hz
//! - f_max: 16000 Hz (below Nyquist)
//!
//! # References
//! - Slaney, M. (1998). Auditory toolbox.
//! - Stevens, Volkmann, Newman (1937). A scale for the measurement of the
//!   psychological magnitude pitch.

use rustfft::{num_complex::Complex, FftPlanner};

/// Parameters for mel-spectrogram extraction.
#[derive(Debug, Clone)]
pub struct MelConfig {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    pub f_min: f32,
    pub f_max: f32,
}

impl Default for MelConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            n_fft: 2048,
            hop_length: 512,
            n_mels: 128,
            f_min: 20.0,
            f_max: 16000.0,
        }
    }
}

/// Mel-spectrogram extractor with precomputed filterbank.
pub struct MelExtractor {
    config: MelConfig,
    fft_planner: FftPlanner<f32>,
    hann_window: Vec<f32>,
    mel_filterbank: Vec<Vec<f32>>, // [n_mels][n_fft/2 + 1]
}

impl MelExtractor {
    pub fn new(config: MelConfig) -> Self {
        let hann_window: Vec<f32> = (0..config.n_fft)
            .map(|i| {
                0.5 * (1.0
                    - (2.0 * std::f32::consts::PI * i as f32 / (config.n_fft - 1) as f32).cos())
            })
            .collect();

        let mel_filterbank = build_mel_filterbank(&config);

        Self {
            config,
            fft_planner: FftPlanner::new(),
            hann_window,
            mel_filterbank,
        }
    }

    /// Extract log-mel spectrogram from mono audio samples.
    ///
    /// Returns `Vec<Vec<f32>>` where outer dimension is time (frames) and
    /// inner is frequency (mel bins).
    pub fn extract(&mut self, samples: &[f32]) -> Vec<Vec<f32>> {
        let n_fft = self.config.n_fft;
        let hop = self.config.hop_length;
        let n_mels = self.config.n_mels;

        if samples.len() < n_fft {
            return Vec::new();
        }

        let fft = self.fft_planner.plan_fft_forward(n_fft);
        let mut frames = Vec::new();

        let mut pos = 0;
        while pos + n_fft <= samples.len() {
            // Window the frame
            let mut buffer: Vec<Complex<f32>> = samples[pos..pos + n_fft]
                .iter()
                .zip(&self.hann_window)
                .map(|(&s, &w)| Complex { re: s * w, im: 0.0 })
                .collect();

            // Forward FFT
            fft.process(&mut buffer);

            // Magnitude spectrum (first half only, real input)
            let mag: Vec<f32> = buffer[..n_fft / 2 + 1]
                .iter()
                .map(|c| (c.re * c.re + c.im * c.im).sqrt())
                .collect();

            // Apply mel filterbank
            let mut mel_frame = vec![0.0f32; n_mels];
            for (m, filter) in self.mel_filterbank.iter().enumerate() {
                let mut energy = 0.0_f32;
                for (bin, &weight) in filter.iter().enumerate() {
                    energy += mag[bin] * weight;
                }
                // Log compression (add small epsilon to avoid log(0))
                mel_frame[m] = (energy + 1e-6).ln();
            }

            frames.push(mel_frame);
            pos += hop;
        }

        frames
    }

    /// Get the config used to build this extractor.
    pub fn config(&self) -> &MelConfig {
        &self.config
    }
}

/// Build a mel filterbank: triangular filters spaced uniformly on the mel scale.
fn build_mel_filterbank(config: &MelConfig) -> Vec<Vec<f32>> {
    let n_freqs = config.n_fft / 2 + 1;
    let n_mels = config.n_mels;

    // Convert min/max to mel scale
    let mel_min = hz_to_mel(config.f_min);
    let mel_max = hz_to_mel(config.f_max);

    // Equally spaced mel points (with 2 boundary points for filter edges)
    let mel_points: Vec<f32> = (0..n_mels + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();

    // Convert back to Hz, then to FFT bin indices
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let bin_points: Vec<f32> = hz_points
        .iter()
        .map(|&hz| hz * config.n_fft as f32 / config.sample_rate as f32)
        .collect();

    // Build triangular filters
    let mut filterbank = vec![vec![0.0f32; n_freqs]; n_mels];
    for m in 0..n_mels {
        let lower = bin_points[m];
        let center = bin_points[m + 1];
        let upper = bin_points[m + 2];

        for bin in 0..n_freqs {
            let bin_f = bin as f32;
            let weight = if bin_f < lower || bin_f > upper {
                0.0
            } else if bin_f < center {
                (bin_f - lower) / (center - lower).max(1e-6)
            } else {
                (upper - bin_f) / (upper - center).max(1e-6)
            };
            filterbank[m][bin] = weight;
        }
    }

    filterbank
}

/// Hz to mel scale (Slaney formula).
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Mel to Hz scale (inverse of Slaney formula).
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hz_mel_roundtrip() {
        for hz in [20.0, 100.0, 440.0, 1000.0, 4000.0, 16000.0] {
            let mel = hz_to_mel(hz);
            let hz_back = mel_to_hz(mel);
            assert!(
                (hz - hz_back).abs() < 0.01,
                "roundtrip failed: {} → {} → {}",
                hz, mel, hz_back
            );
        }
    }

    #[test]
    fn mel_scale_monotonic() {
        let mels: Vec<f32> = (0..100).map(|i| hz_to_mel(i as f32 * 200.0)).collect();
        for w in mels.windows(2) {
            assert!(w[1] > w[0], "mel scale should be monotonic");
        }
    }

    #[test]
    fn extract_sine_produces_peak() {
        let config = MelConfig::default();
        let mut extractor = MelExtractor::new(config.clone());

        // Generate 1 second of 440 Hz sine
        let samples: Vec<f32> = (0..config.sample_rate)
            .map(|i| {
                (i as f32 * 440.0 * std::f32::consts::TAU / config.sample_rate as f32).sin() * 0.5
            })
            .collect();

        let frames = extractor.extract(&samples);
        assert!(!frames.is_empty(), "should produce frames");
        assert_eq!(frames[0].len(), config.n_mels);

        // The mel bin containing 440 Hz should have highest energy
        let target_mel = hz_to_mel(440.0);
        let mel_range = hz_to_mel(config.f_max) - hz_to_mel(config.f_min);
        let target_bin = (((target_mel - hz_to_mel(config.f_min)) / mel_range) * config.n_mels as f32) as usize;

        let frame = &frames[frames.len() / 2]; // middle frame
        let max_bin = (0..config.n_mels)
            .max_by(|&a, &b| frame[a].partial_cmp(&frame[b]).unwrap())
            .unwrap();

        // Allow ±3 bins tolerance for mel filter spread
        assert!(
            (max_bin as i32 - target_bin as i32).abs() <= 3,
            "440Hz should peak near bin {}, found bin {}",
            target_bin, max_bin
        );
    }

    #[test]
    fn filterbank_covers_range() {
        let config = MelConfig::default();
        let fb = build_mel_filterbank(&config);
        assert_eq!(fb.len(), config.n_mels);
        for filter in &fb {
            assert_eq!(filter.len(), config.n_fft / 2 + 1);
            // Each filter should have at least one non-zero weight
            assert!(filter.iter().any(|&w| w > 0.0));
        }
    }
}
