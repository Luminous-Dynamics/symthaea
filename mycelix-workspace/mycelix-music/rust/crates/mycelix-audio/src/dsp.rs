// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Digital Signal Processing utilities

use crate::{AudioBuffer, AudioError, AudioResult};

/// Biquad filter types
#[derive(Debug, Clone, Copy)]
pub enum FilterType {
    LowPass,
    HighPass,
    BandPass,
    Notch,
    LowShelf,
    HighShelf,
    Peaking,
}

/// Biquad filter coefficients
#[derive(Debug, Clone)]
pub struct BiquadCoeffs {
    pub b0: f32,
    pub b1: f32,
    pub b2: f32,
    pub a1: f32,
    pub a2: f32,
}

impl BiquadCoeffs {
    /// Create low-pass filter coefficients
    pub fn low_pass(sample_rate: f32, frequency: f32, q: f32) -> Self {
        let omega = 2.0 * std::f32::consts::PI * frequency / sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * q);

        let b0 = (1.0 - cos_omega) / 2.0;
        let b1 = 1.0 - cos_omega;
        let b2 = (1.0 - cos_omega) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Create high-pass filter coefficients
    pub fn high_pass(sample_rate: f32, frequency: f32, q: f32) -> Self {
        let omega = 2.0 * std::f32::consts::PI * frequency / sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * q);

        let b0 = (1.0 + cos_omega) / 2.0;
        let b1 = -(1.0 + cos_omega);
        let b2 = (1.0 + cos_omega) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Create band-pass filter coefficients
    pub fn band_pass(sample_rate: f32, frequency: f32, q: f32) -> Self {
        let omega = 2.0 * std::f32::consts::PI * frequency / sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * q);

        let b0 = alpha;
        let b1 = 0.0;
        let b2 = -alpha;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }
}

/// Biquad filter processor
#[derive(Debug, Clone)]
pub struct BiquadFilter {
    coeffs: BiquadCoeffs,
    // State variables per channel
    x1: Vec<f32>,
    x2: Vec<f32>,
    y1: Vec<f32>,
    y2: Vec<f32>,
}

impl BiquadFilter {
    pub fn new(coeffs: BiquadCoeffs, channels: usize) -> Self {
        Self {
            coeffs,
            x1: vec![0.0; channels],
            x2: vec![0.0; channels],
            y1: vec![0.0; channels],
            y2: vec![0.0; channels],
        }
    }

    pub fn process_sample(&mut self, sample: f32, channel: usize) -> f32 {
        let output = self.coeffs.b0 * sample
            + self.coeffs.b1 * self.x1[channel]
            + self.coeffs.b2 * self.x2[channel]
            - self.coeffs.a1 * self.y1[channel]
            - self.coeffs.a2 * self.y2[channel];

        self.x2[channel] = self.x1[channel];
        self.x1[channel] = sample;
        self.y2[channel] = self.y1[channel];
        self.y1[channel] = output;

        output
    }

    pub fn process(&mut self, buffer: &mut AudioBuffer) -> AudioResult<()> {
        let channels = buffer.channels;
        for (i, sample) in buffer.samples.iter_mut().enumerate() {
            let channel = i % channels;
            *sample = self.process_sample(*sample, channel);
        }
        Ok(())
    }

    pub fn reset(&mut self) {
        self.x1.fill(0.0);
        self.x2.fill(0.0);
        self.y1.fill(0.0);
        self.y2.fill(0.0);
    }
}

/// Window function types
#[derive(Debug, Clone, Copy)]
pub enum WindowType {
    Rectangular,
    Hann,
    Hamming,
    Blackman,
    BlackmanHarris,
}

/// Generate window function
pub fn generate_window(window_type: WindowType, size: usize) -> Vec<f32> {
    let n = size as f32;
    match window_type {
        WindowType::Rectangular => vec![1.0; size],
        WindowType::Hann => {
            (0..size)
                .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (n - 1.0)).cos()))
                .collect()
        }
        WindowType::Hamming => {
            (0..size)
                .map(|i| 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (n - 1.0)).cos())
                .collect()
        }
        WindowType::Blackman => {
            (0..size)
                .map(|i| {
                    let x = i as f32 / (n - 1.0);
                    0.42 - 0.5 * (2.0 * std::f32::consts::PI * x).cos()
                        + 0.08 * (4.0 * std::f32::consts::PI * x).cos()
                })
                .collect()
        }
        WindowType::BlackmanHarris => {
            (0..size)
                .map(|i| {
                    let x = i as f32 / (n - 1.0);
                    0.35875 - 0.48829 * (2.0 * std::f32::consts::PI * x).cos()
                        + 0.14128 * (4.0 * std::f32::consts::PI * x).cos()
                        - 0.01168 * (6.0 * std::f32::consts::PI * x).cos()
                })
                .collect()
        }
    }
}

/// Apply window to samples
pub fn apply_window(samples: &mut [f32], window: &[f32]) {
    for (sample, &w) in samples.iter_mut().zip(window.iter()) {
        *sample *= w;
    }
}

/// Convert linear amplitude to decibels
pub fn linear_to_db(linear: f32) -> f32 {
    if linear > 0.0 {
        20.0 * linear.log10()
    } else {
        -96.0 // Floor
    }
}

/// Convert decibels to linear amplitude
pub fn db_to_linear(db: f32) -> f32 {
    10.0f32.powf(db / 20.0)
}

/// Convert frequency to MIDI note number
pub fn freq_to_midi(freq: f32) -> f32 {
    69.0 + 12.0 * (freq / 440.0).log2()
}

/// Convert MIDI note number to frequency
pub fn midi_to_freq(midi: f32) -> f32 {
    440.0 * 2.0f32.powf((midi - 69.0) / 12.0)
}

/// Simple envelope follower
#[derive(Debug, Clone)]
pub struct EnvelopeFollower {
    attack_coeff: f32,
    release_coeff: f32,
    envelope: f32,
}

impl EnvelopeFollower {
    pub fn new(sample_rate: f32, attack_ms: f32, release_ms: f32) -> Self {
        Self {
            attack_coeff: (-1.0 / (attack_ms * sample_rate / 1000.0)).exp(),
            release_coeff: (-1.0 / (release_ms * sample_rate / 1000.0)).exp(),
            envelope: 0.0,
        }
    }

    pub fn process(&mut self, input: f32) -> f32 {
        let input_abs = input.abs();
        let coeff = if input_abs > self.envelope {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.envelope = coeff * self.envelope + (1.0 - coeff) * input_abs;
        self.envelope
    }

    pub fn reset(&mut self) {
        self.envelope = 0.0;
    }
}

/// DC blocker filter
#[derive(Debug, Clone)]
pub struct DcBlocker {
    x_prev: f32,
    y_prev: f32,
    coeff: f32,
}

impl DcBlocker {
    pub fn new() -> Self {
        Self {
            x_prev: 0.0,
            y_prev: 0.0,
            coeff: 0.995, // Standard DC blocking coefficient
        }
    }

    pub fn process(&mut self, input: f32) -> f32 {
        let output = input - self.x_prev + self.coeff * self.y_prev;
        self.x_prev = input;
        self.y_prev = output;
        output
    }

    pub fn reset(&mut self) {
        self.x_prev = 0.0;
        self.y_prev = 0.0;
    }
}

impl Default for DcBlocker {
    fn default() -> Self {
        Self::new()
    }
}
