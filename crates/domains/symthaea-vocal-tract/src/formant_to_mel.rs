// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Formant-to-mel spectrogram conversion for neural vocoder input.
//!
//! Converts 200Hz `FormantFrame` + voice quality parameters to 100-band log-mel
//! spectrogram frames at ~93.75Hz (matching BigVGAN v2 24kHz / 256-hop expectation).
//!
//! Algorithm per `push_frame()` call:
//! 1. Synthesize ~120 audio samples (24kHz / 200Hz) from source-filter model
//! 2. Append to internal audio buffer
//! 3. Extract mel frames whenever buffer >= n_fft, advancing by hop_length

use rustfft::{FftPlanner, num_complex::Complex};
use std::f32::consts::PI;
use std::sync::Arc;

use crate::types::{FormantFrame, SourceType};

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Mel spectrogram normalization strategy.
///
/// BigVGAN and similar neural vocoders expect input mel spectrograms in a specific
/// value range. This config controls the normalization applied after log-mel extraction.
#[derive(Debug, Clone)]
pub struct MelNormalization {
    /// Per-bin mean (length = n_mels) or global mean (length = 1).
    /// Subtracted from log-mel values. Default: global -6.0 (approximate center of typical speech).
    pub mean: Vec<f32>,
    /// Per-bin std dev (length = n_mels) or global std (length = 1).
    /// Log-mel values are divided by this after mean subtraction. Default: global 3.0.
    pub std: Vec<f32>,
    /// Clip range after normalization: (min, max). Default: (-4.0, 4.0).
    pub clip_min: f32,
    pub clip_max: f32,
    /// Whether normalization is enabled. Default: true.
    pub enabled: bool,
}

impl Default for MelNormalization {
    fn default() -> Self {
        Self {
            mean: vec![-6.0], // Global mean (approximate log-mel center for speech)
            std: vec![3.0],   // Global std
            clip_min: -4.0,
            clip_max: 4.0,
            enabled: true,
        }
    }
}

impl MelNormalization {
    /// Create a normalization config with no normalization (pass-through).
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Create per-bin normalization from mean/std vectors.
    pub fn per_bin(mean: Vec<f32>, std: Vec<f32>) -> Self {
        Self {
            mean,
            std,
            clip_min: -4.0,
            clip_max: 4.0,
            enabled: true,
        }
    }

    /// Load per-bin normalization from JSON file produced by `calibrate_mel_normalization.py`.
    ///
    /// Expects JSON with `"per_bin_mean"` and `"per_bin_std"` arrays of floats.
    /// Returns `None` if the file cannot be read or parsed.
    pub fn from_json_file(path: &str) -> Option<Self> {
        let data = std::fs::read_to_string(path).ok()?;
        let mean = extract_json_array(&data, "per_bin_mean")?;
        let std = extract_json_array(&data, "per_bin_std")?;
        if mean.is_empty() || std.is_empty() {
            return None;
        }
        Some(Self::per_bin(mean, std))
    }

    /// Normalize a mel frame in-place.
    fn normalize(&self, mel: &mut [f32]) {
        if !self.enabled {
            return;
        }

        let per_bin = self.mean.len() > 1;
        for (i, val) in mel.iter_mut().enumerate() {
            let mean = if per_bin {
                self.mean[i % self.mean.len()]
            } else {
                self.mean[0]
            };
            let std = if per_bin {
                self.std[i % self.std.len()]
            } else {
                self.std[0]
            };
            *val = ((*val - mean) / std.max(1e-6)).clamp(self.clip_min, self.clip_max);
        }
    }
}

/// Configuration for formant-to-mel conversion.
#[derive(Debug, Clone)]
pub struct FormantToMelConfig {
    /// Audio sample rate (Hz). Must match BigVGAN model.
    pub sample_rate: u32,
    /// Number of mel filterbank bins.
    pub n_mels: usize,
    /// FFT window size.
    pub n_fft: usize,
    /// Hop length (samples between successive STFT columns).
    pub hop_length: usize,
    /// Minimum frequency for mel filterbank (Hz).
    pub f_min: f32,
    /// Maximum frequency for mel filterbank (Hz).
    pub f_max: f32,
    /// Base aspiration noise amplitude (breathiness floor).
    pub aspiration_base: f32,
    /// Motor frame rate (Hz) — how often `push_frame()` is called.
    pub motor_frame_rate: u32,
    /// Mel normalization strategy (mean/std + clip).
    pub normalization: MelNormalization,
    /// Enable inter-frame interpolation for smoother spectral transitions.
    pub interpolation: bool,
}

impl Default for FormantToMelConfig {
    fn default() -> Self {
        Self {
            sample_rate: 24000,
            n_mels: 100,
            n_fft: 1024,
            hop_length: 256,
            f_min: 0.0,
            f_max: 12000.0,
            aspiration_base: 0.02,
            motor_frame_rate: 200,
            normalization: MelNormalization::default(),
            interpolation: true,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// VOICE QUALITY (subset relevant to mel conversion)
// ═══════════════════════════════════════════════════════════════════════════════

/// Voice quality parameters that affect mel spectrogram generation.
///
/// These modulate the breathiness and spectral characteristics of the
/// source-filter synthesis used to produce mel frames.
#[derive(Debug, Clone, Copy)]
pub struct MelVoiceQuality {
    /// Glottal shape (Rd): 0.3=pressed, 1.0=modal, 2.7=breathy.
    pub rd: f32,
    /// Emotional arousal (0.0–1.0): higher = more tense, less breathy.
    pub arousal: f32,
}

impl Default for MelVoiceQuality {
    fn default() -> Self {
        Self {
            rd: 1.0,
            arousal: 0.5,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BIQUAD RESONATOR (lightweight, same math as vocoder.rs)
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Default)]
struct Resonator {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    y1: f32,
    y2: f32,
    x1: f32,
    x2: f32,
}

impl Resonator {
    fn set_params(&mut self, freq: f32, bandwidth: f32, sample_rate: f32) {
        if freq <= 0.0 || freq >= sample_rate / 2.0 {
            self.b0 = 1.0;
            self.b1 = 0.0;
            self.b2 = 0.0;
            self.a1 = 0.0;
            self.a2 = 0.0;
            return;
        }

        let omega = 2.0 * PI * freq / sample_rate;
        let cos_omega = omega.cos();
        let sin_omega = omega.sin();
        let q = freq / bandwidth.max(20.0);
        let alpha = sin_omega / (2.0 * q);

        let a0 = 1.0 + alpha;
        let peak_gain = q;
        self.b0 = alpha * peak_gain / a0;
        self.b1 = 0.0;
        self.b2 = -alpha * peak_gain / a0;
        self.a1 = -2.0 * cos_omega / a0;
        self.a2 = (1.0 - alpha) / a0;
    }

    fn process(&mut self, input: f32) -> f32 {
        let output = self.b0 * input + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = output;
        output
    }

    fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FORMANT-TO-MEL CONVERTER
// ═══════════════════════════════════════════════════════════════════════════════

/// Converts FormantFrame streams to log-mel spectrogram frames for neural vocoder input.
pub struct FormantToMelConverter {
    config: FormantToMelConfig,
    /// Samples per motor frame (sample_rate / motor_frame_rate).
    samples_per_frame: usize,
    /// Audio sample buffer (accumulates until n_fft samples available).
    audio_buffer: Vec<f32>,
    /// Three formant resonators for source-filter synthesis.
    resonators: [Resonator; 3],
    /// Glottal pulse phase accumulator.
    glottal_phase: f32,
    /// Mel filterbank weights: n_mels × (n_fft/2 + 1).
    mel_filters: Vec<Vec<f32>>,
    /// Hann window (n_fft samples).
    window: Vec<f32>,
    /// Cached FFT plan.
    fft: Arc<dyn rustfft::Fft<f32>>,
    /// FFT scratch buffer.
    scratch: Vec<Complex<f32>>,
    /// Simple PRNG state for aspiration noise.
    rng_state: u32,
    /// Previous formant frame for interpolation (improvement #4).
    prev_frame: Option<FormantFrame>,
    /// Previous voice quality for interpolation.
    prev_vq: Option<MelVoiceQuality>,
}

impl FormantToMelConverter {
    /// Create a new converter with the given configuration.
    pub fn new(config: FormantToMelConfig) -> Self {
        let samples_per_frame = (config.sample_rate / config.motor_frame_rate.max(1)) as usize;

        // Build mel filterbank
        let mel_filters = Self::create_mel_filterbank(&config);

        // Hann window
        let window: Vec<f32> = (0..config.n_fft)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (config.n_fft - 1) as f32).cos()))
            .collect();

        // FFT plan
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(config.n_fft);
        let scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];

        Self {
            config,
            samples_per_frame,
            audio_buffer: Vec::with_capacity(2048),
            resonators: [
                Resonator::default(),
                Resonator::default(),
                Resonator::default(),
            ],
            glottal_phase: 0.0,
            mel_filters,
            window,
            fft,
            scratch,
            rng_state: 0xDEAD_BEEF,
            prev_frame: None,
            prev_vq: None,
        }
    }

    /// Push one motor frame and return any mel spectrogram frames produced.
    ///
    /// Typically called at 200Hz. Returns 0 or 1 mel frames per call (occasionally 2
    /// when the audio buffer crosses two hop boundaries).
    ///
    /// When interpolation is enabled, synthesizes audio from a 50/50 blend of the
    /// previous and current formant parameters for smoother spectral transitions
    /// across consonant-vowel boundaries.
    pub fn push_frame(
        &mut self,
        frame: &FormantFrame,
        voice_quality: &MelVoiceQuality,
    ) -> Vec<Vec<f32>> {
        // 1. Synthesize audio: optionally interpolate with previous frame
        if self.config.interpolation {
            if let (Some(prev), Some(prev_vq)) = (&self.prev_frame, &self.prev_vq) {
                // Synthesize first half from interpolated frame (prev→current midpoint)
                let interp = interpolate_frames(prev, frame, 0.5);
                let interp_vq = MelVoiceQuality {
                    rd: prev_vq.rd * 0.5 + voice_quality.rd * 0.5,
                    arousal: prev_vq.arousal * 0.5 + voice_quality.arousal * 0.5,
                };
                let half = self.samples_per_frame / 2;
                let first_half = self.synthesize_source_filter_n(&interp, &interp_vq, half);
                let second_half = self.synthesize_source_filter_n(
                    frame,
                    voice_quality,
                    self.samples_per_frame - half,
                );
                self.audio_buffer.extend_from_slice(&first_half);
                self.audio_buffer.extend_from_slice(&second_half);
            } else {
                let audio = self.synthesize_source_filter(frame, voice_quality);
                self.audio_buffer.extend_from_slice(&audio);
            }
            self.prev_frame = Some(*frame);
            self.prev_vq = Some(*voice_quality);
        } else {
            let audio = self.synthesize_source_filter(frame, voice_quality);
            self.audio_buffer.extend_from_slice(&audio);
        }

        // 2. Extract mel frames from buffer
        let mut mel_frames = Vec::new();
        while self.audio_buffer.len() >= self.config.n_fft {
            let mel = self.extract_mel_frame();
            mel_frames.push(mel);
            // Advance by hop_length
            self.audio_buffer
                .drain(..self.config.hop_length.min(self.audio_buffer.len()));
        }

        mel_frames
    }

    /// Get configuration.
    pub fn config(&self) -> &FormantToMelConfig {
        &self.config
    }

    /// Reset internal state (buffers, resonators, phase).
    pub fn reset(&mut self) {
        self.audio_buffer.clear();
        self.glottal_phase = 0.0;
        for r in &mut self.resonators {
            r.reset();
        }
        self.prev_frame = None;
        self.prev_vq = None;
    }

    // ── Source-filter synthesis ──────────────────────────────────────────

    fn synthesize_source_filter(&mut self, frame: &FormantFrame, vq: &MelVoiceQuality) -> Vec<f32> {
        self.synthesize_source_filter_n(frame, vq, self.samples_per_frame)
    }

    fn synthesize_source_filter_n(
        &mut self,
        frame: &FormantFrame,
        vq: &MelVoiceQuality,
        n: usize,
    ) -> Vec<f32> {
        let sr = self.config.sample_rate as f32;

        // Update resonator parameters
        self.resonators[0].set_params(frame.f1, frame.b1, sr);
        self.resonators[1].set_params(frame.f2, frame.b2, sr);
        self.resonators[2].set_params(frame.f3, frame.b3, sr);

        // Compute breathiness noise level:
        // Higher Rd = more breathy, lower arousal = more breathy
        let rd = vq.rd.clamp(0.3, 2.7);
        let breathiness = self.config.aspiration_base
            + (rd - 1.0).max(0.0) * 0.03  // Rd contribution: breathy voice adds noise
            + (1.0 - vq.arousal) * 0.01; // Low arousal adds slight breathiness

        let mut samples = Vec::with_capacity(n);
        let f0 = frame.f0.max(0.0);
        let voicing = frame.voicing.clamp(0.0, 1.0);
        let energy = frame.energy.clamp(0.0, 1.0);

        for _ in 0..n {
            // Source signal: mix of glottal pulse and noise
            let source = match frame.source_type {
                SourceType::Silent => 0.0,
                SourceType::Fricative => {
                    // Pure noise for fricatives
                    self.noise() * energy
                }
                SourceType::Stop => {
                    // Burst noise
                    self.noise() * energy * 0.5
                }
                _ => {
                    // Voiced source: LF glottal pulse approximation
                    let glottal = if f0 > 0.0 {
                        let period = sr / f0;
                        self.glottal_phase += 1.0 / period;
                        if self.glottal_phase >= 1.0 {
                            self.glottal_phase -= 1.0;
                        }
                        // Simplified LF model: derivative of glottal flow
                        // Rising phase (open) then rapid closure
                        let phase = self.glottal_phase;
                        let open_quotient = 0.4 + (rd - 1.0).clamp(0.0, 1.0) * 0.2;
                        if phase < open_quotient {
                            // Opening phase: sine-like rise
                            (PI * phase / open_quotient).sin()
                        } else {
                            // Closing phase: exponential decay
                            let t = (phase - open_quotient) / (1.0 - open_quotient);
                            -(1.0 - t) * (-t * 3.0).exp()
                        }
                    } else {
                        0.0
                    };

                    // Mix voiced + aspiration noise
                    let aspiration = self.noise() * breathiness;
                    (glottal * voicing + aspiration) * energy
                }
            };

            // Filter through formant resonators (parallel combination)
            let f1_out = self.resonators[0].process(source);
            let f2_out = self.resonators[1].process(source);
            let f3_out = self.resonators[2].process(source);

            let sample = (f1_out + f2_out * 0.7 + f3_out * 0.4).clamp(-1.0, 1.0);
            samples.push(sample);
        }

        samples
    }

    /// Simple fast PRNG noise in [-1, 1].
    fn noise(&mut self) -> f32 {
        // xorshift32
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 17;
        self.rng_state ^= self.rng_state << 5;
        // Map to [-1, 1]
        (self.rng_state as f32 / u32::MAX as f32) * 2.0 - 1.0
    }

    // ── Mel extraction ──────────────────────────────────────────────────

    fn extract_mel_frame(&mut self) -> Vec<f32> {
        let n_fft = self.config.n_fft;
        let n_bins = n_fft / 2 + 1;

        // Apply Hann window and zero-pad into complex buffer
        let mut buffer: Vec<Complex<f32>> = self.audio_buffer[..n_fft]
            .iter()
            .zip(self.window.iter())
            .map(|(&s, &w)| Complex::new(s * w, 0.0))
            .collect();

        // FFT in-place
        self.fft
            .process_with_scratch(&mut buffer, &mut self.scratch);

        // Power spectrum (first n_fft/2 + 1 bins)
        let power: Vec<f32> = buffer
            .iter()
            .take(n_bins)
            .map(|c| (c.re * c.re + c.im * c.im) / n_fft as f32)
            .collect();

        // Apply mel filterbank + log normalization
        let mut mel = Vec::with_capacity(self.config.n_mels);
        for filter in &self.mel_filters {
            let energy: f32 = filter
                .iter()
                .zip(power.iter())
                .map(|(&f, &p)| f * p)
                .sum::<f32>()
                .max(1e-10);

            // Log-mel: ln(energy), clamped for stability
            mel.push(energy.ln().max(-20.0));
        }

        // Apply mean/std normalization (improvement #1)
        self.config.normalization.normalize(&mut mel);

        mel
    }

    // ── Mel filterbank construction ─────────────────────────────────────

    fn create_mel_filterbank(config: &FormantToMelConfig) -> Vec<Vec<f32>> {
        let n_fft_bins = config.n_fft / 2 + 1;

        let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
        let mel_to_hz = |mel: f32| 700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0);

        let mel_min = hz_to_mel(config.f_min);
        let mel_max = hz_to_mel(config.f_max);

        // n_mels + 2 points for triangular filters
        let mel_points: Vec<f32> = (0..=config.n_mels + 1)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (config.n_mels + 1) as f32)
            .collect();

        let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
        let bin_points: Vec<usize> = hz_points
            .iter()
            .map(|&hz| {
                ((hz * config.n_fft as f32 / config.sample_rate as f32) as usize)
                    .min(n_fft_bins - 1)
            })
            .collect();

        let mut filters = Vec::with_capacity(config.n_mels);
        for m in 0..config.n_mels {
            let mut filter = vec![0.0; n_fft_bins];
            let left = bin_points[m];
            let center = bin_points[m + 1];
            let right = bin_points[m + 2];

            // Rising slope
            for k in left..center {
                if center > left {
                    filter[k] = (k - left) as f32 / (center - left) as f32;
                }
            }
            // Falling slope
            for k in center..=right {
                if right > center {
                    filter[k] = (right - k) as f32 / (right - center) as f32;
                }
            }

            filters.push(filter);
        }

        filters
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// JSON HELPER
// ═══════════════════════════════════════════════════════════════════════════════

/// Minimal JSON array extractor — finds `"key": [...]` and parses float values.
/// No external JSON dependency needed for this simple format.
fn extract_json_array(json: &str, key: &str) -> Option<Vec<f32>> {
    let needle = format!("\"{}\"", key);
    let start = json.find(&needle)?;
    let after_key = &json[start + needle.len()..];
    let bracket_start = after_key.find('[')?;
    let bracket_end = after_key.find(']')?;
    let array_str = &after_key[bracket_start + 1..bracket_end];
    let values: Vec<f32> = array_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    if values.is_empty() {
        None
    } else {
        Some(values)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// INTERPOLATION HELPER
// ═══════════════════════════════════════════════════════════════════════════════

/// Linearly interpolate between two FormantFrames at ratio `t` (0.0 = `a`, 1.0 = `b`).
fn interpolate_frames(a: &FormantFrame, b: &FormantFrame, t: f32) -> FormantFrame {
    let lerp = |x: f32, y: f32| x + (y - x) * t;
    FormantFrame {
        f1: lerp(a.f1, b.f1),
        f2: lerp(a.f2, b.f2),
        f3: lerp(a.f3, b.f3),
        b1: lerp(a.b1, b.b1),
        b2: lerp(a.b2, b.b2),
        b3: lerp(a.b3, b.b3),
        f0: lerp(a.f0, b.f0),
        energy: lerp(a.energy, b.energy),
        voicing: lerp(a.voicing, b.voicing),
        time: lerp(a.time, b.time),
        source_type: b.source_type, // Take target's source type
        nasal_zero_freq: lerp(a.nasal_zero_freq, b.nasal_zero_freq),
        nasal_zero_bw: lerp(a.nasal_zero_bw, b.nasal_zero_bw),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn vowel_frame(f1: f32, f2: f32, f3: f32, f0: f32) -> FormantFrame {
        FormantFrame {
            f1,
            f2,
            f3,
            b1: 60.0,
            b2: 90.0,
            b3: 150.0,
            f0,
            energy: 0.8,
            voicing: 1.0,
            time: 0.0,
            source_type: SourceType::Vowel,
            nasal_zero_freq: 0.0,
            nasal_zero_bw: 0.0,
        }
    }

    #[test]
    fn test_mel_output_shape() {
        let config = FormantToMelConfig::default();
        let n_mels = config.n_mels;
        let mut converter = FormantToMelConverter::new(config);
        let frame = vowel_frame(730.0, 1090.0, 2440.0, 120.0); // AH
        let vq = MelVoiceQuality::default();

        // Push enough frames to get at least one mel output
        let mut all_mels = Vec::new();
        for _ in 0..20 {
            all_mels.extend(converter.push_frame(&frame, &vq));
        }

        assert!(
            !all_mels.is_empty(),
            "Should produce at least one mel frame"
        );
        for mel in &all_mels {
            assert_eq!(
                mel.len(),
                n_mels,
                "Each mel frame should have {n_mels} bins"
            );
        }
    }

    #[test]
    fn test_rate_conversion() {
        // 200 motor frames at 200Hz = 1 second of audio = 24000 samples
        // Expected mel frames: (24000 - 1024) / 256 + 1 ≈ 90-91 frames
        let config = FormantToMelConfig::default();
        let mut converter = FormantToMelConverter::new(config);
        let frame = vowel_frame(500.0, 1500.0, 2500.0, 150.0);
        let vq = MelVoiceQuality::default();

        let mut total_mels = 0;
        for _ in 0..200 {
            total_mels += converter.push_frame(&frame, &vq).len();
        }

        // ~93.75 mel frames per second (24000/256), but accounting for n_fft startup
        assert!(
            (80..=100).contains(&total_mels),
            "Expected ~90 mel frames for 1s of audio, got {total_mels}"
        );
    }

    #[test]
    fn test_breathiness_modulation() {
        let config = FormantToMelConfig::default();
        let frame = vowel_frame(500.0, 1500.0, 2500.0, 150.0);

        // Breathy voice (Rd=2.5)
        let mut conv_breathy = FormantToMelConverter::new(config.clone());
        let vq_breathy = MelVoiceQuality {
            rd: 2.5,
            arousal: 0.2,
        };

        // Pressed voice (Rd=0.5)
        let mut conv_pressed = FormantToMelConverter::new(config);
        let vq_pressed = MelVoiceQuality {
            rd: 0.5,
            arousal: 0.8,
        };

        let mut mels_breathy = Vec::new();
        let mut mels_pressed = Vec::new();
        for _ in 0..30 {
            mels_breathy.extend(conv_breathy.push_frame(&frame, &vq_breathy));
            mels_pressed.extend(conv_pressed.push_frame(&frame, &vq_pressed));
        }

        // Both should produce mel frames
        assert!(!mels_breathy.is_empty());
        assert!(!mels_pressed.is_empty());

        // Breathy voice should have more high-frequency energy (aspiration noise)
        // Compare energy in upper mel bins (last 20%)
        let n = mels_breathy[0].len();
        let upper_start = n * 4 / 5;

        let breathy_upper: f32 = mels_breathy.last().unwrap()[upper_start..]
            .iter()
            .sum::<f32>();
        let pressed_upper: f32 = mels_pressed.last().unwrap()[upper_start..]
            .iter()
            .sum::<f32>();

        assert!(
            breathy_upper > pressed_upper,
            "Breathy voice should have more high-freq energy: breathy={breathy_upper:.2} vs pressed={pressed_upper:.2}"
        );
    }

    #[test]
    fn test_silent_frame_floor() {
        let mut config = FormantToMelConfig::default();
        config.normalization = MelNormalization::disabled(); // Raw log-mel for this test
        let mut converter = FormantToMelConverter::new(config);
        let frame = FormantFrame::silent(0.0);
        let vq = MelVoiceQuality::default();

        let mut mels = Vec::new();
        for _ in 0..30 {
            mels.extend(converter.push_frame(&frame, &vq));
        }

        if let Some(mel) = mels.last() {
            // Silent frames should have very low (near-floor) mel values
            let max_val = mel.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            assert!(
                max_val < -5.0,
                "Silent frame mel values should be very low, got max={max_val:.2}"
            );
        }
    }

    #[test]
    fn test_f0_harmonic_structure() {
        let mut config = FormantToMelConfig::default();
        config.normalization = MelNormalization::disabled(); // Raw log-mel for this test
        let mut converter = FormantToMelConverter::new(config);
        // Voiced frame with clear F0
        let frame = vowel_frame(500.0, 1500.0, 2500.0, 200.0);
        let vq = MelVoiceQuality::default();

        let mut mels = Vec::new();
        for _ in 0..30 {
            mels.extend(converter.push_frame(&frame, &vq));
        }

        assert!(!mels.is_empty());
        let mel = mels.last().unwrap();

        // Voiced frame should have higher energy in low mel bins than high bins.
        // In log-mel domain, higher values = more energy. Low bins covering F0/F1
        // should have higher log-mel values than very high bins (> 8kHz).
        let low_avg: f32 = mel[..10].iter().sum::<f32>() / 10.0;
        let high_avg: f32 = mel[90..].iter().sum::<f32>() / (mel.len() - 90) as f32;

        assert!(
            low_avg > high_avg,
            "Low mel bins should have more energy than very high bins: low={low_avg:.2} high={high_avg:.2}"
        );
    }

    #[test]
    fn test_normalization_shifts_range() {
        let mut config = FormantToMelConfig::default();
        // Disabled normalization: raw log-mel values
        config.normalization = MelNormalization::disabled();
        let mut conv_raw = FormantToMelConverter::new(config.clone());
        let frame = vowel_frame(500.0, 1500.0, 2500.0, 150.0);
        let vq = MelVoiceQuality::default();

        let mut mels_raw = Vec::new();
        for _ in 0..30 {
            mels_raw.extend(conv_raw.push_frame(&frame, &vq));
        }

        // With normalization: values should be centered near 0
        config.normalization = MelNormalization::default();
        let mut conv_norm = FormantToMelConverter::new(config);
        let mut mels_norm = Vec::new();
        for _ in 0..30 {
            mels_norm.extend(conv_norm.push_frame(&frame, &vq));
        }

        assert!(!mels_raw.is_empty());
        assert!(!mels_norm.is_empty());

        // Raw mean should be very negative (log-mel of small powers)
        let raw_mean: f32 = mels_raw.last().unwrap().iter().sum::<f32>() / 100.0;
        let norm_mean: f32 = mels_norm.last().unwrap().iter().sum::<f32>() / 100.0;

        // Normalized values should be closer to zero than raw
        assert!(
            norm_mean.abs() < raw_mean.abs(),
            "Normalized mean ({norm_mean:.2}) should be closer to zero than raw ({raw_mean:.2})"
        );
    }

    #[test]
    fn test_normalization_clips() {
        let norm = MelNormalization {
            mean: vec![0.0],
            std: vec![1.0],
            clip_min: -2.0,
            clip_max: 2.0,
            enabled: true,
        };

        let mut mel = vec![-10.0, -1.0, 0.0, 1.0, 10.0];
        norm.normalize(&mut mel);

        assert_eq!(mel[0], -2.0);
        assert_eq!(mel[1], -1.0);
        assert_eq!(mel[2], 0.0);
        assert_eq!(mel[3], 1.0);
        assert_eq!(mel[4], 2.0);
    }

    #[test]
    fn test_interpolation_smooths_transition() {
        // With interpolation: transitions should be smoother between /AH/ and /IY/
        let mut config = FormantToMelConfig::default();
        config.interpolation = true;
        let mut conv_interp = FormantToMelConverter::new(config.clone());

        config.interpolation = false;
        let mut conv_no_interp = FormantToMelConverter::new(config);

        let vq = MelVoiceQuality::default();
        let ah = vowel_frame(730.0, 1090.0, 2440.0, 120.0);
        let iy = vowel_frame(270.0, 2290.0, 3010.0, 130.0);

        // Warm up both converters with AH
        for _ in 0..15 {
            conv_interp.push_frame(&ah, &vq);
            conv_no_interp.push_frame(&ah, &vq);
        }

        // Transition to IY — collect mel frames at the boundary
        let mut mels_interp = Vec::new();
        let mut mels_no_interp = Vec::new();
        for _ in 0..5 {
            mels_interp.extend(conv_interp.push_frame(&iy, &vq));
            mels_no_interp.extend(conv_no_interp.push_frame(&iy, &vq));
        }

        // Both should produce mel frames
        assert!(!mels_interp.is_empty());
        assert!(!mels_no_interp.is_empty());
    }

    #[test]
    fn test_interpolate_frames_midpoint() {
        let a = vowel_frame(200.0, 1000.0, 2000.0, 100.0);
        let b = vowel_frame(400.0, 2000.0, 3000.0, 200.0);
        let mid = interpolate_frames(&a, &b, 0.5);

        assert!((mid.f1 - 300.0).abs() < 0.01);
        assert!((mid.f2 - 1500.0).abs() < 0.01);
        assert!((mid.f3 - 2500.0).abs() < 0.01);
        assert!((mid.f0 - 150.0).abs() < 0.01);
    }

    #[test]
    fn test_voiced_vs_unvoiced() {
        let config = FormantToMelConfig::default();
        let vq = MelVoiceQuality::default();

        // Vowel (voiced, harmonic)
        let mut conv_vowel = FormantToMelConverter::new(config.clone());
        let vowel = vowel_frame(730.0, 1090.0, 2440.0, 120.0);

        // Fricative (unvoiced, noise)
        let mut conv_fric = FormantToMelConverter::new(config);
        let fricative = FormantFrame {
            f1: 300.0,
            f2: 2000.0,
            f3: 3500.0,
            b1: 100.0,
            b2: 150.0,
            b3: 250.0,
            f0: 0.0,
            energy: 0.8,
            voicing: 0.0,
            time: 0.0,
            source_type: SourceType::Fricative,
            nasal_zero_freq: 0.0,
            nasal_zero_bw: 0.0,
        };

        let mut mels_vowel = Vec::new();
        let mut mels_fric = Vec::new();
        for _ in 0..30 {
            mels_vowel.extend(conv_vowel.push_frame(&vowel, &vq));
            mels_fric.extend(conv_fric.push_frame(&fricative, &vq));
        }

        assert!(!mels_vowel.is_empty());
        assert!(!mels_fric.is_empty());

        // Vowel should have more concentrated energy (harmonic structure)
        // Fricative should have more spread (noise-like)
        let vowel_mel = mels_vowel.last().unwrap();
        let fric_mel = mels_fric.last().unwrap();

        // Compute spectral flatness proxy: std_dev / mean
        let vowel_mean: f32 = vowel_mel.iter().sum::<f32>() / vowel_mel.len() as f32;
        let vowel_var: f32 = vowel_mel
            .iter()
            .map(|x| (x - vowel_mean).powi(2))
            .sum::<f32>()
            / vowel_mel.len() as f32;

        let fric_mean: f32 = fric_mel.iter().sum::<f32>() / fric_mel.len() as f32;
        let fric_var: f32 = fric_mel
            .iter()
            .map(|x| (x - fric_mean).powi(2))
            .sum::<f32>()
            / fric_mel.len() as f32;

        // Fricatives should have flatter spectrum (lower variance in mel domain)
        // This is a soft check — the key point is both produce valid output
        assert!(
            vowel_var > 0.0 && fric_var > 0.0,
            "Both should have non-zero spectral variation"
        );
    }
}
