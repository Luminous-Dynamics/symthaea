// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Audio Analysis Module
//!
//! Provides comprehensive audio analysis including:
//! - Loudness measurement (EBU R128, LUFS)
//! - BPM/tempo detection
//! - Key detection
//! - Spectral analysis
//! - Dynamic range analysis

use crate::{AudioBuffer, AudioError, AudioResult};
use std::f32::consts::PI;

// ============================================================================
// LOUDNESS ANALYSIS (EBU R128)
// ============================================================================

/// EBU R128 loudness measurement result
#[derive(Debug, Clone)]
pub struct LoudnessResult {
    /// Integrated loudness in LUFS
    pub integrated_lufs: f32,
    /// Loudness range in LU
    pub loudness_range: f32,
    /// Short-term loudness max in LUFS
    pub short_term_max: f32,
    /// Momentary loudness max in LUFS
    pub momentary_max: f32,
    /// True peak in dBTP
    pub true_peak_db: f32,
}

/// EBU R128 loudness analyzer
pub struct LoudnessAnalyzer {
    sample_rate: u32,
    channels: usize,
    // K-weighting filter state
    pre_filter_state: Vec<[f32; 2]>,
    high_shelf_state: Vec<[f32; 2]>,
    // Gating
    momentary_blocks: Vec<f32>,
    short_term_blocks: Vec<f32>,
}

impl LoudnessAnalyzer {
    /// Create new analyzer for given sample rate and channels
    pub fn new(sample_rate: u32, channels: usize) -> Self {
        Self {
            sample_rate,
            channels,
            pre_filter_state: vec![[0.0; 2]; channels],
            high_shelf_state: vec![[0.0; 2]; channels],
            momentary_blocks: Vec::new(),
            short_term_blocks: Vec::new(),
        }
    }

    /// Analyze loudness of audio buffer
    pub fn analyze(&mut self, buffer: &AudioBuffer) -> AudioResult<LoudnessResult> {
        if buffer.sample_rate != self.sample_rate || buffer.channels != self.channels {
            return Err(AudioError::ProcessingError(
                "Buffer format doesn't match analyzer".to_string(),
            ));
        }

        // Apply K-weighting filter
        let weighted = self.apply_k_weighting(buffer);

        // Calculate mean square for each channel
        let block_size = (self.sample_rate as f32 * 0.4) as usize; // 400ms blocks
        let hop_size = block_size / 4; // 75% overlap

        let frame_count = weighted.frame_count();
        let mut block_loudness: Vec<f32> = Vec::new();

        let mut pos = 0;
        while pos + block_size <= frame_count {
            let mut channel_sum = 0.0f32;

            for ch in 0..self.channels {
                let mut sum_squares = 0.0f32;
                for i in 0..block_size {
                    let sample = weighted.samples[(pos + i) * self.channels + ch];
                    sum_squares += sample * sample;
                }
                let mean_square = sum_squares / block_size as f32;

                // Apply channel weighting (1.0 for L/R, 1.41 for surround)
                let weight = if ch < 2 { 1.0 } else { 1.41 };
                channel_sum += weight * mean_square;
            }

            // Convert to LUFS
            let loudness = -0.691 + 10.0 * (channel_sum + 1e-10).log10();
            block_loudness.push(loudness);

            pos += hop_size;
        }

        // Absolute gating at -70 LUFS
        let gated_blocks: Vec<f32> = block_loudness
            .iter()
            .filter(|&&l| l > -70.0)
            .copied()
            .collect();

        if gated_blocks.is_empty() {
            return Ok(LoudnessResult {
                integrated_lufs: -70.0,
                loudness_range: 0.0,
                short_term_max: -70.0,
                momentary_max: -70.0,
                true_peak_db: self.calculate_true_peak(buffer),
            });
        }

        // Calculate mean of gated blocks for relative threshold
        let mean_gated: f32 = gated_blocks.iter().sum::<f32>() / gated_blocks.len() as f32;
        let relative_threshold = mean_gated - 10.0; // -10 LU below mean

        // Apply relative gating
        let final_blocks: Vec<f32> = gated_blocks
            .iter()
            .filter(|&&l| l > relative_threshold)
            .copied()
            .collect();

        // Integrated loudness
        let integrated = if final_blocks.is_empty() {
            -70.0
        } else {
            // Ungated power sum
            let power_sum: f32 = final_blocks.iter().map(|&l| 10.0f32.powf(l / 10.0)).sum();
            -0.691 + 10.0 * (power_sum / final_blocks.len() as f32).log10()
        };

        // Loudness range (LRA)
        let loudness_range = self.calculate_loudness_range(&final_blocks);

        // Short-term and momentary max
        let short_term_max = self.calculate_short_term_max(buffer);
        let momentary_max = block_loudness
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .unwrap_or(-70.0);

        Ok(LoudnessResult {
            integrated_lufs: integrated,
            loudness_range,
            short_term_max,
            momentary_max,
            true_peak_db: self.calculate_true_peak(buffer),
        })
    }

    fn apply_k_weighting(&mut self, buffer: &AudioBuffer) -> AudioBuffer {
        let mut output = buffer.clone();

        // Pre-filter (high-pass shelving)
        // Coefficients for 48kHz, would need adjustment for other rates
        let pre_b = [1.53512485958697, -2.69169618940638, 1.19839281085285];
        let pre_a = [1.0, -1.69065929318241, 0.73248077421585];

        // High shelf filter
        let shelf_b = [1.0, -2.0, 1.0];
        let shelf_a = [1.0, -1.99004745483398, 0.99007225036621];

        for frame in 0..buffer.frame_count() {
            for ch in 0..self.channels {
                let idx = frame * self.channels + ch;
                let mut sample = output.samples[idx];

                // Apply pre-filter
                let filtered = pre_b[0] * sample
                    + pre_b[1] * self.pre_filter_state[ch][0]
                    + pre_b[2] * self.pre_filter_state[ch][1]
                    - pre_a[1] * self.high_shelf_state[ch][0]
                    - pre_a[2] * self.high_shelf_state[ch][1];

                self.pre_filter_state[ch][1] = self.pre_filter_state[ch][0];
                self.pre_filter_state[ch][0] = sample;

                sample = filtered;

                // Apply high shelf
                let shelved = shelf_b[0] * sample
                    + shelf_b[1] * self.high_shelf_state[ch][0]
                    + shelf_b[2] * self.high_shelf_state[ch][1]
                    - shelf_a[1] * self.high_shelf_state[ch][0]
                    - shelf_a[2] * self.high_shelf_state[ch][1];

                self.high_shelf_state[ch][1] = self.high_shelf_state[ch][0];
                self.high_shelf_state[ch][0] = sample;

                output.samples[idx] = shelved;
            }
        }

        output
    }

    fn calculate_loudness_range(&self, blocks: &[f32]) -> f32 {
        if blocks.len() < 2 {
            return 0.0;
        }

        let mut sorted = blocks.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // 10th and 95th percentile
        let low_idx = (sorted.len() as f32 * 0.10) as usize;
        let high_idx = (sorted.len() as f32 * 0.95) as usize;

        sorted[high_idx.min(sorted.len() - 1)] - sorted[low_idx]
    }

    fn calculate_short_term_max(&self, buffer: &AudioBuffer) -> f32 {
        let block_size = (self.sample_rate as f32 * 3.0) as usize; // 3 second blocks
        let hop_size = self.sample_rate as usize; // 1 second hop

        let frame_count = buffer.frame_count();
        let mut max_loudness = -70.0f32;

        let mut pos = 0;
        while pos + block_size <= frame_count {
            let mut channel_sum = 0.0f32;

            for ch in 0..self.channels {
                let mut sum_squares = 0.0f32;
                for i in 0..block_size {
                    let sample = buffer.samples[(pos + i) * self.channels + ch];
                    sum_squares += sample * sample;
                }
                channel_sum += sum_squares / block_size as f32;
            }

            let loudness = -0.691 + 10.0 * (channel_sum + 1e-10).log10();
            max_loudness = max_loudness.max(loudness);

            pos += hop_size;
        }

        max_loudness
    }

    fn calculate_true_peak(&self, buffer: &AudioBuffer) -> f32 {
        // 4x oversampling for true peak detection
        let oversample = 4;
        let mut max_peak = 0.0f32;

        for ch in 0..self.channels {
            let channel_samples = buffer.channel(ch);

            // Simple linear interpolation oversampling
            for i in 0..channel_samples.len() - 1 {
                let s0 = channel_samples[i];
                let s1 = channel_samples[i + 1];

                for j in 0..oversample {
                    let t = j as f32 / oversample as f32;
                    let interpolated = s0 + t * (s1 - s0);
                    max_peak = max_peak.max(interpolated.abs());
                }
            }
        }

        20.0 * (max_peak + 1e-10).log10()
    }

    /// Reset analyzer state
    pub fn reset(&mut self) {
        self.pre_filter_state = vec![[0.0; 2]; self.channels];
        self.high_shelf_state = vec![[0.0; 2]; self.channels];
        self.momentary_blocks.clear();
        self.short_term_blocks.clear();
    }
}

// ============================================================================
// BPM DETECTION
// ============================================================================

/// BPM detection result
#[derive(Debug, Clone)]
pub struct BpmResult {
    /// Detected BPM
    pub bpm: f32,
    /// Confidence (0.0 - 1.0)
    pub confidence: f32,
    /// Alternative BPM candidates
    pub alternatives: Vec<(f32, f32)>, // (bpm, confidence)
    /// Beat positions in seconds
    pub beat_positions: Vec<f32>,
}

/// BPM detector using onset detection and autocorrelation
pub struct BpmDetector {
    sample_rate: u32,
    min_bpm: f32,
    max_bpm: f32,
}

impl BpmDetector {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            min_bpm: 60.0,
            max_bpm: 200.0,
        }
    }

    pub fn with_range(mut self, min_bpm: f32, max_bpm: f32) -> Self {
        self.min_bpm = min_bpm;
        self.max_bpm = max_bpm;
        self
    }

    /// Detect BPM from audio buffer
    pub fn detect(&self, buffer: &AudioBuffer) -> AudioResult<BpmResult> {
        // Convert to mono if needed
        let mono = buffer.to_mono();

        // Calculate onset strength envelope
        let onset_env = self.calculate_onset_envelope(&mono);

        // Autocorrelation for tempo estimation
        let (bpm, confidence, alternatives) = self.estimate_tempo(&onset_env);

        // Find beat positions
        let beat_positions = self.find_beats(&onset_env, bpm);

        Ok(BpmResult {
            bpm,
            confidence,
            alternatives,
            beat_positions,
        })
    }

    fn calculate_onset_envelope(&self, buffer: &AudioBuffer) -> Vec<f32> {
        let hop_size = 512;
        let frame_count = buffer.frame_count();
        let num_frames = frame_count / hop_size;

        let mut envelope = Vec::with_capacity(num_frames);
        let mut prev_energy = 0.0f32;

        for i in 0..num_frames {
            let start = i * hop_size;
            let end = (start + hop_size).min(frame_count);

            // Calculate spectral flux (simplified as energy difference)
            let energy: f32 = buffer.samples[start..end]
                .iter()
                .map(|s| s * s)
                .sum::<f32>()
                / hop_size as f32;

            let flux = (energy - prev_energy).max(0.0);
            envelope.push(flux);
            prev_energy = energy;
        }

        // Normalize
        let max_val = envelope.iter().cloned().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(1.0);
        if max_val > 0.0 {
            for v in &mut envelope {
                *v /= max_val;
            }
        }

        envelope
    }

    fn estimate_tempo(&self, onset_env: &[f32]) -> (f32, f32, Vec<(f32, f32)>) {
        let hop_size = 512;
        let envelope_sr = self.sample_rate as f32 / hop_size as f32;

        let min_lag = (60.0 / self.max_bpm * envelope_sr) as usize;
        let max_lag = (60.0 / self.min_bpm * envelope_sr) as usize;

        // Autocorrelation
        let mut correlations = Vec::with_capacity(max_lag - min_lag);

        for lag in min_lag..max_lag {
            let mut correlation = 0.0f32;
            let n = onset_env.len().saturating_sub(lag);

            for i in 0..n {
                correlation += onset_env[i] * onset_env[i + lag];
            }

            correlation /= n as f32;
            let bpm = 60.0 * envelope_sr / lag as f32;
            correlations.push((bpm, correlation));
        }

        // Find peaks
        let mut peaks: Vec<(f32, f32)> = correlations
            .windows(3)
            .enumerate()
            .filter_map(|(i, w)| {
                if w[1].1 > w[0].1 && w[1].1 > w[2].1 {
                    Some((w[1].0, w[1].1))
                } else {
                    None
                }
            })
            .collect();

        peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        if peaks.is_empty() {
            return (120.0, 0.0, Vec::new());
        }

        let best = peaks[0];
        let confidence = best.1;
        let alternatives = peaks.into_iter().skip(1).take(3).collect();

        (best.0, confidence, alternatives)
    }

    fn find_beats(&self, onset_env: &[f32], bpm: f32) -> Vec<f32> {
        let hop_size = 512;
        let envelope_sr = self.sample_rate as f32 / hop_size as f32;
        let beat_period = 60.0 / bpm * envelope_sr;

        // Find peaks in onset envelope
        let threshold = onset_env.iter().sum::<f32>() / onset_env.len() as f32 * 1.5;

        let mut beat_positions = Vec::new();
        let mut last_beat = 0.0f32;

        for (i, &val) in onset_env.iter().enumerate() {
            if val > threshold {
                let time = i as f32 / envelope_sr;

                // Ensure minimum distance between beats
                if time - last_beat > beat_period * 0.5 {
                    beat_positions.push(time);
                    last_beat = time;
                }
            }
        }

        beat_positions
    }
}

// ============================================================================
// KEY DETECTION
// ============================================================================

/// Musical key
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Key {
    CMajor, CMinor,
    CSharpMajor, CSharpMinor,
    DMajor, DMinor,
    DSharpMajor, DSharpMinor,
    EMajor, EMinor,
    FMajor, FMinor,
    FSharpMajor, FSharpMinor,
    GMajor, GMinor,
    GSharpMajor, GSharpMinor,
    AMajor, AMinor,
    ASharpMajor, ASharpMinor,
    BMajor, BMinor,
}

impl Key {
    pub fn name(&self) -> &'static str {
        match self {
            Key::CMajor => "C Major",
            Key::CMinor => "C Minor",
            Key::CSharpMajor => "C# Major",
            Key::CSharpMinor => "C# Minor",
            Key::DMajor => "D Major",
            Key::DMinor => "D Minor",
            Key::DSharpMajor => "D# Major",
            Key::DSharpMinor => "D# Minor",
            Key::EMajor => "E Major",
            Key::EMinor => "E Minor",
            Key::FMajor => "F Major",
            Key::FMinor => "F Minor",
            Key::FSharpMajor => "F# Major",
            Key::FSharpMinor => "F# Minor",
            Key::GMajor => "G Major",
            Key::GMinor => "G Minor",
            Key::GSharpMajor => "G# Major",
            Key::GSharpMinor => "G# Minor",
            Key::AMajor => "A Major",
            Key::AMinor => "A Minor",
            Key::ASharpMajor => "A# Major",
            Key::ASharpMinor => "A# Minor",
            Key::BMajor => "B Major",
            Key::BMinor => "B Minor",
        }
    }

    pub fn camelot(&self) -> &'static str {
        match self {
            Key::CMajor => "8B",
            Key::CMinor => "5A",
            Key::CSharpMajor => "3B",
            Key::CSharpMinor => "12A",
            Key::DMajor => "10B",
            Key::DMinor => "7A",
            Key::DSharpMajor => "5B",
            Key::DSharpMinor => "2A",
            Key::EMajor => "12B",
            Key::EMinor => "9A",
            Key::FMajor => "7B",
            Key::FMinor => "4A",
            Key::FSharpMajor => "2B",
            Key::FSharpMinor => "11A",
            Key::GMajor => "9B",
            Key::GMinor => "6A",
            Key::GSharpMajor => "4B",
            Key::GSharpMinor => "1A",
            Key::AMajor => "11B",
            Key::AMinor => "8A",
            Key::ASharpMajor => "6B",
            Key::ASharpMinor => "3A",
            Key::BMajor => "1B",
            Key::BMinor => "10A",
        }
    }
}

/// Key detection result
#[derive(Debug, Clone)]
pub struct KeyResult {
    /// Detected key
    pub key: Key,
    /// Confidence (0.0 - 1.0)
    pub confidence: f32,
    /// Alternative keys
    pub alternatives: Vec<(Key, f32)>,
}

/// Key detector using chroma features
pub struct KeyDetector {
    sample_rate: u32,
}

impl KeyDetector {
    pub fn new(sample_rate: u32) -> Self {
        Self { sample_rate }
    }

    /// Detect key from audio buffer
    pub fn detect(&self, buffer: &AudioBuffer) -> AudioResult<KeyResult> {
        let mono = buffer.to_mono();

        // Calculate chroma features
        let chroma = self.calculate_chroma(&mono);

        // Match against key profiles
        let (key, confidence, alternatives) = self.match_key_profile(&chroma);

        Ok(KeyResult {
            key,
            confidence,
            alternatives,
        })
    }

    fn calculate_chroma(&self, buffer: &AudioBuffer) -> [f32; 12] {
        // Simplified chroma calculation
        // In production, would use FFT and proper pitch class mapping

        let mut chroma = [0.0f32; 12];
        let fft_size = 4096;
        let hop_size = 2048;

        // For each frame, estimate pitch and accumulate in chroma bins
        let frame_count = buffer.frame_count();
        let num_frames = frame_count / hop_size;

        for i in 0..num_frames {
            let start = i * hop_size;
            let end = (start + fft_size).min(frame_count);

            if end - start < fft_size / 2 {
                break;
            }

            // Estimate fundamental frequency using autocorrelation
            let samples = &buffer.samples[start..end];
            let f0 = self.estimate_pitch(samples);

            if f0 > 0.0 {
                // Convert frequency to pitch class
                let midi = 12.0 * (f0 / 440.0).log2() + 69.0;
                let pitch_class = (midi.round() as i32 % 12) as usize;
                chroma[pitch_class] += 1.0;
            }
        }

        // Normalize
        let sum: f32 = chroma.iter().sum();
        if sum > 0.0 {
            for v in &mut chroma {
                *v /= sum;
            }
        }

        chroma
    }

    fn estimate_pitch(&self, samples: &[f32]) -> f32 {
        // Simple autocorrelation-based pitch detection
        let min_lag = (self.sample_rate as f32 / 1000.0) as usize; // ~1000 Hz max
        let max_lag = (self.sample_rate as f32 / 50.0) as usize; // ~50 Hz min

        let n = samples.len();
        let mut best_lag = 0;
        let mut best_corr = 0.0f32;

        for lag in min_lag..max_lag.min(n / 2) {
            let mut correlation = 0.0f32;
            for i in 0..n - lag {
                correlation += samples[i] * samples[i + lag];
            }
            correlation /= (n - lag) as f32;

            if correlation > best_corr {
                best_corr = correlation;
                best_lag = lag;
            }
        }

        if best_lag > 0 && best_corr > 0.5 {
            self.sample_rate as f32 / best_lag as f32
        } else {
            0.0
        }
    }

    fn match_key_profile(&self, chroma: &[f32; 12]) -> (Key, f32, Vec<(Key, f32)>) {
        // Krumhansl-Kessler key profiles
        let major_profile = [
            6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
        ];
        let minor_profile = [
            6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
        ];

        let keys = [
            (Key::CMajor, 0, true),
            (Key::CMinor, 0, false),
            (Key::CSharpMajor, 1, true),
            (Key::CSharpMinor, 1, false),
            (Key::DMajor, 2, true),
            (Key::DMinor, 2, false),
            (Key::DSharpMajor, 3, true),
            (Key::DSharpMinor, 3, false),
            (Key::EMajor, 4, true),
            (Key::EMinor, 4, false),
            (Key::FMajor, 5, true),
            (Key::FMinor, 5, false),
            (Key::FSharpMajor, 6, true),
            (Key::FSharpMinor, 6, false),
            (Key::GMajor, 7, true),
            (Key::GMinor, 7, false),
            (Key::GSharpMajor, 8, true),
            (Key::GSharpMinor, 8, false),
            (Key::AMajor, 9, true),
            (Key::AMinor, 9, false),
            (Key::ASharpMajor, 10, true),
            (Key::ASharpMinor, 10, false),
            (Key::BMajor, 11, true),
            (Key::BMinor, 11, false),
        ];

        let mut correlations: Vec<(Key, f32)> = keys
            .iter()
            .map(|(key, shift, is_major)| {
                let profile = if *is_major {
                    &major_profile
                } else {
                    &minor_profile
                };

                // Rotate profile by shift
                let rotated: Vec<f32> = (0..12)
                    .map(|i| profile[(i + 12 - shift) % 12])
                    .collect();

                // Calculate correlation
                let corr = self.pearson_correlation(chroma, &rotated);
                (*key, corr)
            })
            .collect();

        correlations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let best = correlations[0];
        let alternatives: Vec<(Key, f32)> = correlations.into_iter().skip(1).take(3).collect();

        (best.0, best.1, alternatives)
    }

    fn pearson_correlation(&self, a: &[f32], b: &[f32]) -> f32 {
        let n = a.len() as f32;
        let mean_a = a.iter().sum::<f32>() / n;
        let mean_b = b.iter().sum::<f32>() / n;

        let mut cov = 0.0f32;
        let mut var_a = 0.0f32;
        let mut var_b = 0.0f32;

        for i in 0..a.len() {
            let da = a[i] - mean_a;
            let db = b[i] - mean_b;
            cov += da * db;
            var_a += da * da;
            var_b += db * db;
        }

        if var_a > 0.0 && var_b > 0.0 {
            cov / (var_a * var_b).sqrt()
        } else {
            0.0
        }
    }
}

// ============================================================================
// SPECTRAL ANALYSIS
// ============================================================================

/// Spectral analysis result
#[derive(Debug, Clone)]
pub struct SpectralAnalysis {
    /// Spectral centroid (brightness indicator)
    pub centroid: f32,
    /// Spectral rolloff
    pub rolloff: f32,
    /// Spectral flatness (0 = tonal, 1 = noisy)
    pub flatness: f32,
    /// Zero crossing rate
    pub zero_crossing_rate: f32,
    /// RMS energy
    pub rms: f32,
}

/// Spectral analyzer
pub struct SpectralAnalyzer {
    sample_rate: u32,
}

impl SpectralAnalyzer {
    pub fn new(sample_rate: u32) -> Self {
        Self { sample_rate }
    }

    /// Analyze spectral features
    pub fn analyze(&self, buffer: &AudioBuffer) -> AudioResult<SpectralAnalysis> {
        let mono = buffer.to_mono();
        let samples = &mono.samples;

        // Zero crossing rate
        let zcr = self.calculate_zcr(samples);

        // RMS
        let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();

        // Simplified spectral features (would use FFT in production)
        let centroid = self.estimate_centroid(samples);
        let rolloff = self.estimate_rolloff(samples);
        let flatness = self.estimate_flatness(samples);

        Ok(SpectralAnalysis {
            centroid,
            rolloff,
            flatness,
            zero_crossing_rate: zcr,
            rms,
        })
    }

    fn calculate_zcr(&self, samples: &[f32]) -> f32 {
        if samples.len() < 2 {
            return 0.0;
        }

        let mut crossings = 0;
        for i in 1..samples.len() {
            if (samples[i] >= 0.0) != (samples[i - 1] >= 0.0) {
                crossings += 1;
            }
        }

        crossings as f32 / samples.len() as f32
    }

    fn estimate_centroid(&self, samples: &[f32]) -> f32 {
        // Simplified centroid estimation
        let zcr = self.calculate_zcr(samples);
        // ZCR correlates with spectral centroid
        zcr * self.sample_rate as f32 / 2.0
    }

    fn estimate_rolloff(&self, samples: &[f32]) -> f32 {
        // Estimate rolloff as 85% energy frequency
        self.estimate_centroid(samples) * 1.5
    }

    fn estimate_flatness(&self, samples: &[f32]) -> f32 {
        // Estimate flatness from amplitude variance
        let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
        let peak = samples.iter().map(|s| s.abs()).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(0.0);

        if peak > 0.0 {
            rms / peak
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loudness_analyzer() {
        // Create a simple test buffer
        let samples: Vec<f32> = (0..44100 * 2)
            .map(|i| (i as f32 * 440.0 * 2.0 * PI / 44100.0).sin() * 0.5)
            .collect();

        let buffer = AudioBuffer::from_samples(samples, 44100, 1, crate::ChannelLayout::Mono);
        let mut analyzer = LoudnessAnalyzer::new(44100, 1);

        let result = analyzer.analyze(&buffer).unwrap();
        assert!(result.integrated_lufs < 0.0); // Should be negative LUFS
    }

    #[test]
    fn test_bpm_detector() {
        // Create a simple click track at 120 BPM
        let sample_rate = 44100;
        let duration_secs = 10;
        let bpm = 120.0;
        let samples_per_beat = (60.0 / bpm * sample_rate as f32) as usize;

        let mut samples = vec![0.0f32; sample_rate * duration_secs];
        for i in (0..samples.len()).step_by(samples_per_beat) {
            for j in 0..100.min(samples.len() - i) {
                samples[i + j] = 0.8;
            }
        }

        let buffer = AudioBuffer::from_samples(samples, sample_rate as u32, 1, crate::ChannelLayout::Mono);
        let detector = BpmDetector::new(sample_rate as u32);

        let result = detector.detect(&buffer).unwrap();
        // Allow some tolerance
        assert!((result.bpm - bpm).abs() < 5.0);
    }
}
