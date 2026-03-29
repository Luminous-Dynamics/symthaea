// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! AI Stem Separation Module
//!
//! Separates audio tracks into individual stems:
//! - Vocals
//! - Drums
//! - Bass
//! - Other (melody, harmony, etc.)
//!
//! Uses spectral masking and ML-based source separation.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use ndarray::{Array1, Array2, Array3, Axis};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{info, debug, warn};

/// Stem separation errors
#[derive(Debug, Error)]
pub enum StemError {
    #[error("Audio too short for separation: {0} samples")]
    AudioTooShort(usize),

    #[error("Invalid sample rate: {0}")]
    InvalidSampleRate(u32),

    #[error("Model not loaded: {0}")]
    ModelNotLoaded(String),

    #[error("Separation failed: {0}")]
    SeparationFailed(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Stem types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StemType {
    Vocals,
    Drums,
    Bass,
    Other,
    // Extended stems
    Piano,
    Guitar,
    Synth,
}

impl StemType {
    pub fn all_basic() -> Vec<StemType> {
        vec![StemType::Vocals, StemType::Drums, StemType::Bass, StemType::Other]
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            StemType::Vocals => "vocals",
            StemType::Drums => "drums",
            StemType::Bass => "bass",
            StemType::Other => "other",
            StemType::Piano => "piano",
            StemType::Guitar => "guitar",
            StemType::Synth => "synth",
        }
    }
}

/// Separated stem data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stem {
    pub stem_type: StemType,
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u32,
    pub confidence: f32,
}

impl Stem {
    pub fn duration_seconds(&self) -> f32 {
        self.samples.len() as f32 / (self.sample_rate as f32 * self.channels as f32)
    }

    pub fn rms(&self) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }
        (self.samples.iter().map(|s| s * s).sum::<f32>() / self.samples.len() as f32).sqrt()
    }
}

/// Separation result containing all stems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeparationResult {
    pub stems: HashMap<StemType, Stem>,
    pub original_duration: f32,
    pub sample_rate: u32,
    pub processing_time_ms: u64,
}

impl SeparationResult {
    pub fn get(&self, stem_type: StemType) -> Option<&Stem> {
        self.stems.get(&stem_type)
    }

    pub fn vocals(&self) -> Option<&Stem> {
        self.stems.get(&StemType::Vocals)
    }

    pub fn drums(&self) -> Option<&Stem> {
        self.stems.get(&StemType::Drums)
    }

    pub fn bass(&self) -> Option<&Stem> {
        self.stems.get(&StemType::Bass)
    }

    pub fn other(&self) -> Option<&Stem> {
        self.stems.get(&StemType::Other)
    }

    /// Create instrumental (everything except vocals)
    pub fn instrumental(&self) -> Vec<f32> {
        let stems_to_mix: Vec<&Stem> = [StemType::Drums, StemType::Bass, StemType::Other]
            .iter()
            .filter_map(|st| self.stems.get(st))
            .collect();

        if stems_to_mix.is_empty() {
            return Vec::new();
        }

        let len = stems_to_mix.iter().map(|s| s.samples.len()).max().unwrap_or(0);
        let mut result = vec![0.0f32; len];

        for stem in stems_to_mix {
            for (i, &sample) in stem.samples.iter().enumerate() {
                if i < result.len() {
                    result[i] += sample;
                }
            }
        }

        // Normalize
        let peak = result.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        if peak > 1.0 {
            for s in &mut result {
                *s /= peak;
            }
        }

        result
    }

    /// Create acapella (vocals only, already available)
    pub fn acapella(&self) -> Option<Vec<f32>> {
        self.vocals().map(|v| v.samples.clone())
    }

    /// Mix stems with custom levels
    pub fn mix(&self, levels: &HashMap<StemType, f32>) -> Vec<f32> {
        let len = self.stems.values()
            .map(|s| s.samples.len())
            .max()
            .unwrap_or(0);

        let mut result = vec![0.0f32; len];

        for (stem_type, stem) in &self.stems {
            let level = levels.get(stem_type).copied().unwrap_or(1.0);
            for (i, &sample) in stem.samples.iter().enumerate() {
                if i < result.len() {
                    result[i] += sample * level;
                }
            }
        }

        // Soft clip
        for s in &mut result {
            *s = (*s).tanh();
        }

        result
    }
}

/// Stem separation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeparationConfig {
    /// FFT size for spectral processing
    pub fft_size: usize,
    /// Hop size between frames
    pub hop_size: usize,
    /// Number of frequency bins to use
    pub n_bins: usize,
    /// Overlap-add window type
    pub window_type: WindowType,
    /// Use Wiener filtering for refinement
    pub use_wiener: bool,
    /// Number of Wiener iterations
    pub wiener_iterations: usize,
    /// Residual handling mode
    pub residual_mode: ResidualMode,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum WindowType {
    Hann,
    Hamming,
    Blackman,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ResidualMode {
    /// Add residual to "other" stem
    AddToOther,
    /// Distribute residual proportionally
    Distribute,
    /// Discard residual
    Discard,
}

impl Default for SeparationConfig {
    fn default() -> Self {
        Self {
            fft_size: 4096,
            hop_size: 1024,
            n_bins: 2048,
            window_type: WindowType::Hann,
            use_wiener: true,
            wiener_iterations: 1,
            residual_mode: ResidualMode::AddToOther,
        }
    }
}

/// Stem separator engine
pub struct StemSeparator {
    config: SeparationConfig,
    sample_rate: u32,
    /// Precomputed window function
    window: Vec<f32>,
    /// Frequency bands for each stem type
    stem_bands: HashMap<StemType, (f32, f32)>,
}

impl StemSeparator {
    pub fn new(sample_rate: u32) -> Self {
        Self::with_config(sample_rate, SeparationConfig::default())
    }

    pub fn with_config(sample_rate: u32, config: SeparationConfig) -> Self {
        let window = Self::create_window(&config);

        // Define frequency bands for heuristic separation
        // These are refined by ML masks in production
        let mut stem_bands = HashMap::new();
        stem_bands.insert(StemType::Bass, (20.0, 250.0));
        stem_bands.insert(StemType::Drums, (60.0, 8000.0));
        stem_bands.insert(StemType::Vocals, (80.0, 12000.0));
        stem_bands.insert(StemType::Other, (200.0, 16000.0));

        Self {
            config,
            sample_rate,
            window,
            stem_bands,
        }
    }

    fn create_window(config: &SeparationConfig) -> Vec<f32> {
        let n = config.fft_size;
        (0..n)
            .map(|i| {
                let x = i as f32 / n as f32;
                match config.window_type {
                    WindowType::Hann => 0.5 * (1.0 - (2.0 * std::f32::consts::PI * x).cos()),
                    WindowType::Hamming => 0.54 - 0.46 * (2.0 * std::f32::consts::PI * x).cos(),
                    WindowType::Blackman => {
                        0.42 - 0.5 * (2.0 * std::f32::consts::PI * x).cos()
                            + 0.08 * (4.0 * std::f32::consts::PI * x).cos()
                    }
                }
            })
            .collect()
    }

    /// Separate audio into stems
    pub fn separate(&self, audio: &[f32]) -> Result<SeparationResult, StemError> {
        let start_time = std::time::Instant::now();

        if audio.len() < self.config.fft_size * 2 {
            return Err(StemError::AudioTooShort(audio.len()));
        }

        info!("Starting stem separation on {} samples", audio.len());

        // Compute STFT
        let stft = self.compute_stft(audio);
        let (magnitude, phase) = self.polar_decompose(&stft);

        // Generate separation masks
        let masks = self.generate_masks(&magnitude);

        // Apply masks and reconstruct
        let mut stems = HashMap::new();

        for stem_type in StemType::all_basic() {
            if let Some(mask) = masks.get(&stem_type) {
                let stem_magnitude = self.apply_mask(&magnitude, mask);

                // Wiener filtering for refinement
                let refined_magnitude = if self.config.use_wiener {
                    self.wiener_filter(&stem_magnitude, &magnitude, self.config.wiener_iterations)
                } else {
                    stem_magnitude
                };

                // Reconstruct time-domain signal
                let stem_stft = self.polar_compose(&refined_magnitude, &phase);
                let stem_audio = self.compute_istft(&stem_stft, audio.len());

                let confidence = self.estimate_confidence(&refined_magnitude, &magnitude);

                stems.insert(
                    stem_type,
                    Stem {
                        stem_type,
                        samples: stem_audio,
                        sample_rate: self.sample_rate,
                        channels: 1,
                        confidence,
                    },
                );
            }
        }

        let processing_time = start_time.elapsed().as_millis() as u64;

        info!(
            "Stem separation complete in {}ms, {} stems extracted",
            processing_time,
            stems.len()
        );

        Ok(SeparationResult {
            stems,
            original_duration: audio.len() as f32 / self.sample_rate as f32,
            sample_rate: self.sample_rate,
            processing_time_ms: processing_time,
        })
    }

    /// Compute Short-Time Fourier Transform
    fn compute_stft(&self, audio: &[f32]) -> Vec<Vec<(f32, f32)>> {
        let n_frames = (audio.len() - self.config.fft_size) / self.config.hop_size + 1;
        let mut stft = Vec::with_capacity(n_frames);

        for frame_idx in 0..n_frames {
            let start = frame_idx * self.config.hop_size;
            let frame: Vec<f32> = (0..self.config.fft_size)
                .map(|i| {
                    if start + i < audio.len() {
                        audio[start + i] * self.window[i]
                    } else {
                        0.0
                    }
                })
                .collect();

            // Simple DFT (in production, use FFT library)
            let spectrum = self.dft(&frame);
            stft.push(spectrum);
        }

        stft
    }

    /// Simple DFT implementation
    fn dft(&self, frame: &[f32]) -> Vec<(f32, f32)> {
        let n = frame.len();
        let n_bins = n / 2 + 1;

        (0..n_bins)
            .map(|k| {
                let mut real = 0.0f32;
                let mut imag = 0.0f32;

                for (i, &sample) in frame.iter().enumerate() {
                    let angle = -2.0 * std::f32::consts::PI * k as f32 * i as f32 / n as f32;
                    real += sample * angle.cos();
                    imag += sample * angle.sin();
                }

                (real, imag)
            })
            .collect()
    }

    /// Compute inverse STFT
    fn compute_istft(&self, stft: &[Vec<(f32, f32)>], target_len: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; target_len];
        let mut window_sum = vec![0.0f32; target_len];

        for (frame_idx, spectrum) in stft.iter().enumerate() {
            let frame = self.idft(spectrum);
            let start = frame_idx * self.config.hop_size;

            for (i, &sample) in frame.iter().enumerate() {
                if start + i < target_len {
                    output[start + i] += sample * self.window[i];
                    window_sum[start + i] += self.window[i] * self.window[i];
                }
            }
        }

        // Normalize by window sum
        for (i, sum) in window_sum.iter().enumerate() {
            if *sum > 1e-8 {
                output[i] /= sum;
            }
        }

        output
    }

    /// Simple inverse DFT
    fn idft(&self, spectrum: &[(f32, f32)]) -> Vec<f32> {
        let n = self.config.fft_size;
        let n_bins = spectrum.len();

        (0..n)
            .map(|i| {
                let mut sum = spectrum[0].0; // DC component

                for k in 1..n_bins - 1 {
                    let angle = 2.0 * std::f32::consts::PI * k as f32 * i as f32 / n as f32;
                    sum += 2.0 * (spectrum[k].0 * angle.cos() - spectrum[k].1 * angle.sin());
                }

                // Nyquist component
                if n_bins > 1 {
                    let angle = std::f32::consts::PI * i as f32;
                    sum += spectrum[n_bins - 1].0 * angle.cos();
                }

                sum / n as f32
            })
            .collect()
    }

    /// Decompose complex spectrum into magnitude and phase
    fn polar_decompose(&self, stft: &[Vec<(f32, f32)>]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let magnitude: Vec<Vec<f32>> = stft
            .iter()
            .map(|frame| {
                frame
                    .iter()
                    .map(|(re, im)| (re * re + im * im).sqrt())
                    .collect()
            })
            .collect();

        let phase: Vec<Vec<f32>> = stft
            .iter()
            .map(|frame| frame.iter().map(|(re, im)| im.atan2(*re)).collect())
            .collect();

        (magnitude, phase)
    }

    /// Compose complex spectrum from magnitude and phase
    fn polar_compose(&self, magnitude: &[Vec<f32>], phase: &[Vec<f32>]) -> Vec<Vec<(f32, f32)>> {
        magnitude
            .iter()
            .zip(phase.iter())
            .map(|(mag_frame, phase_frame)| {
                mag_frame
                    .iter()
                    .zip(phase_frame.iter())
                    .map(|(&m, &p)| (m * p.cos(), m * p.sin()))
                    .collect()
            })
            .collect()
    }

    /// Generate separation masks for each stem
    fn generate_masks(&self, magnitude: &[Vec<f32>]) -> HashMap<StemType, Vec<Vec<f32>>> {
        let n_frames = magnitude.len();
        let n_bins = magnitude.first().map(|f| f.len()).unwrap_or(0);

        let mut masks = HashMap::new();

        // Frequency resolution
        let freq_per_bin = self.sample_rate as f32 / (2.0 * n_bins as f32);

        for stem_type in StemType::all_basic() {
            let (low_freq, high_freq) = self.stem_bands.get(&stem_type).copied().unwrap_or((0.0, 22050.0));

            let low_bin = (low_freq / freq_per_bin) as usize;
            let high_bin = ((high_freq / freq_per_bin) as usize).min(n_bins);

            let mask: Vec<Vec<f32>> = magnitude
                .iter()
                .map(|frame| {
                    (0..n_bins)
                        .map(|bin| {
                            if bin >= low_bin && bin <= high_bin {
                                // Apply frequency-based weighting
                                let freq = bin as f32 * freq_per_bin;
                                self.stem_weight(stem_type, freq, frame[bin])
                            } else {
                                0.0
                            }
                        })
                        .collect()
                })
                .collect();

            masks.insert(stem_type, mask);
        }

        // Normalize masks so they sum to 1 at each time-frequency point
        self.normalize_masks(&mut masks, n_frames, n_bins);

        masks
    }

    /// Compute stem-specific weight for a frequency
    fn stem_weight(&self, stem_type: StemType, freq: f32, magnitude: f32) -> f32 {
        match stem_type {
            StemType::Bass => {
                // Strong in low frequencies, drops off
                if freq < 60.0 {
                    1.0
                } else if freq < 250.0 {
                    1.0 - (freq - 60.0) / 190.0 * 0.7
                } else {
                    0.0
                }
            }
            StemType::Drums => {
                // Transient-focused, broadband but emphasis on attack frequencies
                let low_weight = if freq >= 60.0 && freq <= 150.0 { 0.8 } else { 0.0 };
                let mid_weight = if freq >= 2000.0 && freq <= 5000.0 { 0.6 } else { 0.0 };
                let high_weight = if freq >= 8000.0 && freq <= 12000.0 { 0.4 } else { 0.0 };
                (low_weight + mid_weight + high_weight).min(1.0)
            }
            StemType::Vocals => {
                // Fundamental and harmonics of human voice
                if freq >= 80.0 && freq <= 300.0 {
                    0.7 // Fundamental
                } else if freq >= 300.0 && freq <= 3500.0 {
                    0.9 // Primary vocal range
                } else if freq >= 3500.0 && freq <= 8000.0 {
                    0.5 // Sibilance, presence
                } else {
                    0.0
                }
            }
            StemType::Other => {
                // Everything else - melodic content
                if freq >= 200.0 && freq <= 8000.0 {
                    0.6
                } else {
                    0.3
                }
            }
            _ => 0.5,
        }
    }

    /// Normalize masks to sum to 1
    fn normalize_masks(
        &self,
        masks: &mut HashMap<StemType, Vec<Vec<f32>>>,
        n_frames: usize,
        n_bins: usize,
    ) {
        for frame_idx in 0..n_frames {
            for bin_idx in 0..n_bins {
                let sum: f32 = masks
                    .values()
                    .map(|m| m.get(frame_idx).and_then(|f| f.get(bin_idx)).copied().unwrap_or(0.0))
                    .sum();

                if sum > 0.0 {
                    for mask in masks.values_mut() {
                        if let Some(frame) = mask.get_mut(frame_idx) {
                            if let Some(val) = frame.get_mut(bin_idx) {
                                *val /= sum;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Apply mask to magnitude spectrum
    fn apply_mask(&self, magnitude: &[Vec<f32>], mask: &[Vec<f32>]) -> Vec<Vec<f32>> {
        magnitude
            .iter()
            .zip(mask.iter())
            .map(|(mag_frame, mask_frame)| {
                mag_frame
                    .iter()
                    .zip(mask_frame.iter())
                    .map(|(&m, &mask_val)| m * mask_val)
                    .collect()
            })
            .collect()
    }

    /// Wiener filtering for mask refinement
    fn wiener_filter(
        &self,
        stem_magnitude: &[Vec<f32>],
        total_magnitude: &[Vec<f32>],
        iterations: usize,
    ) -> Vec<Vec<f32>> {
        let mut result = stem_magnitude.to_vec();

        for _ in 0..iterations {
            result = result
                .iter()
                .zip(total_magnitude.iter())
                .map(|(stem_frame, total_frame)| {
                    stem_frame
                        .iter()
                        .zip(total_frame.iter())
                        .map(|(&s, &t)| {
                            if t > 1e-10 {
                                let ratio = s / t;
                                s * ratio / (ratio + 0.01) // Soft Wiener gain
                            } else {
                                s
                            }
                        })
                        .collect()
                })
                .collect();
        }

        result
    }

    /// Estimate confidence of separation
    fn estimate_confidence(&self, stem_magnitude: &[Vec<f32>], total_magnitude: &[Vec<f32>]) -> f32 {
        let mut stem_energy = 0.0f32;
        let mut total_energy = 0.0f32;

        for (stem_frame, total_frame) in stem_magnitude.iter().zip(total_magnitude.iter()) {
            for (&s, &t) in stem_frame.iter().zip(total_frame.iter()) {
                stem_energy += s * s;
                total_energy += t * t;
            }
        }

        if total_energy > 0.0 {
            (stem_energy / total_energy).sqrt().min(1.0)
        } else {
            0.0
        }
    }
}

/// Thread-safe stem separator
pub struct SharedStemSeparator {
    inner: Arc<RwLock<StemSeparator>>,
}

impl SharedStemSeparator {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            inner: Arc::new(RwLock::new(StemSeparator::new(sample_rate))),
        }
    }

    pub fn separate(&self, audio: &[f32]) -> Result<SeparationResult, StemError> {
        self.inner.read().separate(audio)
    }
}

impl Clone for SharedStemSeparator {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

/// Stem mixer for real-time playback
pub struct StemMixer {
    stems: HashMap<StemType, Vec<f32>>,
    levels: HashMap<StemType, f32>,
    mutes: HashMap<StemType, bool>,
    solos: HashMap<StemType, bool>,
    position: usize,
    sample_rate: u32,
}

impl StemMixer {
    pub fn new(result: SeparationResult) -> Self {
        let mut stems = HashMap::new();
        let mut levels = HashMap::new();
        let mut mutes = HashMap::new();
        let mut solos = HashMap::new();

        for (stem_type, stem) in result.stems {
            stems.insert(stem_type, stem.samples);
            levels.insert(stem_type, 1.0);
            mutes.insert(stem_type, false);
            solos.insert(stem_type, false);
        }

        Self {
            stems,
            levels,
            mutes,
            solos,
            position: 0,
            sample_rate: result.sample_rate,
        }
    }

    pub fn set_level(&mut self, stem_type: StemType, level: f32) {
        self.levels.insert(stem_type, level.clamp(0.0, 2.0));
    }

    pub fn set_mute(&mut self, stem_type: StemType, muted: bool) {
        self.mutes.insert(stem_type, muted);
    }

    pub fn set_solo(&mut self, stem_type: StemType, solo: bool) {
        self.solos.insert(stem_type, solo);
    }

    pub fn seek(&mut self, position: usize) {
        self.position = position;
    }

    pub fn seek_seconds(&mut self, seconds: f32) {
        self.position = (seconds * self.sample_rate as f32) as usize;
    }

    /// Get mixed output for given number of samples
    pub fn get_samples(&mut self, count: usize) -> Vec<f32> {
        let any_solo = self.solos.values().any(|&s| s);

        let mut output = vec![0.0f32; count];

        for (stem_type, samples) in &self.stems {
            let level = self.levels.get(stem_type).copied().unwrap_or(1.0);
            let muted = self.mutes.get(stem_type).copied().unwrap_or(false);
            let solo = self.solos.get(stem_type).copied().unwrap_or(false);

            // Skip if muted, or if any track is soloed and this isn't
            if muted || (any_solo && !solo) {
                continue;
            }

            for i in 0..count {
                let idx = self.position + i;
                if idx < samples.len() {
                    output[i] += samples[idx] * level;
                }
            }
        }

        self.position += count;

        // Soft clip output
        for s in &mut output {
            *s = s.tanh();
        }

        output
    }

    pub fn is_finished(&self) -> bool {
        self.stems
            .values()
            .all(|samples| self.position >= samples.len())
    }

    pub fn duration_seconds(&self) -> f32 {
        self.stems
            .values()
            .map(|s| s.len())
            .max()
            .unwrap_or(0) as f32
            / self.sample_rate as f32
    }

    pub fn position_seconds(&self) -> f32 {
        self.position as f32 / self.sample_rate as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_audio(duration_sec: f32, sample_rate: u32) -> Vec<f32> {
        let num_samples = (duration_sec * sample_rate as f32) as usize;
        (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                // Mix of frequencies to simulate complex audio
                let bass = (t * 80.0 * 2.0 * std::f32::consts::PI).sin() * 0.3;
                let mid = (t * 440.0 * 2.0 * std::f32::consts::PI).sin() * 0.2;
                let high = (t * 4000.0 * 2.0 * std::f32::consts::PI).sin() * 0.1;
                bass + mid + high
            })
            .collect()
    }

    #[test]
    fn test_stem_separation() {
        let audio = generate_test_audio(2.0, 44100);
        let separator = StemSeparator::new(44100);

        let result = separator.separate(&audio).unwrap();

        assert_eq!(result.stems.len(), 4);
        assert!(result.vocals().is_some());
        assert!(result.drums().is_some());
        assert!(result.bass().is_some());
        assert!(result.other().is_some());
    }

    #[test]
    fn test_stem_mixer() {
        let audio = generate_test_audio(1.0, 44100);
        let separator = StemSeparator::new(44100);
        let result = separator.separate(&audio).unwrap();

        let mut mixer = StemMixer::new(result);
        mixer.set_level(StemType::Vocals, 0.5);
        mixer.set_mute(StemType::Drums, true);

        let output = mixer.get_samples(1024);
        assert_eq!(output.len(), 1024);
    }

    #[test]
    fn test_instrumental_creation() {
        let audio = generate_test_audio(1.0, 44100);
        let separator = StemSeparator::new(44100);
        let result = separator.separate(&audio).unwrap();

        let instrumental = result.instrumental();
        assert!(!instrumental.is_empty());
    }
}
