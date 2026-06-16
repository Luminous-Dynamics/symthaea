// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cetacean Acoustic Classifier
//!
//! CfC-based classifier that maps whale audio to Cetacean Articulatory Units.
//!
//! # Feature Extraction
//!
//! Unlike human speech (mel spectrograms), cetacean vocalizations require:
//! - **Transient detection**: For click identification (Teager-Kaiser energy)
//! - **Harmonicity analysis**: For whistle vs burst discrimination
//! - **Frequency tracking**: For modulation direction (upsweep/downsweep/flat)
//!
//! # Architecture
//!
//! ```text
//! Audio → [Feature Extractor] → [CfC Backbone] → [Manner Head] → Click/Whistle/Burst
//!                                              → [Place Head]  → Upsweep/Downsweep/Flat
//! ```

use crate::audio::AudioFrontend;
use crate::cetacean_scorer::{CetaceanManner, CetaceanPlace, CetaceanUnit};
use crate::ltc::{LtcCell, LtcConfig};
use rustfft::{FftPlanner, num_complex::Complex};
use std::path::Path;
use std::sync::Arc;

/// Configuration for cetacean acoustic analysis
#[derive(Debug, Clone)]
pub struct CetaceanConfig {
    /// Sample rate (Hz) - whale recordings typically 44.1kHz or 96kHz
    pub sample_rate: u32,
    /// Frame duration for analysis (seconds)
    pub frame_duration: f32,
    /// Hop between frames (seconds)
    pub hop_duration: f32,
    /// FFT size for spectral analysis
    pub n_fft: usize,
    /// Minimum frequency for analysis (Hz)
    pub f_min: f32,
    /// Maximum frequency for analysis (Hz) - whales use wide range
    pub f_max: f32,
    /// Threshold for transient detection
    pub transient_threshold: f32,
    /// Threshold for harmonicity (whistle detection)
    pub harmonicity_threshold: f32,
    /// Threshold for FM slope (Hz/frame) to count as sweep
    pub fm_slope_threshold: f32,
}

impl Default for CetaceanConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            frame_duration: 0.02, // 20ms frames
            hop_duration: 0.01,   // 10ms hop
            n_fft: 2048,          // High resolution for frequency tracking
            f_min: 100.0,         // Whale vocalizations start low
            f_max: 20000.0,       // But can go quite high
            transient_threshold: 0.5,
            harmonicity_threshold: 0.4,
            fm_slope_threshold: 500.0, // Hz/frame
        }
    }
}

/// Acoustic features extracted from each frame
#[derive(Debug, Clone, Default)]
pub struct CetaceanFrame {
    /// Time of this frame (seconds)
    pub time: f32,
    /// Transient energy (Teager-Kaiser)
    pub transient: f32,
    /// Harmonicity ratio (0=noise, 1=pure tone)
    pub harmonicity: f32,
    /// Spectral centroid (Hz)
    pub centroid: f32,
    /// Spectral flux (change from previous frame)
    pub flux: f32,
    /// Dominant frequency (Hz)
    pub f0: f32,
    /// Frequency slope (Hz/frame) - positive=upsweep, negative=downsweep
    pub fm_slope: f32,
    /// Frame energy (RMS)
    pub energy: f32,
}

/// Cetacean acoustic classifier using CfC dynamics
pub struct CetaceanClassifier {
    config: CetaceanConfig,

    /// CfC backbone for temporal integration
    cfc: LtcCell,

    /// FFT instance
    fft: Arc<dyn rustfft::Fft<f32>>,

    /// Previous spectrum for flux computation
    prev_spectrum: Vec<f32>,

    /// Previous f0 for slope computation
    prev_f0: f32,

    /// Hann window
    window: Vec<f32>,

    /// Frame buffer
    frame_buffer: Vec<f32>,

    /// Current position in audio
    position: usize,
}

impl CetaceanClassifier {
    /// Create a new cetacean classifier
    pub fn new(config: CetaceanConfig) -> Self {
        // CfC backbone: 6 features → 32 hidden
        let cfc_config = LtcConfig {
            hidden_size: 32,
            tau_min: 0.005,
            tau_max: 0.1,
            tau_init: 0.02,
            tau_lr: 0.01,
            adaptive_tau: true,
        };

        let frame_samples = (config.sample_rate as f32 * config.frame_duration) as usize;
        let window = Self::hann_window(frame_samples);

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(config.n_fft);

        Self {
            cfc: LtcCell::new(6, cfc_config), // 6 input features
            fft,
            prev_spectrum: vec![0.0; config.n_fft / 2 + 1],
            prev_f0: 0.0,
            window,
            frame_buffer: Vec::with_capacity(frame_samples),
            position: 0,
            config,
        }
    }

    /// Create Hann window
    fn hann_window(size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32).cos())
            })
            .collect()
    }

    /// Reset classifier state
    pub fn reset(&mut self) {
        self.cfc.reset();
        self.prev_spectrum.fill(0.0);
        self.prev_f0 = 0.0;
        self.frame_buffer.clear();
        self.position = 0;
    }

    /// Process audio and return classified CAU sequence
    pub fn classify_audio(&mut self, samples: &[f32]) -> Vec<(f32, CetaceanUnit)> {
        self.reset();

        let frame_samples = (self.config.sample_rate as f32 * self.config.frame_duration) as usize;
        let hop_samples = (self.config.sample_rate as f32 * self.config.hop_duration) as usize;

        let mut results = Vec::new();
        let mut pos = 0;

        while pos + frame_samples <= samples.len() {
            let frame = &samples[pos..pos + frame_samples];
            let time = pos as f32 / self.config.sample_rate as f32;

            // Extract features
            let features = self.extract_features(frame, time);

            // Classify if there's enough energy
            if features.energy > 0.01 {
                let cau = self.classify_frame(&features);
                results.push((time, cau));
            }

            pos += hop_samples;
        }

        results
    }

    /// Extract acoustic features from a frame
    fn extract_features(&mut self, frame: &[f32], time: f32) -> CetaceanFrame {
        // Apply window
        let windowed: Vec<f32> = frame
            .iter()
            .zip(self.window.iter())
            .map(|(s, w)| s * w)
            .collect();

        // Compute energy (RMS)
        let energy = (windowed.iter().map(|x| x * x).sum::<f32>() / windowed.len() as f32).sqrt();

        // Compute transient (Teager-Kaiser)
        let transient = self.compute_transient(&windowed);

        // Compute spectrum
        let mut fft_buffer: Vec<Complex<f32>> = windowed
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .chain(std::iter::repeat(Complex::new(0.0, 0.0)))
            .take(self.config.n_fft)
            .collect();

        self.fft.process(&mut fft_buffer);

        let spectrum: Vec<f32> = fft_buffer[..self.config.n_fft / 2 + 1]
            .iter()
            .map(|c| c.norm())
            .collect();

        // Compute spectral features
        let centroid = self.compute_centroid(&spectrum);
        let flux = self.compute_flux(&spectrum);
        let (f0, harmonicity) = self.compute_pitch_and_harmonicity(&spectrum);

        // Compute FM slope
        let fm_slope = if self.prev_f0 > 0.0 && f0 > 0.0 {
            (f0 - self.prev_f0) / self.config.hop_duration
        } else {
            0.0
        };

        // Update state
        self.prev_spectrum.copy_from_slice(&spectrum);
        self.prev_f0 = f0;

        CetaceanFrame {
            time,
            transient,
            harmonicity,
            centroid,
            flux,
            f0,
            fm_slope,
            energy,
        }
    }

    /// Compute Teager-Kaiser transient energy
    fn compute_transient(&self, samples: &[f32]) -> f32 {
        if samples.len() < 3 {
            return 0.0;
        }

        let mut teager_sum = 0.0;
        for i in 1..samples.len() - 1 {
            let tk = samples[i] * samples[i] - samples[i - 1] * samples[i + 1];
            teager_sum += tk.abs();
        }

        teager_sum / (samples.len() - 2) as f32
    }

    /// Compute spectral centroid
    fn compute_centroid(&self, spectrum: &[f32]) -> f32 {
        let freq_resolution = self.config.sample_rate as f32 / self.config.n_fft as f32;

        let mut weighted_sum = 0.0;
        let mut total_energy = 0.0;

        for (i, &mag) in spectrum.iter().enumerate() {
            let freq = i as f32 * freq_resolution;
            if freq >= self.config.f_min && freq <= self.config.f_max {
                weighted_sum += freq * mag;
                total_energy += mag;
            }
        }

        if total_energy > 1e-10 {
            weighted_sum / total_energy
        } else {
            0.0
        }
    }

    /// Compute spectral flux
    fn compute_flux(&self, spectrum: &[f32]) -> f32 {
        spectrum
            .iter()
            .zip(self.prev_spectrum.iter())
            .map(|(curr, prev)| {
                let diff = curr - prev;
                if diff > 0.0 { diff * diff } else { 0.0 }
            })
            .sum::<f32>()
            .sqrt()
    }

    /// Compute pitch (f0) and harmonicity
    fn compute_pitch_and_harmonicity(&self, spectrum: &[f32]) -> (f32, f32) {
        let freq_resolution = self.config.sample_rate as f32 / self.config.n_fft as f32;

        // Find dominant frequency
        let min_bin = (self.config.f_min / freq_resolution) as usize;
        let max_bin = ((self.config.f_max / freq_resolution) as usize).min(spectrum.len());

        let mut max_mag = 0.0;
        let mut max_bin_idx = min_bin;

        for i in min_bin..max_bin {
            if spectrum[i] > max_mag {
                max_mag = spectrum[i];
                max_bin_idx = i;
            }
        }

        let f0 = max_bin_idx as f32 * freq_resolution;

        // Compute harmonicity by checking harmonic peaks
        let mut harmonic_energy = 0.0;
        let total_energy: f32 = spectrum[min_bin..max_bin].iter().sum();

        if f0 > 0.0 && total_energy > 1e-10 {
            for harmonic in 1..=5 {
                let harmonic_freq = f0 * harmonic as f32;
                let harmonic_bin = (harmonic_freq / freq_resolution) as usize;

                if harmonic_bin < spectrum.len() {
                    // Sum energy around harmonic (±2 bins)
                    for offset in -2i32..=2 {
                        let bin = (harmonic_bin as i32 + offset) as usize;
                        if bin < spectrum.len() {
                            harmonic_energy += spectrum[bin];
                        }
                    }
                }
            }

            (f0, (harmonic_energy / total_energy).min(1.0))
        } else {
            (0.0, 0.0)
        }
    }

    /// Classify a frame into a Cetacean Articulatory Unit
    fn classify_frame(&mut self, features: &CetaceanFrame) -> CetaceanUnit {
        // Normalize features for CfC input
        let input = [
            features.transient.min(1.0),
            features.harmonicity,
            features.flux.min(1.0),
            (features.fm_slope / 1000.0).clamp(-1.0, 1.0),
            (features.centroid / 10000.0).min(1.0),
            features.energy.min(1.0),
        ];

        // Feed through CfC for temporal smoothing
        let _state = self.cfc.forward(&input, self.config.hop_duration);

        // Classify Manner based on acoustic properties
        let manner =
            if features.transient > self.config.transient_threshold && features.harmonicity < 0.3 {
                // High transient, low harmonicity = Click
                CetaceanManner::Click
            } else if features.harmonicity > self.config.harmonicity_threshold {
                // High harmonicity = Whistle (tonal)
                CetaceanManner::Whistle
            } else {
                // Everything else = Burst (broadband)
                CetaceanManner::Burst
            };

        // Classify Place based on frequency modulation
        let place = if features.fm_slope > self.config.fm_slope_threshold {
            CetaceanPlace::Upsweep
        } else if features.fm_slope < -self.config.fm_slope_threshold {
            CetaceanPlace::Downsweep
        } else {
            CetaceanPlace::Flat
        };

        CetaceanUnit::new(manner, place)
    }

    /// Load and classify a whale audio file
    pub fn classify_file<P: AsRef<Path>>(
        &mut self,
        path: P,
    ) -> std::io::Result<Vec<(f32, CetaceanUnit)>> {
        let (samples, sample_rate) = AudioFrontend::load_audio(path)?;

        // Resample if needed (simple linear for now)
        let samples = if sample_rate != self.config.sample_rate {
            self.resample(&samples, sample_rate, self.config.sample_rate)
        } else {
            samples
        };

        Ok(self.classify_audio(&samples))
    }

    /// Simple linear resampling
    fn resample(&self, samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
        let ratio = from_rate as f32 / to_rate as f32;
        let new_len = (samples.len() as f32 / ratio) as usize;

        (0..new_len)
            .map(|i| {
                let src_pos = i as f32 * ratio;
                let src_idx = src_pos as usize;
                let frac = src_pos - src_idx as f32;

                if src_idx + 1 < samples.len() {
                    samples[src_idx] * (1.0 - frac) + samples[src_idx + 1] * frac
                } else if src_idx < samples.len() {
                    samples[src_idx]
                } else {
                    0.0
                }
            })
            .collect()
    }
}

/// Collapse consecutive identical CAUs into segments
pub fn collapse_to_segments(
    classifications: &[(f32, CetaceanUnit)],
) -> Vec<(f32, f32, CetaceanUnit)> {
    if classifications.is_empty() {
        return Vec::new();
    }

    let mut segments = Vec::new();
    let mut seg_start = classifications[0].0;
    let mut seg_unit = classifications[0].1;
    let mut seg_end = seg_start;

    for &(time, unit) in &classifications[1..] {
        if unit == seg_unit {
            seg_end = time;
        } else {
            segments.push((seg_start, seg_end, seg_unit));
            seg_start = time;
            seg_unit = unit;
            seg_end = time;
        }
    }

    segments.push((seg_start, seg_end, seg_unit));
    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classifier_creation() {
        let classifier = CetaceanClassifier::new(CetaceanConfig::default());
        assert_eq!(classifier.config.sample_rate, 44100);
    }

    #[test]
    fn test_synthetic_click() {
        let mut classifier = CetaceanClassifier::new(CetaceanConfig::default());

        // Generate synthetic click (short impulse)
        let mut samples = vec![0.0f32; 4410]; // 100ms at 44.1kHz
        samples[2200] = 1.0; // Impulse at 50ms
        samples[2201] = 0.5;
        samples[2202] = -0.3;

        let results = classifier.classify_audio(&samples);

        // Should detect something around the click
        assert!(!results.is_empty(), "Should detect the click");
    }

    #[test]
    fn test_synthetic_whistle() {
        let mut classifier = CetaceanClassifier::new(CetaceanConfig::default());

        // Generate synthetic whistle (sustained tone with upsweep)
        let sample_rate = 44100.0;
        let duration = 0.5; // 500ms
        let samples: Vec<f32> = (0..(sample_rate * duration) as usize)
            .map(|i| {
                let t = i as f32 / sample_rate;
                let freq = 2000.0 + t * 2000.0; // 2kHz → 4kHz upsweep
                (2.0 * std::f32::consts::PI * freq * t).sin() * 0.5
            })
            .collect();

        let results = classifier.classify_audio(&samples);

        // Should classify as whistle with upsweep
        let whistles: Vec<_> = results
            .iter()
            .filter(|(_, u)| u.manner == CetaceanManner::Whistle)
            .collect();

        assert!(!whistles.is_empty(), "Should detect whistle");
    }

    #[test]
    fn test_collapse_segments() {
        let classifications = vec![
            (
                0.0,
                CetaceanUnit::new(CetaceanManner::Click, CetaceanPlace::Flat),
            ),
            (
                0.01,
                CetaceanUnit::new(CetaceanManner::Click, CetaceanPlace::Flat),
            ),
            (
                0.02,
                CetaceanUnit::new(CetaceanManner::Whistle, CetaceanPlace::Upsweep),
            ),
            (
                0.03,
                CetaceanUnit::new(CetaceanManner::Whistle, CetaceanPlace::Upsweep),
            ),
            (
                0.04,
                CetaceanUnit::new(CetaceanManner::Whistle, CetaceanPlace::Upsweep),
            ),
        ];

        let segments = collapse_to_segments(&classifications);

        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].2.manner, CetaceanManner::Click);
        assert_eq!(segments[1].2.manner, CetaceanManner::Whistle);
    }
}
