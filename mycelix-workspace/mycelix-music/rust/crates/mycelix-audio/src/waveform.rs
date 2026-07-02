// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Waveform generation for visualization

use crate::{AudioBuffer, AudioResult};
use serde::{Deserialize, Serialize};

/// Waveform data for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaveformData {
    /// Peak values (positive peaks)
    pub peaks: Vec<f32>,
    /// RMS values for each segment
    pub rms: Vec<f32>,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Sample rate of source audio
    pub sample_rate: u32,
    /// Number of samples per peak
    pub samples_per_peak: usize,
}

impl WaveformData {
    /// Get waveform as min/max pairs for drawing
    pub fn as_min_max(&self) -> Vec<(f32, f32)> {
        self.peaks.iter().map(|&p| (-p, p)).collect()
    }

    /// Downsample to target number of points
    pub fn downsample(&self, target_points: usize) -> WaveformData {
        if target_points >= self.peaks.len() {
            return self.clone();
        }

        let ratio = self.peaks.len() as f32 / target_points as f32;
        let mut new_peaks = Vec::with_capacity(target_points);
        let mut new_rms = Vec::with_capacity(target_points);

        for i in 0..target_points {
            let start = (i as f32 * ratio) as usize;
            let end = ((i + 1) as f32 * ratio) as usize;
            let end = end.min(self.peaks.len());

            if start < end {
                let peak = self.peaks[start..end]
                    .iter()
                    .cloned()
                    .fold(0.0f32, f32::max);
                let rms = self.rms[start..end]
                    .iter()
                    .cloned()
                    .fold(0.0f32, f32::max);

                new_peaks.push(peak);
                new_rms.push(rms);
            }
        }

        WaveformData {
            peaks: new_peaks,
            rms: new_rms,
            duration_ms: self.duration_ms,
            sample_rate: self.sample_rate,
            samples_per_peak: (self.samples_per_peak as f32 * ratio) as usize,
        }
    }

    /// Get peak at a specific time
    pub fn peak_at_time(&self, time_ms: u64) -> Option<f32> {
        let index = (time_ms as f64 / self.duration_ms as f64 * self.peaks.len() as f64) as usize;
        self.peaks.get(index).copied()
    }

    /// Get a slice of the waveform
    pub fn slice(&self, start_ms: u64, end_ms: u64) -> WaveformData {
        let total_points = self.peaks.len();
        let duration = self.duration_ms as f64;

        let start_idx = ((start_ms as f64 / duration) * total_points as f64) as usize;
        let end_idx = ((end_ms as f64 / duration) * total_points as f64) as usize;

        let start_idx = start_idx.min(total_points);
        let end_idx = end_idx.min(total_points);

        WaveformData {
            peaks: self.peaks[start_idx..end_idx].to_vec(),
            rms: self.rms[start_idx..end_idx].to_vec(),
            duration_ms: end_ms - start_ms,
            sample_rate: self.sample_rate,
            samples_per_peak: self.samples_per_peak,
        }
    }
}

/// Waveform generator configuration
#[derive(Debug, Clone)]
pub struct WaveformConfig {
    /// Target number of data points
    pub target_points: usize,
    /// Whether to compute RMS values
    pub compute_rms: bool,
    /// Whether to normalize peaks
    pub normalize: bool,
}

impl Default for WaveformConfig {
    fn default() -> Self {
        Self {
            target_points: 1800, // Good for typical player widths
            compute_rms: true,
            normalize: true,
        }
    }
}

/// Waveform generator
pub struct WaveformGenerator {
    config: WaveformConfig,
}

impl WaveformGenerator {
    pub fn new(config: WaveformConfig) -> Self {
        Self { config }
    }

    /// Generate waveform from audio buffer
    pub fn generate(&self, buffer: &AudioBuffer) -> AudioResult<WaveformData> {
        // Convert to mono if needed
        let mono = if buffer.channels > 1 {
            buffer.to_mono()
        } else {
            buffer.clone()
        };

        let sample_count = mono.samples.len();
        let samples_per_peak = (sample_count / self.config.target_points).max(1);
        let num_peaks = (sample_count + samples_per_peak - 1) / samples_per_peak;

        let mut peaks = Vec::with_capacity(num_peaks);
        let mut rms_values = Vec::with_capacity(num_peaks);

        for chunk in mono.samples.chunks(samples_per_peak) {
            // Calculate peak
            let peak = chunk.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
            peaks.push(peak);

            // Calculate RMS if requested
            if self.config.compute_rms {
                let rms = (chunk.iter().map(|s| s * s).sum::<f32>() / chunk.len() as f32).sqrt();
                rms_values.push(rms);
            }
        }

        // Normalize if requested
        if self.config.normalize {
            let max_peak = peaks.iter().cloned().fold(0.0f32, f32::max);
            if max_peak > 0.0 {
                for peak in &mut peaks {
                    *peak /= max_peak;
                }
                if self.config.compute_rms {
                    for rms in &mut rms_values {
                        *rms /= max_peak;
                    }
                }
            }
        }

        let duration_ms = (sample_count as f64 / mono.sample_rate as f64 * 1000.0) as u64;

        Ok(WaveformData {
            peaks,
            rms: rms_values,
            duration_ms,
            sample_rate: mono.sample_rate,
            samples_per_peak,
        })
    }

    /// Generate waveform from raw samples
    pub fn generate_from_samples(
        &self,
        samples: &[f32],
        sample_rate: u32,
    ) -> WaveformData {
        let sample_count = samples.len();
        let samples_per_peak = (sample_count / self.config.target_points).max(1);
        let num_peaks = (sample_count + samples_per_peak - 1) / samples_per_peak;

        let mut peaks = Vec::with_capacity(num_peaks);
        let mut rms_values = Vec::with_capacity(num_peaks);

        for chunk in samples.chunks(samples_per_peak) {
            let peak = chunk.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
            peaks.push(peak);

            if self.config.compute_rms {
                let rms = (chunk.iter().map(|s| s * s).sum::<f32>() / chunk.len() as f32).sqrt();
                rms_values.push(rms);
            }
        }

        if self.config.normalize {
            let max_peak = peaks.iter().cloned().fold(0.0f32, f32::max);
            if max_peak > 0.0 {
                for peak in &mut peaks {
                    *peak /= max_peak;
                }
                if self.config.compute_rms {
                    for rms in &mut rms_values {
                        *rms /= max_peak;
                    }
                }
            }
        }

        let duration_ms = (sample_count as f64 / sample_rate as f64 * 1000.0) as u64;

        WaveformData {
            peaks,
            rms: rms_values,
            duration_ms,
            sample_rate,
            samples_per_peak,
        }
    }
}

impl Default for WaveformGenerator {
    fn default() -> Self {
        Self::new(WaveformConfig::default())
    }
}

/// Generate a simple waveform overview
pub fn generate_overview(samples: &[f32], points: usize) -> Vec<f32> {
    if samples.is_empty() || points == 0 {
        return Vec::new();
    }

    let samples_per_point = (samples.len() / points).max(1);
    samples
        .chunks(samples_per_point)
        .map(|chunk| chunk.iter().map(|s| s.abs()).fold(0.0f32, f32::max))
        .collect()
}

/// Smooth waveform data using moving average
pub fn smooth_waveform(data: &[f32], window_size: usize) -> Vec<f32> {
    if window_size <= 1 || data.is_empty() {
        return data.to_vec();
    }

    let half_window = window_size / 2;
    let mut result = Vec::with_capacity(data.len());

    for i in 0..data.len() {
        let start = i.saturating_sub(half_window);
        let end = (i + half_window + 1).min(data.len());
        let sum: f32 = data[start..end].iter().sum();
        result.push(sum / (end - start) as f32);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ChannelLayout;

    #[test]
    fn test_waveform_generation() {
        let samples: Vec<f32> = (0..44100)
            .map(|i| (i as f32 / 44100.0 * 440.0 * 2.0 * std::f32::consts::PI).sin())
            .collect();

        let buffer = AudioBuffer::from_samples(samples, 44100, 1, ChannelLayout::Mono);
        let generator = WaveformGenerator::default();
        let waveform = generator.generate(&buffer).unwrap();

        assert!(waveform.peaks.len() > 0);
        // Allow slight overshoot due to ceil division in chunking
        assert!(waveform.peaks.len() <= 1900);
        assert_eq!(waveform.duration_ms, 1000);
    }

    #[test]
    fn test_waveform_downsample() {
        let data = WaveformData {
            peaks: vec![1.0; 1000],
            rms: vec![0.5; 1000],
            duration_ms: 10000,
            sample_rate: 44100,
            samples_per_peak: 44,
        };

        let downsampled = data.downsample(100);
        assert_eq!(downsampled.peaks.len(), 100);
    }

    #[test]
    fn test_smooth_waveform() {
        let data = vec![0.0, 1.0, 0.0, 1.0, 0.0];
        let smoothed = smooth_waveform(&data, 3);
        assert_eq!(smoothed.len(), data.len());
        // Middle values should be smoothed
        assert!(smoothed[2] > 0.0 && smoothed[2] < 1.0);
    }
}
