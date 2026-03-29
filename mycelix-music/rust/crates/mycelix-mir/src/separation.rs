// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Source separation (vocals, drums, bass, other)

use crate::{MirError, Result};
use std::f32::consts::PI;

/// Source separation result
#[derive(Debug, Clone)]
pub struct SeparatedSources {
    /// Vocal track
    pub vocals: Vec<f32>,
    /// Drum track
    pub drums: Vec<f32>,
    /// Bass track
    pub bass: Vec<f32>,
    /// Other instruments
    pub other: Vec<f32>,
    /// Sample rate
    pub sample_rate: u32,
}

/// Source separator using spectral methods
/// Note: Production would use deep learning models like Demucs/Spleeter
pub struct SourceSeparator {
    sample_rate: u32,
    frame_size: usize,
    hop_size: usize,
}

impl SourceSeparator {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            frame_size: 4096,
            hop_size: 1024,
        }
    }

    /// Separate audio into sources
    /// This is a simplified spectral method - production would use ML models
    pub fn separate(&self, audio: &[f32]) -> Result<SeparatedSources> {
        if audio.len() < self.frame_size {
            return Err(MirError::BufferTooSmall {
                needed: self.frame_size,
                actual: audio.len(),
            });
        }

        // Process in STFT domain
        let num_frames = (audio.len() - self.frame_size) / self.hop_size + 1;
        let freq_bins = self.frame_size / 2 + 1;

        // Compute STFT
        let spectrogram = self.compute_stft(audio);

        // Create separation masks
        let vocal_mask = self.create_vocal_mask(freq_bins, num_frames, &spectrogram);
        let drum_mask = self.create_drum_mask(freq_bins, num_frames, &spectrogram);
        let bass_mask = self.create_bass_mask(freq_bins, num_frames);

        // Apply masks and inverse STFT
        let vocals = self.apply_mask_and_istft(&spectrogram, &vocal_mask, audio.len());
        let drums = self.apply_mask_and_istft(&spectrogram, &drum_mask, audio.len());
        let bass = self.apply_mask_and_istft(&spectrogram, &bass_mask, audio.len());

        // Other = original - (vocals + drums + bass)
        let other: Vec<f32> = audio
            .iter()
            .zip(vocals.iter())
            .zip(drums.iter())
            .zip(bass.iter())
            .map(|(((a, v), d), b)| a - v - d - b)
            .collect();

        Ok(SeparatedSources {
            vocals,
            drums,
            bass,
            other,
            sample_rate: self.sample_rate,
        })
    }

    fn compute_stft(&self, audio: &[f32]) -> Vec<Vec<(f32, f32)>> {
        let num_frames = (audio.len() - self.frame_size) / self.hop_size + 1;
        let mut spectrogram = Vec::with_capacity(num_frames);

        for frame_idx in 0..num_frames {
            let start = frame_idx * self.hop_size;
            let frame = &audio[start..start + self.frame_size];

            // Apply window and compute FFT
            let spectrum = self.fft_frame(frame);
            spectrogram.push(spectrum);
        }

        spectrogram
    }

    fn fft_frame(&self, frame: &[f32]) -> Vec<(f32, f32)> {
        let n = frame.len();
        let mut spectrum = Vec::with_capacity(n / 2 + 1);

        for k in 0..=n / 2 {
            let mut real = 0.0;
            let mut imag = 0.0;

            for (i, &sample) in frame.iter().enumerate() {
                let window = 0.5 - 0.5 * (2.0 * PI * i as f32 / n as f32).cos();
                let angle = 2.0 * PI * k as f32 * i as f32 / n as f32;
                let windowed = sample * window;
                real += windowed * angle.cos();
                imag -= windowed * angle.sin();
            }

            spectrum.push((real, imag));
        }

        spectrum
    }

    fn create_vocal_mask(
        &self,
        freq_bins: usize,
        num_frames: usize,
        spectrogram: &[Vec<(f32, f32)>],
    ) -> Vec<Vec<f32>> {
        // Vocals typically 100Hz - 10kHz, centered
        // Use harmonic-percussive separation concept
        let mut mask = vec![vec![0.0f32; freq_bins]; num_frames];

        let low_bin = (100.0 * self.frame_size as f32 / self.sample_rate as f32) as usize;
        let high_bin = (10000.0 * self.frame_size as f32 / self.sample_rate as f32) as usize;

        for (frame_idx, spectrum) in spectrogram.iter().enumerate() {
            // Compute harmonic emphasis (time-direction smoothing)
            for k in low_bin..high_bin.min(freq_bins) {
                // Check temporal consistency (vocals are harmonic)
                let mut harmonic_score = 0.0;
                let window = 3;

                for dt in 0..window {
                    let prev_idx = frame_idx.saturating_sub(dt);
                    let next_idx = (frame_idx + dt).min(spectrogram.len() - 1);

                    if prev_idx < spectrogram.len() && next_idx < spectrogram.len() {
                        let (r1, i1) = spectrogram[prev_idx][k];
                        let (r2, i2) = spectrogram[next_idx][k];
                        let mag1 = (r1 * r1 + i1 * i1).sqrt();
                        let mag2 = (r2 * r2 + i2 * i2).sqrt();

                        let (r, i) = spectrum[k];
                        let mag = (r * r + i * i).sqrt();

                        if mag > 0.0 {
                            let consistency = (mag1 + mag2) / (2.0 * mag);
                            harmonic_score += consistency.min(2.0);
                        }
                    }
                }

                harmonic_score /= window as f32;
                mask[frame_idx][k] = harmonic_score.clamp(0.0, 1.0);
            }
        }

        mask
    }

    fn create_drum_mask(
        &self,
        freq_bins: usize,
        num_frames: usize,
        spectrogram: &[Vec<(f32, f32)>],
    ) -> Vec<Vec<f32>> {
        // Drums are percussive - high frequency-direction variation
        let mut mask = vec![vec![0.0f32; freq_bins]; num_frames];

        for (frame_idx, spectrum) in spectrogram.iter().enumerate() {
            // Detect transients
            let onset_strength = if frame_idx > 0 {
                let prev = &spectrogram[frame_idx - 1];
                spectrum
                    .iter()
                    .zip(prev.iter())
                    .map(|((r1, i1), (r2, i2))| {
                        let mag1 = (r1 * r1 + i1 * i1).sqrt();
                        let mag2 = (r2 * r2 + i2 * i2).sqrt();
                        (mag1 - mag2).max(0.0)
                    })
                    .sum::<f32>()
            } else {
                0.0
            };

            let is_onset = onset_strength > 1.0;

            for k in 0..freq_bins {
                if is_onset {
                    // Percussive content at onsets
                    let (r, i) = spectrum[k];
                    let mag = (r * r + i * i).sqrt();
                    mask[frame_idx][k] = (mag * 0.5).min(1.0);
                }
            }
        }

        mask
    }

    fn create_bass_mask(&self, freq_bins: usize, num_frames: usize) -> Vec<Vec<f32>> {
        // Bass is simply low frequency content < 250Hz
        let mut mask = vec![vec![0.0f32; freq_bins]; num_frames];

        let bass_cutoff = (250.0 * self.frame_size as f32 / self.sample_rate as f32) as usize;

        for frame in &mut mask {
            for k in 0..bass_cutoff.min(freq_bins) {
                // Smooth rolloff
                let freq_ratio = k as f32 / bass_cutoff as f32;
                frame[k] = (1.0 - freq_ratio).max(0.0);
            }
        }

        mask
    }

    fn apply_mask_and_istft(
        &self,
        spectrogram: &[Vec<(f32, f32)>],
        mask: &[Vec<f32>],
        output_len: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; output_len];
        let mut window_sum = vec![0.0f32; output_len];

        for (frame_idx, (spectrum, frame_mask)) in spectrogram.iter().zip(mask.iter()).enumerate() {
            // Apply mask in frequency domain
            let masked: Vec<(f32, f32)> = spectrum
                .iter()
                .zip(frame_mask.iter())
                .map(|(&(r, i), &m)| (r * m, i * m))
                .collect();

            // Inverse FFT
            let frame = self.ifft_frame(&masked);

            // Overlap-add
            let start = frame_idx * self.hop_size;
            for (i, &sample) in frame.iter().enumerate() {
                if start + i < output_len {
                    let window = 0.5 - 0.5 * (2.0 * PI * i as f32 / self.frame_size as f32).cos();
                    output[start + i] += sample * window;
                    window_sum[start + i] += window * window;
                }
            }
        }

        // Normalize by window sum
        for (i, sample) in output.iter_mut().enumerate() {
            if window_sum[i] > 1e-6 {
                *sample /= window_sum[i];
            }
        }

        output
    }

    fn ifft_frame(&self, spectrum: &[(f32, f32)]) -> Vec<f32> {
        let n = (spectrum.len() - 1) * 2;
        let mut output = vec![0.0f32; n];

        for (i, sample) in output.iter_mut().enumerate() {
            let mut sum = 0.0;

            for (k, &(real, imag)) in spectrum.iter().enumerate() {
                let angle = 2.0 * PI * k as f32 * i as f32 / n as f32;

                // Real part of complex exponential
                sum += real * angle.cos() - imag * angle.sin();

                // Mirror for negative frequencies (except DC and Nyquist)
                if k > 0 && k < spectrum.len() - 1 {
                    sum += real * angle.cos() + imag * angle.sin();
                }
            }

            *sample = sum / n as f32;
        }

        output
    }
}

impl Default for SourceSeparator {
    fn default() -> Self {
        Self::new(44100)
    }
}

/// Vocal remover (simplified karaoke effect)
pub struct VocalRemover {
    sample_rate: u32,
}

impl VocalRemover {
    pub fn new(sample_rate: u32) -> Self {
        Self { sample_rate }
    }

    /// Remove vocals using stereo cancellation
    /// Note: Only works well with center-panned vocals
    pub fn remove_vocals(&self, left: &[f32], right: &[f32]) -> Vec<f32> {
        // Simple center cancellation: output = (L - R) / 2
        left.iter()
            .zip(right.iter())
            .map(|(&l, &r)| (l - r) * 0.5)
            .collect()
    }

    /// Extract vocals using stereo isolation
    pub fn extract_vocals(&self, left: &[f32], right: &[f32]) -> Vec<f32> {
        // Extract center: output = (L + R) / 2 - sides
        let mid: Vec<f32> = left.iter().zip(right.iter()).map(|(&l, &r)| (l + r) * 0.5).collect();

        let side: Vec<f32> = left.iter().zip(right.iter()).map(|(&l, &r)| (l - r) * 0.5).collect();

        // Vocals are typically more in mid than side
        mid.iter()
            .zip(side.iter())
            .map(|(&m, &s)| (m - s.abs() * 0.5).max(0.0))
            .collect()
    }
}

impl Default for VocalRemover {
    fn default() -> Self {
        Self::new(44100)
    }
}

/// Harmonic-Percussive Source Separation (HPSS)
pub struct HPSS {
    sample_rate: u32,
    frame_size: usize,
    hop_size: usize,
    /// Harmonic/percussive kernel sizes
    h_kernel: usize,
    p_kernel: usize,
}

impl HPSS {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            frame_size: 4096,
            hop_size: 1024,
            h_kernel: 17, // Horizontal (time) median filter
            p_kernel: 17, // Vertical (frequency) median filter
        }
    }

    /// Separate into harmonic and percussive components
    pub fn separate(&self, audio: &[f32]) -> (Vec<f32>, Vec<f32>) {
        // For this simplified version, use filtering approach
        // Production would use proper STFT + median filtering

        // Low-pass for harmonic (smooth over time)
        let harmonic = self.temporal_smooth(audio);

        // High-pass for percussive (transients)
        let percussive: Vec<f32> = audio
            .iter()
            .zip(harmonic.iter())
            .map(|(&a, &h)| a - h)
            .collect();

        (harmonic, percussive)
    }

    fn temporal_smooth(&self, audio: &[f32]) -> Vec<f32> {
        let kernel_size = 64; // Smooth over ~1.5ms at 44.1kHz
        let mut output = vec![0.0f32; audio.len()];

        for i in 0..audio.len() {
            let start = i.saturating_sub(kernel_size / 2);
            let end = (i + kernel_size / 2).min(audio.len());
            let sum: f32 = audio[start..end].iter().sum();
            output[i] = sum / (end - start) as f32;
        }

        output
    }
}

impl Default for HPSS {
    fn default() -> Self {
        Self::new(44100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocal_remover() {
        let remover = VocalRemover::new(44100);

        // Center-panned mono signal should cancel
        let left = vec![1.0, 0.5, -0.5, -1.0];
        let right = vec![1.0, 0.5, -0.5, -1.0];

        let result = remover.remove_vocals(&left, &right);
        for &sample in &result {
            assert!(sample.abs() < 0.001);
        }
    }

    #[test]
    fn test_hpss() {
        let hpss = HPSS::new(44100);

        // Simple test signal
        let audio = vec![0.0; 1024];
        let (harmonic, percussive) = hpss.separate(&audio);

        assert_eq!(harmonic.len(), audio.len());
        assert_eq!(percussive.len(), audio.len());
    }
}
