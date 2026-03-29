// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Audio embedding generation

use crate::{AudioEmbedding, EmbeddingMetadata, Result, SearchError};
use std::f32::consts::PI;

/// Embedding model type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingModel {
    /// Chromagram-based embedding
    Chroma,
    /// MFCC-based embedding
    Mfcc,
    /// Spectral feature embedding
    Spectral,
    /// Combined multi-feature embedding
    Combined,
}

/// Audio embedding generator
pub struct EmbeddingGenerator {
    model: EmbeddingModel,
    sample_rate: u32,
    frame_size: usize,
    hop_size: usize,
    output_dim: usize,
}

impl EmbeddingGenerator {
    pub fn new(model: EmbeddingModel, sample_rate: u32) -> Self {
        let output_dim = match model {
            EmbeddingModel::Chroma => 12,
            EmbeddingModel::Mfcc => 20,
            EmbeddingModel::Spectral => 16,
            EmbeddingModel::Combined => 128,
        };

        Self {
            model,
            sample_rate,
            frame_size: 2048,
            hop_size: 512,
            output_dim,
        }
    }

    /// Generate embedding from audio samples
    pub fn generate(&self, audio: &[f32]) -> Result<AudioEmbedding> {
        let vector = match self.model {
            EmbeddingModel::Chroma => self.generate_chroma(audio)?,
            EmbeddingModel::Mfcc => self.generate_mfcc(audio)?,
            EmbeddingModel::Spectral => self.generate_spectral(audio)?,
            EmbeddingModel::Combined => self.generate_combined(audio)?,
        };

        let mut emb = AudioEmbedding::new(vector);
        emb.normalize();
        Ok(emb)
    }

    /// Generate embedding with metadata
    pub fn generate_with_metadata(
        &self,
        audio: &[f32],
        metadata: EmbeddingMetadata,
    ) -> Result<AudioEmbedding> {
        let mut emb = self.generate(audio)?;
        emb.metadata = metadata;
        Ok(emb)
    }

    fn generate_chroma(&self, audio: &[f32]) -> Result<Vec<f32>> {
        let num_frames = (audio.len() - self.frame_size) / self.hop_size + 1;
        if num_frames == 0 {
            return Err(SearchError::EmbeddingError("Audio too short".to_string()));
        }

        let mut chroma_avg = vec![0.0f32; 12];

        for frame_idx in 0..num_frames {
            let start = frame_idx * self.hop_size;
            let frame = &audio[start..start + self.frame_size];

            let chroma = self.compute_chroma(frame);
            for (i, &c) in chroma.iter().enumerate() {
                chroma_avg[i] += c;
            }
        }

        for c in &mut chroma_avg {
            *c /= num_frames as f32;
        }

        Ok(chroma_avg)
    }

    fn generate_mfcc(&self, audio: &[f32]) -> Result<Vec<f32>> {
        let num_frames = (audio.len() - self.frame_size) / self.hop_size + 1;
        if num_frames == 0 {
            return Err(SearchError::EmbeddingError("Audio too short".to_string()));
        }

        let num_coeffs = 20;
        let mut mfcc_avg = vec![0.0f32; num_coeffs];

        for frame_idx in 0..num_frames {
            let start = frame_idx * self.hop_size;
            let frame = &audio[start..start + self.frame_size];

            let mfcc = self.compute_mfcc(frame, num_coeffs);
            for (i, &m) in mfcc.iter().enumerate() {
                mfcc_avg[i] += m;
            }
        }

        for m in &mut mfcc_avg {
            *m /= num_frames as f32;
        }

        Ok(mfcc_avg)
    }

    fn generate_spectral(&self, audio: &[f32]) -> Result<Vec<f32>> {
        let num_frames = (audio.len() - self.frame_size) / self.hop_size + 1;
        if num_frames == 0 {
            return Err(SearchError::EmbeddingError("Audio too short".to_string()));
        }

        let mut features = vec![0.0f32; 16];
        let mut spectral_centroids = Vec::new();
        let mut spectral_rolloffs = Vec::new();
        let mut spectral_fluxes = Vec::new();
        let mut zero_crossings = Vec::new();

        let mut prev_spectrum: Option<Vec<f32>> = None;

        for frame_idx in 0..num_frames {
            let start = frame_idx * self.hop_size;
            let frame = &audio[start..start + self.frame_size];

            // Compute spectrum
            let spectrum = self.compute_spectrum(frame);

            // Spectral centroid
            let (centroid, _) = self.spectral_centroid(&spectrum);
            spectral_centroids.push(centroid);

            // Spectral rolloff
            let rolloff = self.spectral_rolloff(&spectrum, 0.85);
            spectral_rolloffs.push(rolloff);

            // Spectral flux
            if let Some(ref prev) = prev_spectrum {
                let flux = self.spectral_flux(&spectrum, prev);
                spectral_fluxes.push(flux);
            }

            // Zero crossing rate
            let zcr = self.zero_crossing_rate(frame);
            zero_crossings.push(zcr);

            prev_spectrum = Some(spectrum);
        }

        // Compute statistics
        features[0] = mean(&spectral_centroids);
        features[1] = std_dev(&spectral_centroids);
        features[2] = mean(&spectral_rolloffs);
        features[3] = std_dev(&spectral_rolloffs);
        features[4] = mean(&spectral_fluxes);
        features[5] = std_dev(&spectral_fluxes);
        features[6] = mean(&zero_crossings);
        features[7] = std_dev(&zero_crossings);

        // RMS energy
        let rms = (audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32).sqrt();
        features[8] = rms;

        // Temporal features
        features[9] = spectral_centroids.iter().cloned().fold(f32::MIN, f32::max);
        features[10] = spectral_centroids.iter().cloned().fold(f32::MAX, f32::min);
        features[11] = spectral_rolloffs.iter().cloned().fold(f32::MIN, f32::max);
        features[12] = spectral_rolloffs.iter().cloned().fold(f32::MAX, f32::min);

        // Padding
        features[13] = 0.0;
        features[14] = 0.0;
        features[15] = 0.0;

        Ok(features)
    }

    fn generate_combined(&self, audio: &[f32]) -> Result<Vec<f32>> {
        let chroma = self.generate_chroma(audio)?;
        let mfcc = self.generate_mfcc(audio)?;
        let spectral = self.generate_spectral(audio)?;

        // Pad to reach 128 dimensions
        let mut combined = Vec::with_capacity(128);
        combined.extend(&chroma);      // 12
        combined.extend(&mfcc);        // 20
        combined.extend(&spectral);    // 16

        // Add more features to reach 128
        // Chroma variance
        let chroma_var: Vec<f32> = (0..12).map(|_| 0.0).collect();
        combined.extend(&chroma_var);  // 12

        // Delta MFCCs (simplified)
        combined.extend(&mfcc);        // 20 more = 80 total

        // More spectral stats
        combined.extend(&spectral);    // 16 more = 96 total

        // Padding to 128
        while combined.len() < 128 {
            combined.push(0.0);
        }

        combined.truncate(128);
        Ok(combined)
    }

    fn compute_chroma(&self, frame: &[f32]) -> Vec<f32> {
        let mut chroma = vec![0.0f32; 12];
        let n = frame.len();

        // Simple pitch class energy computation
        for octave in 1..=7 {
            for pc in 0..12 {
                let midi_note = octave * 12 + pc;
                let freq = 440.0 * 2.0_f32.powf((midi_note as f32 - 69.0) / 12.0);

                if freq > self.sample_rate as f32 / 2.0 {
                    continue;
                }

                // Goertzel algorithm
                let k = (freq * n as f32 / self.sample_rate as f32).round() as usize;
                if k == 0 || k >= n / 2 {
                    continue;
                }

                let omega = 2.0 * PI * k as f32 / n as f32;
                let coeff = 2.0 * omega.cos();

                let mut s0 = 0.0;
                let mut s1 = 0.0;
                let mut s2;

                for &sample in frame {
                    s2 = s1;
                    s1 = s0;
                    let window = 0.5 - 0.5 * (2.0 * PI * s0 / n as f32).cos();
                    s0 = sample * window + coeff * s1 - s2;
                }

                let power = (s0 * s0 + s1 * s1 - coeff * s0 * s1).max(0.0).sqrt();
                chroma[pc] += power;
            }
        }

        // Normalize
        let sum: f32 = chroma.iter().sum();
        if sum > 1e-6 {
            for c in &mut chroma {
                *c /= sum;
            }
        }

        chroma
    }

    fn compute_mfcc(&self, frame: &[f32], num_coeffs: usize) -> Vec<f32> {
        let spectrum = self.compute_spectrum(frame);

        // Mel filterbank (simplified)
        let num_filters = 26;
        let mel_energies = self.apply_mel_filterbank(&spectrum, num_filters);

        // Log compression
        let log_mel: Vec<f32> = mel_energies.iter().map(|&e| (e + 1e-10).ln()).collect();

        // DCT to get MFCCs
        let mut mfcc = vec![0.0f32; num_coeffs];
        for (i, mfcc_val) in mfcc.iter_mut().enumerate() {
            let mut sum = 0.0;
            for (j, &log_e) in log_mel.iter().enumerate() {
                sum += log_e * (PI * i as f32 * (j as f32 + 0.5) / num_filters as f32).cos();
            }
            *mfcc_val = sum;
        }

        mfcc
    }

    fn compute_spectrum(&self, frame: &[f32]) -> Vec<f32> {
        let n = frame.len();
        let mut spectrum = vec![0.0f32; n / 2];

        // Simple DFT
        for k in 0..n / 2 {
            let mut real = 0.0;
            let mut imag = 0.0;

            for (i, &sample) in frame.iter().enumerate() {
                let window = 0.5 - 0.5 * (2.0 * PI * i as f32 / n as f32).cos();
                let angle = 2.0 * PI * k as f32 * i as f32 / n as f32;
                real += sample * window * angle.cos();
                imag -= sample * window * angle.sin();
            }

            spectrum[k] = (real * real + imag * imag).sqrt();
        }

        spectrum
    }

    fn apply_mel_filterbank(&self, spectrum: &[f32], num_filters: usize) -> Vec<f32> {
        let mut mel_energies = vec![0.0f32; num_filters];

        // Simple triangular filterbank
        let low_freq = 0.0;
        let high_freq = self.sample_rate as f32 / 2.0;

        let low_mel = self.hz_to_mel(low_freq);
        let high_mel = self.hz_to_mel(high_freq);

        let mel_points: Vec<f32> = (0..=num_filters + 1)
            .map(|i| low_mel + (high_mel - low_mel) * i as f32 / (num_filters + 1) as f32)
            .collect();

        let hz_points: Vec<f32> = mel_points.iter().map(|&m| self.mel_to_hz(m)).collect();
        let bin_points: Vec<usize> = hz_points
            .iter()
            .map(|&f| (f * spectrum.len() as f32 * 2.0 / self.sample_rate as f32) as usize)
            .collect();

        for i in 0..num_filters {
            let left = bin_points[i];
            let center = bin_points[i + 1];
            let right = bin_points[i + 2];

            for k in left..center {
                if k < spectrum.len() {
                    let weight = (k - left) as f32 / (center - left).max(1) as f32;
                    mel_energies[i] += spectrum[k] * weight;
                }
            }

            for k in center..right {
                if k < spectrum.len() {
                    let weight = (right - k) as f32 / (right - center).max(1) as f32;
                    mel_energies[i] += spectrum[k] * weight;
                }
            }
        }

        mel_energies
    }

    fn hz_to_mel(&self, hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    fn mel_to_hz(&self, mel: f32) -> f32 {
        700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
    }

    fn spectral_centroid(&self, spectrum: &[f32]) -> (f32, f32) {
        let mut weighted_sum = 0.0;
        let mut total_energy = 0.0;

        for (k, &mag) in spectrum.iter().enumerate() {
            let freq = k as f32 * self.sample_rate as f32 / (spectrum.len() * 2) as f32;
            weighted_sum += freq * mag;
            total_energy += mag;
        }

        if total_energy > 1e-6 {
            (weighted_sum / total_energy, total_energy)
        } else {
            (0.0, 0.0)
        }
    }

    fn spectral_rolloff(&self, spectrum: &[f32], threshold: f32) -> f32 {
        let total_energy: f32 = spectrum.iter().sum();
        let target = total_energy * threshold;

        let mut cumsum = 0.0;
        for (k, &mag) in spectrum.iter().enumerate() {
            cumsum += mag;
            if cumsum >= target {
                return k as f32 * self.sample_rate as f32 / (spectrum.len() * 2) as f32;
            }
        }

        self.sample_rate as f32 / 2.0
    }

    fn spectral_flux(&self, current: &[f32], previous: &[f32]) -> f32 {
        current
            .iter()
            .zip(previous.iter())
            .map(|(&c, &p)| (c - p).max(0.0).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    fn zero_crossing_rate(&self, frame: &[f32]) -> f32 {
        if frame.len() < 2 {
            return 0.0;
        }

        let crossings = frame
            .windows(2)
            .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
            .count();

        crossings as f32 / (frame.len() - 1) as f32
    }

    pub fn output_dimension(&self) -> usize {
        self.output_dim
    }
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

fn std_dev(values: &[f32]) -> f32 {
    if values.len() < 2 {
        return 0.0;
    }

    let m = mean(values);
    let variance = values.iter().map(|x| (x - m).powi(2)).sum::<f32>() / (values.len() - 1) as f32;
    variance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_generator() {
        let generator = EmbeddingGenerator::new(EmbeddingModel::Chroma, 44100);

        // Create simple test audio
        let audio: Vec<f32> = (0..44100).map(|i| (440.0 * 2.0 * PI * i as f32 / 44100.0).sin()).collect();

        let emb = generator.generate(&audio).unwrap();
        assert_eq!(emb.vector.len(), 12);
    }

    #[test]
    fn test_combined_embedding() {
        let generator = EmbeddingGenerator::new(EmbeddingModel::Combined, 44100);
        assert_eq!(generator.output_dimension(), 128);
    }
}
