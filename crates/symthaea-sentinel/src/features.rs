// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Audio feature extraction for temporal pattern recognition.
//!
//! This module provides:
//! - `MelFilterbank`: Mel spectrogram computation
//! - `FeatureExtractor`: Complete audio feature extraction pipeline
//! - `AudioFeatures`: Container for extracted features

use std::f32::consts::PI;

// =============================================================================
// Constants
// =============================================================================

pub const MEL_BANDS: usize = 40;
pub const NUM_MFCC: usize = 13;
pub const FFT_SIZE: usize = 2048;
pub const HOP_SIZE: usize = 1024;
pub const SAMPLE_RATE: f32 = 44100.0;
pub const CONTROL_RATE: f32 = SAMPLE_RATE / HOP_SIZE as f32;

/// Frequency signature bins (expanded 8-bin for BPM detection)
pub const FREQ_BINS: [f32; 8] = [
    0.25, // 15 BPM - very slow ambient
    0.5,  // 30 BPM - slow
    1.0,  // 60 BPM - ballad
    1.5,  // 90 BPM - moderate
    2.0,  // 120 BPM - dance/pop
    3.0,  // 180 BPM - fast
    4.0,  // 240 BPM - drum'n'bass
    6.0,  // 360 BPM - sub-beat divisions
];

// =============================================================================
// Mel Filterbank
// =============================================================================

pub struct MelFilterbank {
    filters: Vec<Vec<f32>>,
}

impl MelFilterbank {
    pub fn new() -> Self {
        let n_fft_bins = FFT_SIZE / 2 + 1;
        let f_min = 20.0;
        let f_max = SAMPLE_RATE / 2.0;

        let mel_min = 2595.0_f32 * (1.0_f32 + f_min / 700.0_f32).log10();
        let mel_max = 2595.0_f32 * (1.0_f32 + f_max / 700.0_f32).log10();

        let mel_points: Vec<f32> = (0..=MEL_BANDS + 1)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (MEL_BANDS + 1) as f32)
            .collect();

        let hz_points: Vec<f32> = mel_points
            .iter()
            .map(|m| 700.0 * (10.0_f32.powf(m / 2595.0) - 1.0))
            .collect();

        let bin_points: Vec<usize> = hz_points
            .iter()
            .map(|f| ((FFT_SIZE as f32 + 1.0) * f / SAMPLE_RATE).floor() as usize)
            .collect();

        let mut filters = vec![vec![0.0; n_fft_bins]; MEL_BANDS];

        for m in 0..MEL_BANDS {
            let start = bin_points[m];
            let center = bin_points[m + 1];
            let end = bin_points[m + 2];

            for k in start..center {
                if center > start && k < n_fft_bins {
                    filters[m][k] = (k - start) as f32 / (center - start) as f32;
                }
            }

            for k in center..end {
                if end > center && k < n_fft_bins {
                    filters[m][k] = (end - k) as f32 / (end - center) as f32;
                }
            }
        }

        Self { filters }
    }

    pub fn apply(&self, power_spectrum: &[f32]) -> Vec<f32> {
        self.filters
            .iter()
            .map(|filter| {
                filter
                    .iter()
                    .zip(power_spectrum)
                    .map(|(f, p)| f * p)
                    .sum::<f32>()
                    .max(1e-10)
                    .log10()
            })
            .collect()
    }
}

impl Default for MelFilterbank {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// MFCC Computation
// =============================================================================

/// Compute DCT-II (used for MFCC)
fn dct_ii(input: &[f32], n_output: usize) -> Vec<f32> {
    let n = input.len();
    let mut output = vec![0.0f32; n_output];

    for k in 0..n_output {
        let mut sum = 0.0f32;
        for i in 0..n {
            let angle = PI * k as f32 * (2.0 * i as f32 + 1.0) / (2.0 * n as f32);
            sum += input[i] * angle.cos();
        }
        output[k] = if k == 0 {
            sum * (1.0 / n as f32).sqrt()
        } else {
            sum * (2.0 / n as f32).sqrt()
        };
    }

    output
}

/// Compute MFCCs from mel-band energies
pub fn compute_mfcc(mel_bands: &[f32]) -> Vec<f32> {
    let mfcc = dct_ii(mel_bands, NUM_MFCC);

    let lifter_coeff = 22.0;
    mfcc.iter()
        .enumerate()
        .map(|(i, &c)| {
            if i == 0 {
                c
            } else {
                c * (1.0 + (lifter_coeff / 2.0) * (PI * i as f32 / lifter_coeff).sin())
            }
        })
        .collect()
}

/// Compute delta (first derivative) of MFCC over time
pub fn compute_mfcc_delta(mfcc_history: &[Vec<f32>], window: usize) -> Vec<f32> {
    if mfcc_history.len() < 2 * window + 1 {
        return vec![0.0; NUM_MFCC];
    }

    let n = mfcc_history.len();
    let center = n - window - 1;
    let num_coeffs = mfcc_history[0].len();

    let mut delta = vec![0.0f32; num_coeffs];
    let mut norm = 0.0f32;

    for i in 1..=window {
        norm += (i * i) as f32;
        for j in 0..num_coeffs {
            let future = mfcc_history.get(center + i).map(|m| m[j]).unwrap_or(0.0);
            let past = mfcc_history
                .get(center.saturating_sub(i))
                .map(|m| m[j])
                .unwrap_or(0.0);
            delta[j] += i as f32 * (future - past);
        }
    }

    norm *= 2.0;
    if norm > 1e-10 {
        for d in &mut delta {
            *d /= norm;
        }
    }

    delta
}

/// Compute temporal regularity from onset strength history
pub fn compute_temporal_regularity(onset_history: &[f32]) -> f32 {
    if onset_history.len() < 20 {
        return 0.5;
    }

    let n = onset_history.len();
    let mean: f32 = onset_history.iter().sum::<f32>() / n as f32;
    let variance: f32 = onset_history
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f32>()
        / n as f32;

    if variance < 1e-10 {
        return 0.0;
    }

    let max_lag = n / 2;
    let mut max_autocorr = 0.0f32;

    for lag in 2..max_lag {
        let mut autocorr = 0.0f32;
        let count = n - lag;
        for i in 0..count {
            autocorr += (onset_history[i] - mean) * (onset_history[i + lag] - mean);
        }
        autocorr /= count as f32 * variance;
        if autocorr > max_autocorr {
            max_autocorr = autocorr;
        }
    }

    max_autocorr.max(0.0).min(1.0)
}

// =============================================================================
// Audio Features
// =============================================================================

pub struct AudioFeatures {
    pub mel_bands: Vec<f32>,
    pub mfcc: Vec<f32>,
    pub spectral_centroid: f32,
    pub spectral_rolloff: f32,
    pub onset_strength: f32,
    pub rms_energy: f32,
    pub zero_crossing_rate: f32,
    pub spectral_flatness: f32,
    pub temporal_regularity: f32,
    pub envelope_delta: f32,
    pub envelope_variance: f32,
    pub frame_index: usize,
    pub spectral_flux: f32,
    pub harmonic_ratio: f32,
    pub cfc_theta_gamma: f32,
    pub cfc_delta_beta: f32,
    pub attack_sharpness: f32,
    pub decay_roughness: f32,
    pub silence_ratio: f32,
    pub burst_density: f32,
}

impl AudioFeatures {
    /// Create synthetic features for testing
    #[allow(clippy::too_many_arguments)]
    pub fn synthetic(
        mel_bands: Vec<f32>,
        mfcc: Vec<f32>,
        spectral_centroid: f32,
        spectral_rolloff: f32,
        onset_strength: f32,
        rms_energy: f32,
        zero_crossing_rate: f32,
        spectral_flatness: f32,
        temporal_regularity: f32,
        envelope_delta: f32,
        envelope_variance: f32,
        frame_index: usize,
    ) -> Self {
        Self {
            mel_bands,
            mfcc,
            spectral_centroid,
            spectral_rolloff,
            onset_strength,
            rms_energy,
            zero_crossing_rate,
            spectral_flatness,
            temporal_regularity,
            envelope_delta,
            envelope_variance,
            frame_index,
            spectral_flux: 0.1,
            harmonic_ratio: 0.5,
            cfc_theta_gamma: 0.5,
            cfc_delta_beta: 0.5,
            attack_sharpness: 0.5,
            decay_roughness: 0.5,
            silence_ratio: 0.5,
            burst_density: 0.5,
        }
    }
}

// =============================================================================
// Feature Extractor
// =============================================================================

pub struct FeatureExtractor {
    mel_filterbank: MelFilterbank,
    prev_spectrum: Vec<f32>,
    onset_history: Vec<f32>,
    rms_history: Vec<f32>,
    prev_rms: f32,
    envelope_delta_history: Vec<f32>,
    frame_counter: usize,
    prev_mel_bands: Vec<f32>,
    low_band_history: Vec<f32>,
    high_band_history: Vec<f32>,
    silent_frame_count: usize,
    total_frame_count: usize,
}

impl FeatureExtractor {
    pub fn new() -> Self {
        Self {
            mel_filterbank: MelFilterbank::new(),
            prev_spectrum: vec![0.0; FFT_SIZE / 2 + 1],
            onset_history: Vec::new(),
            rms_history: Vec::new(),
            prev_rms: 0.0,
            envelope_delta_history: Vec::new(),
            frame_counter: 0,
            prev_mel_bands: vec![0.0; MEL_BANDS],
            low_band_history: Vec::new(),
            high_band_history: Vec::new(),
            silent_frame_count: 0,
            total_frame_count: 0,
        }
    }

    /// Compute cross-frequency coupling (phase-amplitude coupling)
    fn compute_cfc(&self, low_band: &[f32], high_band: &[f32]) -> f32 {
        if low_band.len() < 16 || high_band.len() < 16 {
            return 0.5;
        }

        let low_phase: Vec<f32> = low_band
            .windows(2)
            .map(|w| (w[1] - w[0]).atan2(1.0))
            .collect();

        let high_amp: Vec<f32> = high_band.iter().map(|&x| x.abs()).collect();

        let n = low_phase.len().min(high_amp.len() - 1);
        if n < 8 {
            return 0.5;
        }

        let mut bin_sums = [0.0f32; 4];
        let mut bin_counts = [0usize; 4];

        for i in 0..n {
            let phase_norm = (low_phase[i] / PI + 1.0) / 2.0;
            let bin = ((phase_norm * 4.0) as usize).min(3);
            bin_sums[bin] += high_amp[i];
            bin_counts[bin] += 1;
        }

        let bin_means: Vec<f32> = bin_sums
            .iter()
            .zip(&bin_counts)
            .map(|(&s, &c)| if c > 0 { s / c as f32 } else { 0.0 })
            .collect();

        let overall_mean: f32 = bin_means.iter().sum::<f32>() / 4.0;
        let variance: f32 = bin_means
            .iter()
            .map(|&m| (m - overall_mean).powi(2))
            .sum::<f32>()
            / 4.0;

        (variance.sqrt() / (overall_mean + 1e-10)).clamp(0.0, 1.0)
    }

    /// Compute harmonic-to-noise ratio (HNR)
    fn compute_hnr(&self, samples: &[f32]) -> f32 {
        let n = samples.len();
        if n < 64 {
            return 0.5;
        }

        let r0: f32 = samples.iter().map(|&x| x * x).sum();
        if r0 < 1e-10 {
            return 0.0;
        }

        let min_lag = (SAMPLE_RATE / 500.0) as usize;
        let max_lag = (SAMPLE_RATE / 50.0).min(n as f32 / 2.0) as usize;

        let mut max_r = 0.0f32;
        for lag in min_lag..max_lag.min(n / 2) {
            let r: f32 = (0..(n - lag)).map(|i| samples[i] * samples[i + lag]).sum();
            if r > max_r {
                max_r = r;
            }
        }

        let hnr_ratio = max_r / (r0 - max_r + 1e-10);
        (hnr_ratio / (1.0 + hnr_ratio)).clamp(0.0, 1.0)
    }

    /// Extract features from a power spectrum
    pub fn extract(&mut self, power_spectrum: &[f32], samples: &[f32]) -> AudioFeatures {
        let mel_bands = self.mel_filterbank.apply(power_spectrum);
        let mfcc = compute_mfcc(&mel_bands);

        // Spectral centroid
        let total_power: f32 = power_spectrum.iter().sum();
        let spectral_centroid = if total_power > 1e-10 {
            power_spectrum
                .iter()
                .enumerate()
                .map(|(i, p)| i as f32 * p)
                .sum::<f32>()
                / total_power
                * (SAMPLE_RATE / FFT_SIZE as f32)
        } else {
            0.0
        };

        // Spectral rolloff (85% of energy)
        let mut cumsum = 0.0;
        let threshold = total_power * 0.85;
        let mut rolloff_bin = 0;
        for (i, p) in power_spectrum.iter().enumerate() {
            cumsum += p;
            if cumsum >= threshold {
                rolloff_bin = i;
                break;
            }
        }
        let spectral_rolloff = rolloff_bin as f32 * (SAMPLE_RATE / FFT_SIZE as f32);

        // Onset strength
        let onset_strength: f32 = power_spectrum
            .iter()
            .zip(&self.prev_spectrum)
            .map(|(curr, prev)| (curr - prev).max(0.0))
            .sum::<f32>()
            / power_spectrum.len() as f32;
        self.prev_spectrum = power_spectrum.to_vec();

        // RMS energy
        let rms_energy = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();

        // Zero crossing rate
        let zero_crossings: usize = samples
            .windows(2)
            .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
            .count();
        let zero_crossing_rate = zero_crossings as f32 / samples.len() as f32;

        // Spectral flatness
        let n = power_spectrum.len() as f32;
        let epsilon = 1e-10;
        let log_sum: f32 = power_spectrum.iter().map(|&x| (x + epsilon).ln()).sum();
        let geometric_mean = (log_sum / n).exp();
        let arithmetic_mean = power_spectrum.iter().sum::<f32>() / n;
        let spectral_flatness = (geometric_mean / (arithmetic_mean + epsilon))
            .min(1.0)
            .max(0.0);

        // Temporal regularity
        self.onset_history.push(onset_strength);
        if self.onset_history.len() > 100 {
            self.onset_history.remove(0);
        }
        let temporal_regularity = compute_temporal_regularity(&self.onset_history);

        // Burst density: rate of significant onset events
        let burst_density = if self.onset_history.len() < 10 {
            0.5 // Default for insufficient data
        } else {
            let mean: f32 =
                self.onset_history.iter().sum::<f32>() / self.onset_history.len() as f32;
            let variance: f32 = self
                .onset_history
                .iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f32>()
                / self.onset_history.len() as f32;
            let std_dev = variance.sqrt();
            let threshold = mean + 0.5 * std_dev;

            // Count bursts (transitions from below to above threshold)
            let mut burst_count = 0;
            let mut in_burst = false;
            for &onset in &self.onset_history {
                if onset > threshold && !in_burst {
                    burst_count += 1;
                    in_burst = true;
                } else if onset <= threshold * 0.8 {
                    in_burst = false;
                }
            }

            // Convert to bursts per second (CONTROL_RATE frames/sec)
            let window_duration = self.onset_history.len() as f32 / CONTROL_RATE;
            let bursts_per_second = burst_count as f32 / window_duration;
            (bursts_per_second / 8.0).min(1.0)
        };

        // Envelope features
        self.rms_history.push(rms_energy);
        if self.rms_history.len() > 32 {
            self.rms_history.remove(0);
        }

        let envelope_delta = (rms_energy - self.prev_rms).abs();

        let envelope_variance = if self.rms_history.len() > 2 {
            let mean: f32 = self.rms_history.iter().sum::<f32>() / self.rms_history.len() as f32;
            self.rms_history
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>()
                / self.rms_history.len() as f32
        } else {
            0.0
        };

        // Transient analyzer
        let signed_delta = rms_energy - self.prev_rms;
        self.envelope_delta_history.push(signed_delta);
        if self.envelope_delta_history.len() > 64 {
            self.envelope_delta_history.remove(0);
        }

        let attack_sharpness = if self.envelope_delta_history.len() > 4 {
            let max_positive: f32 = self
                .envelope_delta_history
                .iter()
                .filter(|&&d| d > 0.0)
                .copied()
                .fold(0.0f32, |max, d| max.max(d));
            (max_positive * 2.0).clamp(0.0, 1.0)
        } else {
            0.5
        };

        let decay_roughness = if self.envelope_delta_history.len() > 8 {
            let negative_deltas: Vec<f32> = self
                .envelope_delta_history
                .iter()
                .filter(|&&d| d < 0.0)
                .copied()
                .collect();
            if negative_deltas.len() > 2 {
                let mean: f32 = negative_deltas.iter().sum::<f32>() / negative_deltas.len() as f32;
                let variance: f32 = negative_deltas
                    .iter()
                    .map(|&d| (d - mean).powi(2))
                    .sum::<f32>()
                    / negative_deltas.len() as f32;
                (variance * 100.0).sqrt().clamp(0.0, 1.0)
            } else {
                0.5
            }
        } else {
            0.5
        };

        self.prev_rms = rms_energy;

        let frame_index = self.frame_counter;
        self.frame_counter += 1;

        // Spectral flux
        let spectral_flux: f32 = mel_bands
            .iter()
            .zip(&self.prev_mel_bands)
            .map(|(curr, prev)| (curr - prev).powi(2))
            .sum::<f32>()
            .sqrt()
            / mel_bands.len() as f32;
        self.prev_mel_bands = mel_bands.clone();

        // Harmonic ratio
        let harmonic_ratio = self.compute_hnr(samples);

        // Cross-frequency coupling
        let low_band_energy: f32 = mel_bands.iter().take(10).sum::<f32>() / 10.0;
        let high_band_energy: f32 = mel_bands.iter().skip(20).sum::<f32>() / 20.0;

        self.low_band_history.push(low_band_energy);
        self.high_band_history.push(high_band_energy);
        if self.low_band_history.len() > 64 {
            self.low_band_history.remove(0);
            self.high_band_history.remove(0);
        }

        let cfc_theta_gamma = self.compute_cfc(&self.low_band_history, &self.high_band_history);
        let cfc_delta_beta = if self.low_band_history.len() > 32 {
            self.compute_cfc(
                &self.low_band_history[..self.low_band_history.len() / 2],
                &self.high_band_history[self.high_band_history.len() / 2..],
            )
        } else {
            0.5
        };

        // Silence ratio
        const SILENCE_THRESHOLD: f32 = 0.02;
        self.total_frame_count += 1;
        if rms_energy < SILENCE_THRESHOLD {
            self.silent_frame_count += 1;
        }
        let silence_ratio = self.silent_frame_count as f32 / self.total_frame_count as f32;

        AudioFeatures {
            mel_bands,
            mfcc,
            spectral_centroid,
            spectral_rolloff,
            onset_strength,
            rms_energy,
            zero_crossing_rate,
            spectral_flatness,
            temporal_regularity,
            envelope_delta,
            envelope_variance,
            frame_index,
            spectral_flux,
            harmonic_ratio,
            cfc_theta_gamma,
            cfc_delta_beta,
            attack_sharpness,
            decay_roughness,
            silence_ratio,
            burst_density,
        }
    }
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute power spectrum using simple DFT
pub fn compute_power_spectrum(samples: &[f32]) -> Vec<f32> {
    let n = samples.len().min(FFT_SIZE);
    let mut power = vec![0.0; n / 2 + 1];

    let windowed: Vec<f32> = samples
        .iter()
        .enumerate()
        .take(n)
        .map(|(i, &s)| {
            let w = 0.5 * (1.0 - (2.0 * PI * i as f32 / (n - 1) as f32).cos());
            s * w
        })
        .collect();

    for k in 0..power.len() {
        let mut real = 0.0f32;
        let mut imag = 0.0f32;

        for (i, &x) in windowed.iter().enumerate() {
            let angle = 2.0 * PI * k as f32 * i as f32 / n as f32;
            real += x * angle.cos();
            imag -= x * angle.sin();
        }

        power[k] = (real * real + imag * imag) / n as f32;
    }

    power
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_filterbank() {
        let fb = MelFilterbank::new();
        assert_eq!(fb.filters.len(), MEL_BANDS);

        let flat = vec![1.0; FFT_SIZE / 2 + 1];
        let mels = fb.apply(&flat);
        assert_eq!(mels.len(), MEL_BANDS);
    }
}
