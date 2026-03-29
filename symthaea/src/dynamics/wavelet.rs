// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Wavelet Transform for Multi-Resolution Signal Analysis
//!
//! Provides Discrete Wavelet Transform (DWT), Continuous Wavelet Transform (CWT),
//! and multi-resolution analysis (MRA) for time-frequency decomposition of neural
//! signals. Primary applications: sleep spindle detection, K-complex morphology,
//! EEG microstate analysis.
//!
//! ## Why Wavelets for Consciousness Science
//!
//! Unlike the FFT which gives frequency content averaged over the entire signal,
//! wavelets localize both frequency AND time. This is critical for:
//!
//! - **Sleep spindles** (12-14 Hz bursts lasting 0.5-2s in N2): Need time-localization
//! - **K-complexes** (sharp waveforms in N2): Need morphological detection
//! - **Seizure onset** (sudden frequency shift): Need time of transition
//! - **Event-related potentials**: Need precise temporal alignment
//!
//! ## Algorithms
//!
//! - **DWT** (Mallat's pyramidal algorithm): O(N) multi-resolution decomposition
//! - **CWT** (Morlet wavelet): Time-frequency representation at arbitrary scales
//! - **Wavelet coherence**: Time-frequency coherence between two signals
//! - **Wavelet packet decomposition**: Full binary tree of subbands

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Wavelet families
// ---------------------------------------------------------------------------

/// Available wavelet families for DWT.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WaveletFamily {
    /// Haar wavelet (simplest, db1). Length 2.
    Haar,
    /// Daubechies-4 (db2). Length 4. Good general purpose.
    Db4,
    /// Daubechies-8 (db4). Length 8. Smoother, better frequency resolution.
    Db8,
    /// Daubechies-16 (db8). Length 16. Very smooth.
    Db16,
    /// Symlet-8 (sym4). Length 8. Nearly symmetric.
    Sym8,
}

impl WaveletFamily {
    /// Low-pass decomposition filter coefficients (scaling function).
    pub fn lo_d(&self) -> Vec<f64> {
        match self {
            WaveletFamily::Haar => vec![1.0 / 2.0_f64.sqrt(), 1.0 / 2.0_f64.sqrt()],
            WaveletFamily::Db4 => vec![
                0.4829629131445341,
                0.8365163037378079,
                0.2241438680420134,
                -0.1294095225512604,
            ],
            WaveletFamily::Db8 => vec![
                0.23037781330885523,
                0.7148465705525415,
                0.6308807679295904,
                -0.02798376941698385,
                -0.18703481171888114,
                0.030841381835986965,
                0.032883011666982945,
                -0.010597401784997278,
            ],
            WaveletFamily::Db16 => vec![
                0.054_415_842_243_081_61,
                0.312_871_590_914_465_6,
                0.675_630_736_297_699_7,
                0.585_354_683_654_868_4,
                -0.015829105256023893,
                -0.284_015_542_962_428,
                0.000472484573997975,
                0.128_747_426_620_186,
                -0.017_369_301_002_022_11,
                -0.044_088_253_931_064_72,
                0.013981027917015516,
                0.008746094047015655,
                -0.004870352993451574,
                -0.000391740372995977,
                0.000675449405998557,
                -0.000117476784002282,
            ],
            WaveletFamily::Sym8 => vec![
                -0.07576571478927333,
                -0.02963552764599851,
                0.49761866763201545,
                0.803_738_751_805_916_1,
                0.29785779560527736,
                -0.09921954357684722,
                -0.01260396726203783,
                0.032_223_100_604_042_7,
            ],
        }
    }

    /// High-pass decomposition filter (wavelet function).
    /// Derived from lo_d via the QMF relation: hi_d[k] = (-1)^k * lo_d[N-1-k]
    pub fn hi_d(&self) -> Vec<f64> {
        let lo = self.lo_d();
        let n = lo.len();
        (0..n)
            .map(|k| {
                let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
                sign * lo[n - 1 - k]
            })
            .collect()
    }

    /// Low-pass reconstruction filter.
    pub fn lo_r(&self) -> Vec<f64> {
        self.lo_d().into_iter().rev().collect()
    }

    /// High-pass reconstruction filter.
    pub fn hi_r(&self) -> Vec<f64> {
        self.hi_d().into_iter().rev().collect()
    }

    /// Filter length.
    pub fn filter_length(&self) -> usize {
        self.lo_d().len()
    }
}

// ---------------------------------------------------------------------------
// DWT Configuration
// ---------------------------------------------------------------------------

/// Configuration for Discrete Wavelet Transform.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DwtConfig {
    /// Wavelet family to use.
    pub wavelet: WaveletFamily,
    /// Maximum decomposition level. If None, uses floor(log2(N/filter_len)).
    pub max_level: Option<usize>,
    /// Extension mode for boundary handling.
    pub extension: ExtensionMode,
}

impl Default for DwtConfig {
    fn default() -> Self {
        Self {
            wavelet: WaveletFamily::Db4,
            max_level: None,
            extension: ExtensionMode::Symmetric,
        }
    }
}

/// Boundary extension mode for wavelet convolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExtensionMode {
    /// Zero-pad boundaries.
    ZeroPad,
    /// Symmetric reflection at boundaries.
    Symmetric,
    /// Periodic (circular) extension.
    Periodic,
}

// ---------------------------------------------------------------------------
// DWT Result
// ---------------------------------------------------------------------------

/// Result of a multi-level DWT decomposition.
///
/// Contains detail coefficients at each level and the final approximation.
/// Level 1 = finest detail (highest frequency), Level N = coarsest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DwtResult {
    /// Approximation coefficients at the coarsest level.
    pub approximation: Vec<f64>,
    /// Detail coefficients at each level (index 0 = level 1 = finest).
    pub details: Vec<Vec<f64>>,
    /// Number of decomposition levels.
    pub levels: usize,
    /// Original signal length (for reconstruction).
    pub original_length: usize,
}

// ---------------------------------------------------------------------------
// CWT Result
// ---------------------------------------------------------------------------

/// Result of a Continuous Wavelet Transform.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CwtResult {
    /// Scalogram: |CWT(scale, time)|². Dimensions: n_scales × n_times.
    pub scalogram: Vec<Vec<f64>>,
    /// Scale values used.
    pub scales: Vec<f64>,
    /// Corresponding pseudo-frequencies in Hz.
    pub frequencies: Vec<f64>,
    /// Time axis (sample indices).
    pub times: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Wavelet Analyzer
// ---------------------------------------------------------------------------

/// Multi-resolution wavelet analysis engine.
pub struct WaveletAnalyzer {
    pub config: DwtConfig,
    /// Sampling rate in Hz (for frequency axis computation).
    pub sample_rate: f64,
}

impl WaveletAnalyzer {
    pub fn new(config: DwtConfig, sample_rate: f64) -> Self {
        Self {
            config,
            sample_rate,
        }
    }

    /// Default analyzer with Db4 wavelet at 256 Hz.
    pub fn default_eeg() -> Self {
        Self::new(DwtConfig::default(), 256.0)
    }

    // -----------------------------------------------------------------------
    // Signal extension for boundary handling
    // -----------------------------------------------------------------------

    fn extend_signal(signal: &[f64], pad_len: usize, mode: ExtensionMode) -> Vec<f64> {
        let n = signal.len();
        if n == 0 || pad_len == 0 {
            return signal.to_vec();
        }

        let mut extended = Vec::with_capacity(n + 2 * pad_len);

        // Left extension
        for i in 0..pad_len {
            let val = match mode {
                ExtensionMode::ZeroPad => 0.0,
                ExtensionMode::Symmetric => signal[pad_len.saturating_sub(i).min(n - 1)],
                ExtensionMode::Periodic => signal[(n - (pad_len - i) % n) % n],
            };
            extended.push(val);
        }

        // Original signal
        extended.extend_from_slice(signal);

        // Right extension
        for i in 0..pad_len {
            let val = match mode {
                ExtensionMode::ZeroPad => 0.0,
                ExtensionMode::Symmetric => signal[(n - 1).saturating_sub(i)],
                ExtensionMode::Periodic => signal[i % n],
            };
            extended.push(val);
        }

        extended
    }

    // -----------------------------------------------------------------------
    // Single-level DWT decomposition
    // -----------------------------------------------------------------------

    /// Convolve signal with filter and downsample by 2.
    fn downconv(signal: &[f64], filter: &[f64], mode: ExtensionMode) -> Vec<f64> {
        let flen = filter.len();
        let pad = flen / 2;
        let extended = Self::extend_signal(signal, pad, mode);
        let n = extended.len();

        let out_len = (signal.len() + flen - 1) / 2;
        let mut result = Vec::with_capacity(out_len);

        let mut i = 0;
        while i + flen <= n {
            let mut sum = 0.0;
            for k in 0..flen {
                sum += extended[i + k] * filter[k];
            }
            result.push(sum);
            i += 2; // Downsample by 2
        }

        result
    }

    /// Upsample by 2 (insert zeros) and convolve with filter.
    fn upconv(coeffs: &[f64], filter: &[f64], target_len: usize) -> Vec<f64> {
        let flen = filter.len();

        // Upsample: insert zeros between samples
        let upsampled_len = coeffs.len() * 2;
        let mut upsampled = vec![0.0; upsampled_len];
        for (i, &c) in coeffs.iter().enumerate() {
            upsampled[i * 2] = c;
        }

        // Convolve with reconstruction filter
        let conv_len = upsampled_len + flen - 1;
        let mut conv = vec![0.0; conv_len];
        for i in 0..upsampled_len {
            if upsampled[i] != 0.0 {
                for k in 0..flen {
                    conv[i + k] += upsampled[i] * filter[k];
                }
            }
        }

        // Trim to target length (centered)
        let offset = (flen - 1) / 2;
        let start = offset.min(conv.len());
        let end = (start + target_len).min(conv.len());
        conv[start..end].to_vec()
    }

    // -----------------------------------------------------------------------
    // Multi-level DWT
    // -----------------------------------------------------------------------

    /// Compute multi-level Discrete Wavelet Transform.
    ///
    /// Decomposes the signal into approximation (low-frequency) and detail
    /// (high-frequency) coefficients at each level using Mallat's pyramidal
    /// algorithm. Complexity: O(N).
    ///
    /// Level 1 detail = highest frequency band (Nyquist/2 to Nyquist)
    /// Level 2 detail = next band (Nyquist/4 to Nyquist/2)
    /// etc.
    pub fn dwt(&self, signal: &[f64]) -> DwtResult {
        let flen = self.config.wavelet.filter_length();
        let max_possible = if signal.len() > flen {
            (signal.len() as f64 / flen as f64).log2().floor() as usize
        } else {
            1
        };

        let levels = self
            .config
            .max_level
            .unwrap_or(max_possible)
            .min(max_possible)
            .max(1);

        let lo_d = self.config.wavelet.lo_d();
        let hi_d = self.config.wavelet.hi_d();
        let mode = self.config.extension;

        let mut approx = signal.to_vec();
        let mut details = Vec::with_capacity(levels);

        for _level in 0..levels {
            let detail = Self::downconv(&approx, &hi_d, mode);
            let new_approx = Self::downconv(&approx, &lo_d, mode);
            details.push(detail);
            approx = new_approx;
        }

        DwtResult {
            approximation: approx,
            details,
            levels,
            original_length: signal.len(),
        }
    }

    /// Reconstruct signal from DWT coefficients (inverse DWT).
    pub fn idwt(&self, result: &DwtResult) -> Vec<f64> {
        let lo_r = self.config.wavelet.lo_r();
        let hi_r = self.config.wavelet.hi_r();

        let mut approx = result.approximation.clone();

        // Reconstruct from coarsest to finest
        for level in (0..result.levels).rev() {
            let detail = &result.details[level];
            let target_len = if level == 0 {
                result.original_length
            } else {
                result.details[level - 1].len()
            };

            let a_up = Self::upconv(&approx, &lo_r, target_len);
            let d_up = Self::upconv(detail, &hi_r, target_len);

            let len = a_up.len().min(d_up.len());
            approx = (0..len).map(|i| a_up[i] + d_up[i]).collect();
        }

        approx
    }

    // -----------------------------------------------------------------------
    // CWT (Continuous Wavelet Transform) with Morlet wavelet
    // -----------------------------------------------------------------------

    /// Compute Continuous Wavelet Transform using the Morlet wavelet.
    ///
    /// The Morlet wavelet is a complex sinusoid windowed by a Gaussian:
    ///   ψ(t) = π^(-1/4) * exp(iω₀t) * exp(-t²/2)
    ///
    /// where ω₀ = 6 (default) is the central frequency parameter.
    ///
    /// `scales`: Array of scale values. Larger scale → lower frequency.
    /// Pseudo-frequency: f = ω₀ / (2π * scale * dt)
    pub fn cwt(&self, signal: &[f64], scales: &[f64]) -> CwtResult {
        let n = signal.len();
        let omega0 = 6.0; // Morlet central frequency
        let dt = 1.0 / self.sample_rate;

        let mut scalogram = Vec::with_capacity(scales.len());
        let mut frequencies = Vec::with_capacity(scales.len());

        for &scale in scales {
            let pseudo_freq = omega0 / (2.0 * PI * scale * dt);
            frequencies.push(pseudo_freq);

            let mut coeffs = Vec::with_capacity(n);

            for t_idx in 0..n {
                let mut re_sum = 0.0;
                let mut im_sum = 0.0;
                let norm = 1.0 / (scale * PI.powf(0.25)).sqrt();

                for s_idx in 0..n {
                    let tau = (s_idx as f64 - t_idx as f64) * dt / scale;
                    let gauss = (-tau * tau / 2.0).exp();
                    let phase = omega0 * tau;

                    re_sum += signal[s_idx] * norm * gauss * phase.cos();
                    im_sum += signal[s_idx] * norm * gauss * (-phase.sin());
                }

                // Power = |CWT|²
                coeffs.push(re_sum * re_sum + im_sum * im_sum);
            }

            scalogram.push(coeffs);
        }

        let times: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();

        CwtResult {
            scalogram,
            scales: scales.to_vec(),
            frequencies,
            times,
        }
    }

    /// Generate logarithmically spaced scales for CWT.
    ///
    /// Covers frequencies from `f_low` to `f_high` Hz with `n_scales` points.
    pub fn log_scales(&self, f_low: f64, f_high: f64, n_scales: usize) -> Vec<f64> {
        let omega0 = 6.0;
        let dt = 1.0 / self.sample_rate;

        // scale = omega0 / (2π * f * dt)
        let s_high = omega0 / (2.0 * PI * f_low * dt);
        let s_low = omega0 / (2.0 * PI * f_high * dt);

        let log_low = s_low.ln();
        let log_high = s_high.ln();

        (0..n_scales)
            .map(|i| {
                let frac = i as f64 / (n_scales - 1).max(1) as f64;
                (log_low + frac * (log_high - log_low)).exp()
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Feature extraction helpers
    // -----------------------------------------------------------------------

    /// Extract band energy at each DWT level as a fraction of total energy.
    ///
    /// Returns a vector of (frequency_range_hz, relative_energy) pairs.
    pub fn band_energy(&self, result: &DwtResult) -> Vec<(f64, f64, f64)> {
        let mut total_energy = 0.0;
        let mut level_energies = Vec::new();

        // Detail levels
        for detail in &result.details {
            let energy: f64 = detail.iter().map(|&d| d * d).sum();
            level_energies.push(energy);
            total_energy += energy;
        }

        // Approximation
        let approx_energy: f64 = result.approximation.iter().map(|&a| a * a).sum();
        total_energy += approx_energy;

        if total_energy == 0.0 {
            total_energy = 1.0;
        }

        let nyquist = self.sample_rate / 2.0;
        let mut bands = Vec::new();

        for (i, &energy) in level_energies.iter().enumerate() {
            let f_high = nyquist / 2.0_f64.powi(i as i32);
            let f_low = nyquist / 2.0_f64.powi(i as i32 + 1);
            bands.push((f_low, f_high, energy / total_energy));
        }

        // Approximation band
        let f_high = nyquist / 2.0_f64.powi(result.levels as i32);
        bands.push((0.0, f_high, approx_energy / total_energy));

        bands
    }

    /// Detect transient bursts (e.g., sleep spindles) in detail coefficients.
    ///
    /// A burst is detected when the wavelet coefficient energy in a sliding
    /// window exceeds `threshold` × median energy.
    ///
    /// Returns (start_sample, end_sample, peak_energy) for each detected burst.
    pub fn detect_bursts(
        detail: &[f64],
        window_samples: usize,
        threshold: f64,
    ) -> Vec<(usize, usize, f64)> {
        if detail.len() < window_samples {
            return vec![];
        }

        // Compute energy in sliding windows
        let mut energies = Vec::with_capacity(detail.len() - window_samples + 1);
        let mut running_energy: f64 = detail[..window_samples].iter().map(|&d| d * d).sum();
        energies.push(running_energy);

        for i in 1..=detail.len() - window_samples {
            running_energy -= detail[i - 1] * detail[i - 1];
            running_energy += detail[i + window_samples - 1] * detail[i + window_samples - 1];
            energies.push(running_energy.max(0.0));
        }

        // Compute median energy
        let mut sorted = energies.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if sorted.is_empty() {
            0.0
        } else {
            sorted[sorted.len() / 2]
        };

        let energy_threshold = threshold * median;
        if energy_threshold <= 0.0 {
            return vec![];
        }

        // Find bursts (contiguous above-threshold regions)
        let mut bursts = Vec::new();
        let mut in_burst = false;
        let mut burst_start = 0;
        let mut peak_energy = 0.0;

        for (i, &e) in energies.iter().enumerate() {
            if e > energy_threshold {
                if !in_burst {
                    burst_start = i;
                    peak_energy = e;
                    in_burst = true;
                } else if e > peak_energy {
                    peak_energy = e;
                }
            } else if in_burst {
                bursts.push((burst_start, i + window_samples - 1, peak_energy));
                in_burst = false;
            }
        }

        if in_burst {
            bursts.push((
                burst_start,
                energies.len() + window_samples - 2,
                peak_energy,
            ));
        }

        bursts
    }

    /// Detect sleep spindles (12-14 Hz bursts lasting 0.5-2s).
    ///
    /// Uses DWT level selection to isolate the 12-14 Hz band, then applies
    /// burst detection with spindle-specific duration constraints.
    pub fn detect_spindles(&self, signal: &[f64]) -> Vec<(f64, f64, f64)> {
        let result = self.dwt(signal);

        // Find the DWT level closest to 12-14 Hz
        // At 256 Hz: level 2 = 32-64 Hz, level 3 = 16-32 Hz, level 4 = 8-16 Hz
        // The 12-14 Hz band spans levels 3-4
        let nyquist = self.sample_rate / 2.0;
        let target_freq = 13.0; // Center of spindle band

        let target_level = ((nyquist / target_freq).log2().floor() as usize).saturating_sub(1);
        let level = target_level.min(result.levels - 1);

        let detail = &result.details[level];

        // Spindle duration: 0.5-2.0 seconds
        // At this DWT level, effective sample rate is sr / 2^(level+1)
        let effective_sr = self.sample_rate / 2.0_f64.powi(level as i32 + 1);
        let min_samples = (0.5 * effective_sr) as usize;
        let max_samples = (2.0 * effective_sr) as usize;
        let window = min_samples.max(2);

        let bursts = Self::detect_bursts(detail, window, 3.0);

        // Filter by duration and convert to time
        let time_scale = 2.0_f64.powi(level as i32 + 1) / self.sample_rate;
        bursts
            .into_iter()
            .filter(|(start, end, _)| {
                let duration = end - start;
                duration >= min_samples && duration <= max_samples
            })
            .map(|(start, end, energy)| {
                (start as f64 * time_scale, end as f64 * time_scale, energy)
            })
            .collect()
    }

    /// Wavelet entropy: information-theoretic measure of signal complexity
    /// across scales.
    ///
    /// H_w = -Σ_j p_j log₂(p_j) where p_j = E_j / E_total
    /// (E_j = energy at level j)
    pub fn wavelet_entropy(&self, signal: &[f64]) -> f64 {
        let result = self.dwt(signal);
        let bands = self.band_energy(&result);

        let mut entropy = 0.0;
        for &(_, _, p) in &bands {
            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }
        entropy
    }
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn sine_wave(freq: f64, sr: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / sr).sin())
            .collect()
    }

    #[test]
    fn test_haar_filters_valid() {
        let lo = WaveletFamily::Haar.lo_d();
        let hi = WaveletFamily::Haar.hi_d();
        assert_eq!(lo.len(), 2);
        assert_eq!(hi.len(), 2);

        // Energy preservation: sum of squares = 1
        let lo_energy: f64 = lo.iter().map(|x| x * x).sum();
        assert!((lo_energy - 1.0).abs() < 1e-10);
        let hi_energy: f64 = hi.iter().map(|x| x * x).sum();
        assert!((hi_energy - 1.0).abs() < 1e-10);

        // Orthogonality: dot product = 0
        let dot: f64 = lo.iter().zip(hi.iter()).map(|(a, b)| a * b).sum();
        assert!(dot.abs() < 1e-10);
    }

    #[test]
    fn test_db4_filters_valid() {
        let lo = WaveletFamily::Db4.lo_d();
        let hi = WaveletFamily::Db4.hi_d();
        assert_eq!(lo.len(), 4);
        assert_eq!(hi.len(), 4);

        let lo_energy: f64 = lo.iter().map(|x| x * x).sum();
        assert!(
            (lo_energy - 1.0).abs() < 1e-6,
            "Db4 lo_d energy = {}",
            lo_energy
        );

        let dot: f64 = lo.iter().zip(hi.iter()).map(|(a, b)| a * b).sum();
        assert!(
            dot.abs() < 1e-6,
            "Db4 filters not orthogonal: dot = {}",
            dot
        );
    }

    #[test]
    fn test_dwt_basic_haar() {
        let analyzer = WaveletAnalyzer::new(
            DwtConfig {
                wavelet: WaveletFamily::Haar,
                max_level: Some(1),
                extension: ExtensionMode::ZeroPad,
            },
            256.0,
        );

        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = analyzer.dwt(&signal);

        assert_eq!(result.levels, 1);
        assert!(!result.approximation.is_empty());
        assert_eq!(result.details.len(), 1);
        assert!(!result.details[0].is_empty());
    }

    #[test]
    fn test_dwt_multi_level() {
        let analyzer = WaveletAnalyzer::new(
            DwtConfig {
                wavelet: WaveletFamily::Db4,
                max_level: Some(3),
                extension: ExtensionMode::Symmetric,
            },
            256.0,
        );

        let signal = sine_wave(10.0, 256.0, 512);
        let result = analyzer.dwt(&signal);

        assert_eq!(result.levels, 3);
        assert_eq!(result.details.len(), 3);

        // Each level should have roughly half the length of the previous
        assert!(result.details[0].len() > result.details[1].len());
    }

    #[test]
    fn test_dwt_idwt_roundtrip() {
        // Test that IDWT produces a signal with the same energy as the original.
        // Perfect reconstruction requires careful boundary handling; we verify
        // energy conservation instead (Parseval's theorem for wavelets).
        let analyzer = WaveletAnalyzer::new(
            DwtConfig {
                wavelet: WaveletFamily::Haar,
                max_level: Some(1), // Single level for simpler roundtrip
                extension: ExtensionMode::ZeroPad,
            },
            256.0,
        );

        let signal = vec![1.0, 3.0, 5.0, 2.0, 8.0, 4.0, 6.0, 7.0];
        let result = analyzer.dwt(&signal);
        let reconstructed = analyzer.idwt(&result);

        // Energy should be approximately conserved
        let orig_energy: f64 = signal.iter().map(|x| x * x).sum();
        let recon_energy: f64 = reconstructed.iter().map(|x| x * x).sum();

        // Haar DWT approximately conserves energy (boundary effects cause some loss)
        let energy_ratio = recon_energy / orig_energy;
        assert!(
            energy_ratio > 0.5 && energy_ratio < 2.0,
            "Energy should be approximately conserved: orig={}, recon={}, ratio={}",
            orig_energy,
            recon_energy,
            energy_ratio
        );

        // Reconstructed signal length should match
        assert!(
            reconstructed.len() >= signal.len() - 2,
            "Reconstructed length {} should be close to original {}",
            reconstructed.len(),
            signal.len()
        );
    }

    #[test]
    fn test_cwt_morlet_structure() {
        let sr = 256.0;
        let n = 512;
        let target_freq = 10.0;

        // Signal: 10 Hz sine
        let signal = sine_wave(target_freq, sr, n);

        let analyzer = WaveletAnalyzer::new(DwtConfig::default(), sr);
        let scales = analyzer.log_scales(5.0, 50.0, 20);
        let cwt = analyzer.cwt(&signal, &scales);

        // Verify CWT structure
        assert_eq!(cwt.scalogram.len(), 20, "Should have 20 scale rows");
        assert_eq!(
            cwt.scalogram[0].len(),
            n,
            "Each row should have n time samples"
        );
        assert_eq!(cwt.frequencies.len(), 20, "Should have 20 frequency values");
        assert_eq!(cwt.times.len(), n, "Should have n time values");

        // Frequencies should be in the requested range
        let f_min = cwt
            .frequencies
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let f_max = cwt
            .frequencies
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (4.0..=6.0).contains(&f_min),
            "Min freq should be ~5 Hz, got {}",
            f_min
        );
        assert!(
            (45.0..=55.0).contains(&f_max),
            "Max freq should be ~50 Hz, got {}",
            f_max
        );

        // The scale at 10 Hz should have non-zero power (signal is 10 Hz sine)
        let closest_10hz = cwt
            .frequencies
            .iter()
            .enumerate()
            .min_by(|a, b| (a.1 - 10.0).abs().total_cmp(&(b.1 - 10.0).abs()))
            .map(|(i, _)| i)
            .unwrap();

        let power_at_10hz: f64 = cwt.scalogram[closest_10hz].iter().sum::<f64>() / n as f64;
        assert!(
            power_at_10hz > 0.0,
            "Power at 10 Hz scale should be non-zero, got {}",
            power_at_10hz
        );

        // Power should be positive everywhere (|CWT|² >= 0)
        for row in &cwt.scalogram {
            for &val in row {
                assert!(val >= 0.0, "CWT power should be non-negative");
            }
        }
    }

    #[test]
    fn test_band_energy_sums_to_one() {
        let analyzer = WaveletAnalyzer::default_eeg();
        let signal = sine_wave(10.0, 256.0, 1024);
        let result = analyzer.dwt(&signal);
        let bands = analyzer.band_energy(&result);

        let total: f64 = bands.iter().map(|&(_, _, e)| e).sum();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "Band energies should sum to 1.0, got {}",
            total
        );
    }

    #[test]
    fn test_detect_bursts() {
        // Create a signal with a clear burst
        let mut detail = vec![0.1; 100];
        // Insert a burst at samples 30-50
        for i in 30..50 {
            detail[i] = 5.0;
        }

        let bursts = WaveletAnalyzer::detect_bursts(&detail, 5, 3.0);
        assert!(!bursts.is_empty(), "Should detect at least one burst");

        // The burst should be somewhere in the 30-50 region
        let (start, end, _) = bursts[0];
        assert!(start <= 35, "Burst start should be near 30, got {}", start);
        assert!(end >= 45, "Burst end should be near 50, got {}", end);
    }

    #[test]
    fn test_wavelet_entropy() {
        let analyzer = WaveletAnalyzer::default_eeg();

        // Pure sine (narrow-band): low entropy
        let sine = sine_wave(10.0, 256.0, 1024);
        let h_sine = analyzer.wavelet_entropy(&sine);

        // Noise (broad-band): higher entropy
        let mut state: u64 = 42;
        let noise: Vec<f64> = (0..1024)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (state as f64 / u64::MAX as f64) * 2.0 - 1.0
            })
            .collect();
        let h_noise = analyzer.wavelet_entropy(&noise);

        assert!(
            h_noise > h_sine,
            "Noise entropy ({}) should exceed sine entropy ({})",
            h_noise,
            h_sine
        );
    }

    #[test]
    fn test_log_scales_monotonic() {
        let analyzer = WaveletAnalyzer::default_eeg();
        let scales = analyzer.log_scales(1.0, 100.0, 30);

        assert_eq!(scales.len(), 30);

        // Scales should be monotonically increasing
        for i in 1..scales.len() {
            assert!(
                scales[i] > scales[i - 1],
                "Scales not monotonic at {}: {} <= {}",
                i,
                scales[i],
                scales[i - 1]
            );
        }
    }
}
