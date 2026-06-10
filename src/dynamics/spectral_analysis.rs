// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Frequency Domain Analysis for CfC Dynamics
//!
//! Provides spectral analysis tools for decomposing neural dynamics signals into
//! their frequency components. Neural oscillations and their frequency content
//! are key signatures of conscious states -- delta rhythms dominate deep sleep,
//! alpha rhythms characterize relaxed wakefulness, gamma oscillations correlate
//! with conscious perception and binding, and cross-frequency coupling reflects
//! information integration across cortical scales.
//!
//! ## Core Capabilities
//!
//! - **FFT**: Cooley-Tukey radix-2 Fast Fourier Transform (and inverse)
//! - **Power Spectral Density**: Single-window and Welch's averaged method
//! - **Band Power Extraction**: Delta, Theta, Alpha, Beta, Gamma
//! - **Coherence**: Frequency-domain correlation between two signals
//! - **Spectral Entropy**: Information-theoretic measure of spectral complexity
//!
//! ## Consciousness Connection
//!
//! The frequency content of CfC time-constant trajectories reveals the temporal
//! structure of neural dynamics. A CfC network undergoing conscious processing
//! produces characteristic spectral signatures:
//!
//! - **High gamma power**: Indicates active binding and integration (high Phi)
//! - **Dominant alpha**: Reflects idle but ready states (moderate Phi)
//! - **Delta dominance**: Suggests consolidated, low-complexity processing
//! - **Broadband (high entropy)**: Indicates rich, complex conscious content
//! - **Narrow-band (low entropy)**: Suggests stereotyped, repetitive processing
//!
//! Cross-signal coherence further reveals how different brain regions (or CfC
//! subnetworks) synchronize at specific frequencies, a hallmark of conscious
//! integration as proposed by IIT and Global Workspace Theory.
//!
//! ## Example
//!
//! ```rust,ignore
//! use symthaea::dynamics::spectral_analysis::{SpectralAnalyzer, SpectralConfig};
//!
//! let config = SpectralConfig {
//!     sample_rate: 256.0,
//!     ..Default::default()
//! };
//! let analyzer = SpectralAnalyzer::new(config);
//!
//! // Analyze CfC tau trajectory
//! let tau_trajectory: Vec<f64> = collect_tau_values();
//! let spectrum = analyzer.welch(&tau_trajectory);
//! let bands = analyzer.band_power(&tau_trajectory);
//! let entropy = analyzer.spectral_entropy(&tau_trajectory);
//!
//! println!("Alpha/Theta ratio: {:.3}", bands.alpha / bands.theta.max(1e-12));
//! println!("Spectral entropy: {:.3} bits", entropy);
//! ```

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Complex number type (minimal, no external dependency)
// ---------------------------------------------------------------------------

/// Minimal complex number for FFT computations.
///
/// We intentionally avoid pulling in a full `num-complex` crate; the operations
/// required for spectral analysis are few and straightforward.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl Complex {
    /// Create a new complex number.
    #[inline]
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// Zero (additive identity).
    #[inline]
    pub fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    /// Construct from polar form: r * exp(i*theta).
    #[inline]
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }

    /// Squared magnitude |z|^2.
    #[inline]
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Magnitude |z|.
    #[inline]
    pub fn norm(self) -> f64 {
        self.norm_sqr().sqrt()
    }

    /// Complex conjugate.
    #[inline]
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Phase angle in radians.
    #[inline]
    pub fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }
}

impl std::ops::Add for Complex {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl std::ops::Mul<f64> for Complex {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self {
            re: self.re * rhs,
            im: self.im * rhs,
        }
    }
}

impl std::ops::AddAssign for Complex {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

// ---------------------------------------------------------------------------
// Configuration & result types
// ---------------------------------------------------------------------------

/// Window function applied before computing the DFT.
///
/// Windowing reduces spectral leakage by tapering the signal edges to zero.
/// Hann is the default and a good general-purpose choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WindowType {
    /// No window (equivalent to a rectangular window). Maximum frequency
    /// resolution but worst spectral leakage.
    Rectangular,
    /// Hann window: w[n] = 0.5 (1 - cos(2 pi n / (N-1))).
    /// Good compromise between resolution and leakage.
    Hann,
    /// Hamming window: w[n] = 0.54 - 0.46 cos(2 pi n / (N-1)).
    /// Slightly less sidelobe suppression than Hann but narrower main lobe.
    Hamming,
    /// Blackman window: w[n] = 0.42 - 0.5 cos(2 pi n / (N-1)) + 0.08 cos(4 pi n / (N-1)).
    /// Excellent sidelobe suppression at the cost of wider main lobe.
    Blackman,
}

impl WindowType {
    /// Compute the window coefficient for sample index `n` in a window of
    /// length `len`.
    #[inline]
    fn coefficient(self, n: usize, len: usize) -> f64 {
        if len <= 1 {
            return 1.0;
        }
        let n_f = n as f64;
        let nm1 = (len - 1) as f64;
        match self {
            WindowType::Rectangular => 1.0,
            WindowType::Hann => 0.5 * (1.0 - (2.0 * PI * n_f / nm1).cos()),
            WindowType::Hamming => 0.54 - 0.46 * (2.0 * PI * n_f / nm1).cos(),
            WindowType::Blackman => {
                // Clamp to 0.0 to avoid tiny negative values from floating point
                (0.42 - 0.5 * (2.0 * PI * n_f / nm1).cos() + 0.08 * (4.0 * PI * n_f / nm1).cos())
                    .max(0.0)
            }
        }
    }

    /// Generate the full window of length `len`.
    fn generate(self, len: usize) -> Vec<f64> {
        (0..len).map(|n| self.coefficient(n, len)).collect()
    }
}

/// Configuration for spectral analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralConfig {
    /// FFT window size (will be rounded up to next power of 2 internally).
    pub window_size: usize,
    /// Overlap fraction for Welch's method, in [0, 1). Typical: 0.5.
    pub overlap: f64,
    /// Window function.
    pub window_type: WindowType,
    /// Sampling rate in Hz.
    pub sample_rate: f64,
}

impl Default for SpectralConfig {
    fn default() -> Self {
        Self {
            window_size: 256,
            overlap: 0.5,
            window_type: WindowType::Hann,
            sample_rate: 256.0,
        }
    }
}

/// Frequency spectrum produced by DFT / PSD analysis.
///
/// Only the non-negative frequencies (0..N/2+1) are stored because the input
/// signal is real-valued and the spectrum is conjugate-symmetric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencySpectrum {
    /// Frequency axis in Hz (length N/2 + 1).
    pub frequencies: Vec<f64>,
    /// Magnitude |X(f)| (linear scale).
    pub magnitudes: Vec<f64>,
    /// Phase angle in radians.
    pub phases: Vec<f64>,
    /// Power spectral density S(f) = |X(f)|^2 / N.
    pub psd: Vec<f64>,
}

/// Coherence analysis result between two signals.
///
/// Coherence C_xy(f) measures the linear correlation at each frequency,
/// analogous to a squared correlation coefficient in the frequency domain.
/// Values range from 0 (no relationship) to 1 (perfect linear relationship).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceResult {
    /// Frequency axis in Hz.
    pub frequencies: Vec<f64>,
    /// Magnitude-squared coherence in [0, 1].
    pub coherence: Vec<f64>,
    /// Phase difference in radians, derived from the cross-spectral density.
    pub phase_difference: Vec<f64>,
}

/// Power in the standard neuroscience frequency bands.
///
/// These bands partition the EEG / neural oscillation spectrum into
/// functionally meaningful ranges. The values represent integrated power
/// (area under the PSD curve) in each band.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandPower {
    /// Delta band power (0.5 -- 4 Hz). Dominant in deep (N3) sleep.
    pub delta: f64,
    /// Theta band power (4 -- 8 Hz). Associated with drowsiness and memory encoding.
    pub theta: f64,
    /// Alpha band power (8 -- 13 Hz). Reflects relaxed wakefulness, cortical idling.
    pub alpha: f64,
    /// Beta band power (13 -- 30 Hz). Associated with active thinking and focus.
    pub beta: f64,
    /// Gamma band power (30 -- 100 Hz). Correlates with conscious binding and integration.
    pub gamma: f64,
    /// Total power across all bands.
    pub total: f64,
}

// ---------------------------------------------------------------------------
// SpectralAnalyzer
// ---------------------------------------------------------------------------

/// Frequency-domain analysis engine for CfC neural dynamics signals.
///
/// Wraps a [`SpectralConfig`] and exposes methods for FFT, PSD, Welch's
/// method, coherence, band power extraction, and spectral entropy. All
/// computations are performed in `f64` precision with no external numeric
/// library dependencies.
#[derive(Debug, Clone)]
pub struct SpectralAnalyzer {
    /// Analysis configuration.
    pub config: SpectralConfig,
}

impl SpectralAnalyzer {
    /// Create a new analyzer with the given configuration.
    pub fn new(config: SpectralConfig) -> Self {
        Self { config }
    }

    // -----------------------------------------------------------------------
    // FFT / IFFT
    // -----------------------------------------------------------------------

    /// Cooley-Tukey radix-2 decimation-in-time FFT.
    ///
    /// The input signal is zero-padded to the next power of two if necessary.
    ///
    /// Returns the full N-point complex spectrum.  For a real input of length
    /// N the spectrum is conjugate-symmetric: X[k] = X*[N-k].
    pub fn fft(signal: &[f64]) -> Vec<Complex> {
        let n = signal.len().next_power_of_two();
        let mut buf: Vec<Complex> = signal
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .chain(std::iter::repeat_n(Complex::zero(), n - signal.len()))
            .collect();
        Self::fft_in_place(&mut buf);
        buf
    }

    /// Inverse FFT.  Applies FFT with conjugate twiddle factors, then divides
    /// by N to recover the original signal.
    ///
    /// Returns the real parts (imaginary parts should be ~0 for a real signal).
    pub fn ifft(spectrum: &[Complex]) -> Vec<f64> {
        let n = spectrum.len().next_power_of_two();
        // Conjugate input
        let mut buf: Vec<Complex> = spectrum
            .iter()
            .map(|z| z.conj())
            .chain(std::iter::repeat_n(Complex::zero(), n - spectrum.len()))
            .collect();
        Self::fft_in_place(&mut buf);
        let inv = 1.0 / n as f64;
        buf.iter().map(|z| z.re * inv).collect()
    }

    /// In-place radix-2 FFT.  `buf.len()` **must** be a power of two.
    fn fft_in_place(buf: &mut [Complex]) {
        let n = buf.len();
        debug_assert!(n.is_power_of_two(), "FFT length must be power of 2");
        if n <= 1 {
            return;
        }

        // Bit-reversal permutation
        let mut j: usize = 0;
        for i in 0..n {
            if i < j {
                buf.swap(i, j);
            }
            let mut m = n >> 1;
            while m >= 1 && j >= m {
                j -= m;
                m >>= 1;
            }
            j += m;
        }

        // Butterfly stages
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let angle = -2.0 * PI / len as f64;
            let wn = Complex::from_polar(1.0, angle);
            let mut start = 0;
            while start < n {
                let mut w = Complex::new(1.0, 0.0);
                for k in 0..half {
                    let u = buf[start + k];
                    let t = w * buf[start + k + half];
                    buf[start + k] = u + t;
                    buf[start + k + half] = u - t;
                    w = w * wn;
                }
                start += len;
            }
            len <<= 1;
        }
    }

    // -----------------------------------------------------------------------
    // Power Spectral Density
    // -----------------------------------------------------------------------

    /// Compute the one-sided power spectral density using a single windowed FFT.
    ///
    /// PSD is defined as S(f) = |X(f)|^2 / N, giving power per frequency bin.
    /// Only non-negative frequencies are returned (N/2 + 1 points).
    pub fn psd(&self, signal: &[f64]) -> FrequencySpectrum {
        let _n = self
            .config
            .window_size
            .max(signal.len())
            .next_power_of_two();

        // Apply window
        let window = self.config.window_type.generate(signal.len());
        let windowed: Vec<f64> = signal
            .iter()
            .zip(window.iter())
            .map(|(&s, &w)| s * w)
            .collect();

        // Window power correction factor (preserves total power)
        let window_power: f64 = window.iter().map(|w| w * w).sum::<f64>() / window.len() as f64;
        let correction = if window_power > 0.0 {
            1.0 / window_power
        } else {
            1.0
        };

        let spectrum = Self::fft(&windowed);
        let n_fft = spectrum.len();
        let n_pos = n_fft / 2 + 1; // non-negative frequencies
        let df = self.config.sample_rate / n_fft as f64;

        let mut frequencies = Vec::with_capacity(n_pos);
        let mut magnitudes = Vec::with_capacity(n_pos);
        let mut phases = Vec::with_capacity(n_pos);
        let mut psd = Vec::with_capacity(n_pos);

        for k in 0..n_pos {
            let freq = k as f64 * df;
            let mag = spectrum[k].norm();
            let phase = spectrum[k].arg();
            // One-sided PSD: double power for non-DC, non-Nyquist bins
            let scale = if k == 0 || k == n_fft / 2 { 1.0 } else { 2.0 };
            let p = scale * spectrum[k].norm_sqr() / n_fft as f64 * correction;

            frequencies.push(freq);
            magnitudes.push(mag);
            phases.push(phase);
            psd.push(p);
        }

        FrequencySpectrum {
            frequencies,
            magnitudes,
            phases,
            psd,
        }
    }

    // -----------------------------------------------------------------------
    // Welch's method
    // -----------------------------------------------------------------------

    /// Estimate the PSD using Welch's method (averaged, overlapping segments).
    ///
    /// The signal is divided into segments of `window_size` samples with
    /// `overlap` fraction overlap.  Each segment is windowed and its
    /// periodogram computed.  The final PSD is the average of all periodograms.
    ///
    /// Welch's method reduces the variance of the PSD estimate at the cost of
    /// frequency resolution.
    pub fn welch(&self, signal: &[f64]) -> FrequencySpectrum {
        let win_len = self.config.window_size;
        if signal.len() <= win_len {
            // Signal too short for segmenting; fall back to single-window PSD.
            return self.psd(signal);
        }

        let step = ((1.0 - self.config.overlap.clamp(0.0, 0.99)) * win_len as f64) as usize;
        let step = step.max(1);

        let window = self.config.window_type.generate(win_len);
        let window_power: f64 = window.iter().map(|w| w * w).sum::<f64>() / win_len as f64;
        let correction = if window_power > 0.0 {
            1.0 / window_power
        } else {
            1.0
        };

        let n_fft = win_len.next_power_of_two();
        let n_pos = n_fft / 2 + 1;
        let df = self.config.sample_rate / n_fft as f64;

        let mut avg_psd = vec![0.0f64; n_pos];
        let mut n_segments = 0usize;

        let mut offset = 0;
        while offset + win_len <= signal.len() {
            let segment = &signal[offset..offset + win_len];
            let windowed: Vec<f64> = segment
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| s * w)
                .collect();

            let spectrum = Self::fft(&windowed);

            for k in 0..n_pos {
                let scale = if k == 0 || k == n_fft / 2 { 1.0 } else { 2.0 };
                avg_psd[k] += scale * spectrum[k].norm_sqr() / n_fft as f64 * correction;
            }

            n_segments += 1;
            offset += step;
        }

        if n_segments > 0 {
            let inv = 1.0 / n_segments as f64;
            for p in avg_psd.iter_mut() {
                *p *= inv;
            }
        }

        // Build output (magnitude and phase from last segment for illustration;
        // the primary output of Welch is the averaged PSD).
        let frequencies: Vec<f64> = (0..n_pos).map(|k| k as f64 * df).collect();
        let magnitudes: Vec<f64> = avg_psd.iter().map(|&p| p.sqrt()).collect();
        // Phase is not meaningful for averaged PSD; fill with zeros.
        let phases = vec![0.0; n_pos];

        FrequencySpectrum {
            frequencies,
            magnitudes,
            phases,
            psd: avg_psd,
        }
    }

    // -----------------------------------------------------------------------
    // Coherence
    // -----------------------------------------------------------------------

    /// Compute magnitude-squared coherence between two signals.
    ///
    /// C_xy(f) = |S_xy(f)|^2 / (S_xx(f) * S_yy(f))
    ///
    /// Uses Welch-style segment averaging to obtain stable cross- and auto-
    /// spectral density estimates.  Coherence values lie in [0, 1].
    ///
    /// Both signals must be the same length; panics otherwise.
    pub fn coherence(&self, x: &[f64], y: &[f64]) -> CoherenceResult {
        assert_eq!(x.len(), y.len(), "Coherence requires equal-length signals");

        let win_len = self.config.window_size.min(x.len());
        let step = ((1.0 - self.config.overlap.clamp(0.0, 0.99)) * win_len as f64) as usize;
        let step = step.max(1);
        let window = self.config.window_type.generate(win_len);

        let n_fft = win_len.next_power_of_two();
        let n_pos = n_fft / 2 + 1;
        let df = self.config.sample_rate / n_fft as f64;

        let mut sxx = vec![0.0f64; n_pos];
        let mut syy = vec![0.0f64; n_pos];
        let mut sxy = vec![Complex::zero(); n_pos];
        let mut n_segments = 0usize;

        let mut offset = 0;
        while offset + win_len <= x.len() {
            let seg_x: Vec<f64> = x[offset..offset + win_len]
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| s * w)
                .collect();
            let seg_y: Vec<f64> = y[offset..offset + win_len]
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| s * w)
                .collect();

            let fx = Self::fft(&seg_x);
            let fy = Self::fft(&seg_y);

            for k in 0..n_pos {
                sxx[k] += fx[k].norm_sqr();
                syy[k] += fy[k].norm_sqr();
                // Cross-spectral density: X*(f) * Y(f)
                sxy[k] += fx[k].conj() * fy[k];
            }

            n_segments += 1;
            offset += step;
        }

        // If signal is too short for even one segment, compute on the full signal.
        if n_segments == 0 {
            let seg_x: Vec<f64> = x
                .iter()
                .zip(window.iter().chain(std::iter::repeat(&1.0)))
                .map(|(&s, &w)| s * w)
                .collect();
            let seg_y: Vec<f64> = y
                .iter()
                .zip(window.iter().chain(std::iter::repeat(&1.0)))
                .map(|(&s, &w)| s * w)
                .collect();
            let fx = Self::fft(&seg_x);
            let fy = Self::fft(&seg_y);
            for k in 0..n_pos.min(fx.len()) {
                sxx[k] += fx[k].norm_sqr();
                syy[k] += fy[k].norm_sqr();
                sxy[k] += fx[k].conj() * fy[k];
            }
            // n_segments remains 1 implicitly (single-pass, no averaging needed)
        }

        let frequencies: Vec<f64> = (0..n_pos).map(|k| k as f64 * df).collect();
        let mut coherence = Vec::with_capacity(n_pos);
        let mut phase_difference = Vec::with_capacity(n_pos);

        for k in 0..n_pos {
            let denom = sxx[k] * syy[k];
            if denom > 0.0 {
                let coh = sxy[k].norm_sqr() / denom;
                coherence.push(coh.clamp(0.0, 1.0));
            } else {
                coherence.push(0.0);
            }
            phase_difference.push(sxy[k].arg());
        }

        CoherenceResult {
            frequencies,
            coherence,
            phase_difference,
        }
    }

    // -----------------------------------------------------------------------
    // Band power extraction
    // -----------------------------------------------------------------------

    /// Extract power in the standard neuroscience frequency bands.
    ///
    /// Uses Welch's method internally for a stable PSD estimate, then integrates
    /// the PSD over each band:
    ///
    /// | Band  | Range (Hz) | Consciousness correlate       |
    /// |-------|------------|-------------------------------|
    /// | Delta | 0.5 -- 4   | Deep sleep, unconsciousness   |
    /// | Theta | 4 -- 8     | Drowsiness, memory encoding   |
    /// | Alpha | 8 -- 13    | Relaxed wakefulness, idling   |
    /// | Beta  | 13 -- 30   | Active thinking, focus         |
    /// | Gamma | 30 -- 100  | Binding, conscious integration |
    pub fn band_power(&self, signal: &[f64]) -> BandPower {
        let spectrum = self.welch(signal);
        let df = if spectrum.frequencies.len() > 1 {
            spectrum.frequencies[1] - spectrum.frequencies[0]
        } else {
            1.0
        };

        let mut delta = 0.0;
        let mut theta = 0.0;
        let mut alpha = 0.0;
        let mut beta = 0.0;
        let mut gamma = 0.0;

        for (i, &freq) in spectrum.frequencies.iter().enumerate() {
            let power = spectrum.psd[i] * df;
            if (0.5..4.0).contains(&freq) {
                delta += power;
            } else if (4.0..8.0).contains(&freq) {
                theta += power;
            } else if (8.0..13.0).contains(&freq) {
                alpha += power;
            } else if (13.0..30.0).contains(&freq) {
                beta += power;
            } else if (30.0..=100.0).contains(&freq) {
                gamma += power;
            }
        }

        let total = delta + theta + alpha + beta + gamma;

        BandPower {
            delta,
            theta,
            alpha,
            beta,
            gamma,
            total,
        }
    }

    // -----------------------------------------------------------------------
    // Spectral entropy
    // -----------------------------------------------------------------------

    /// Compute spectral entropy of the signal.
    ///
    /// H = -sum( p(f) * log2(p(f)) ) where p(f) = S(f) / sum(S).
    ///
    /// Low entropy indicates a narrow-band, periodic signal (e.g. a pure sine
    /// wave has near-zero entropy). High entropy indicates a broadband signal
    /// carrying rich spectral content (white noise approaches log2(N/2)).
    ///
    /// In the context of consciousness, spectral entropy is a proxy for the
    /// complexity and richness of neural dynamics -- a known correlate of the
    /// level of consciousness (as used in the Perturbational Complexity Index).
    pub fn spectral_entropy(&self, signal: &[f64]) -> f64 {
        let spectrum = self.welch(signal);
        let total: f64 = spectrum.psd.iter().sum();
        if total <= 0.0 {
            return 0.0;
        }

        let mut entropy = 0.0;
        for &p in &spectrum.psd {
            let prob = p / total;
            if prob > 1e-15 {
                entropy -= prob * prob.log2();
            }
        }
        entropy
    }

    // -----------------------------------------------------------------------
    // Utility: next-power-of-2 window size
    // -----------------------------------------------------------------------

    /// Return the effective FFT length (next power of 2 >= window_size).
    pub fn fft_length(&self) -> usize {
        self.config.window_size.next_power_of_two()
    }
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: generate a pure sine wave.
    fn sine_wave(freq: f64, sample_rate: f64, n_samples: usize) -> Vec<f64> {
        (0..n_samples)
            .map(|i| (2.0 * PI * freq * i as f64 / sample_rate).sin())
            .collect()
    }

    /// Helper: generate white noise via a simple LCG (deterministic).
    fn white_noise(n_samples: usize) -> Vec<f64> {
        let mut state: u64 = 12345;
        (0..n_samples)
            .map(|_| {
                // Simple LCG: x_{n+1} = (a*x_n + c) mod m
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                // Map to [-1, 1]
                (state as f64 / u64::MAX as f64) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn test_complex_arithmetic() {
        let a = Complex::new(3.0, 4.0);
        let b = Complex::new(1.0, -2.0);

        let sum = a + b;
        assert!((sum.re - 4.0).abs() < 1e-12);
        assert!((sum.im - 2.0).abs() < 1e-12);

        let diff = a - b;
        assert!((diff.re - 2.0).abs() < 1e-12);
        assert!((diff.im - 6.0).abs() < 1e-12);

        let prod = a * b;
        // (3+4i)(1-2i) = 3 -6i +4i -8i^2 = 11 -2i
        assert!((prod.re - 11.0).abs() < 1e-12);
        assert!((prod.im - (-2.0)).abs() < 1e-12);

        assert!((a.norm() - 5.0).abs() < 1e-12);
        assert!((a.conj().im - (-4.0)).abs() < 1e-12);
    }

    #[test]
    fn test_fft_pure_sine_single_peak() {
        // A 10 Hz sine at 256 Hz sample rate over 256 samples.
        let sample_rate = 256.0;
        let n = 256;
        let freq = 10.0;
        let signal = sine_wave(freq, sample_rate, n);

        let spectrum = SpectralAnalyzer::fft(&signal);
        assert_eq!(spectrum.len(), n); // power of 2, same as input

        // Find magnitude peak (excluding DC)
        let df = sample_rate / n as f64; // 1 Hz per bin
        let mut max_mag = 0.0;
        let mut max_bin = 0;
        for k in 1..n / 2 {
            let mag = spectrum[k].norm();
            if mag > max_mag {
                max_mag = mag;
                max_bin = k;
            }
        }

        let peak_freq = max_bin as f64 * df;
        assert!(
            (peak_freq - freq).abs() < df,
            "Peak at {peak_freq} Hz, expected {freq} Hz"
        );
    }

    #[test]
    fn test_fft_ifft_roundtrip() {
        // Verify that FFT followed by IFFT recovers the original signal.
        let signal: Vec<f64> = (0..128)
            .map(|i| (0.3 * i as f64).sin() + 0.5 * (1.7 * i as f64).cos())
            .collect();

        let spectrum = SpectralAnalyzer::fft(&signal);
        let recovered = SpectralAnalyzer::ifft(&spectrum);

        for i in 0..signal.len() {
            assert!(
                (recovered[i] - signal[i]).abs() < 1e-10,
                "Mismatch at index {i}: recovered={}, original={}",
                recovered[i],
                signal[i]
            );
        }
    }

    #[test]
    fn test_psd_white_noise_approximately_flat() {
        let sample_rate = 256.0;
        let noise = white_noise(1024);
        let config = SpectralConfig {
            window_size: 256,
            overlap: 0.5,
            window_type: WindowType::Hann,
            sample_rate,
        };
        let analyzer = SpectralAnalyzer::new(config);
        let spectrum = analyzer.welch(&noise);

        // Exclude DC and Nyquist; check that the PSD does not vary too wildly.
        let psd_inner = &spectrum.psd[1..spectrum.psd.len() - 1];
        let mean: f64 = psd_inner.iter().sum::<f64>() / psd_inner.len() as f64;
        // For white noise, no bin should be more than ~10x the mean with Welch averaging.
        for (i, &p) in psd_inner.iter().enumerate() {
            assert!(
                p < mean * 10.0,
                "PSD bin {i} = {p} is >10x the mean {mean}, not flat enough"
            );
        }
    }

    #[test]
    fn test_band_power_assigns_sine_to_correct_band() {
        let sample_rate = 256.0;

        // 2 Hz sine -> should land in Delta (0.5-4 Hz)
        let delta_sig = sine_wave(2.0, sample_rate, 1024);
        // 10 Hz sine -> should land in Alpha (8-13 Hz)
        let alpha_sig = sine_wave(10.0, sample_rate, 1024);
        // 40 Hz sine -> should land in Gamma (30-100 Hz)
        let gamma_sig = sine_wave(40.0, sample_rate, 1024);

        let config = SpectralConfig {
            window_size: 256,
            overlap: 0.5,
            window_type: WindowType::Hann,
            sample_rate,
        };
        let analyzer = SpectralAnalyzer::new(config);

        let bp_delta = analyzer.band_power(&delta_sig);
        assert!(
            bp_delta.delta > bp_delta.theta
                && bp_delta.delta > bp_delta.alpha
                && bp_delta.delta > bp_delta.beta
                && bp_delta.delta > bp_delta.gamma,
            "2 Hz sine should have highest delta power: {:?}",
            bp_delta
        );

        let bp_alpha = analyzer.band_power(&alpha_sig);
        assert!(
            bp_alpha.alpha > bp_alpha.delta
                && bp_alpha.alpha > bp_alpha.theta
                && bp_alpha.alpha > bp_alpha.beta
                && bp_alpha.alpha > bp_alpha.gamma,
            "10 Hz sine should have highest alpha power: {:?}",
            bp_alpha
        );

        let bp_gamma = analyzer.band_power(&gamma_sig);
        assert!(
            bp_gamma.gamma > bp_gamma.delta
                && bp_gamma.gamma > bp_gamma.theta
                && bp_gamma.gamma > bp_gamma.alpha
                && bp_gamma.gamma > bp_gamma.beta,
            "40 Hz sine should have highest gamma power: {:?}",
            bp_gamma
        );
    }

    #[test]
    fn test_coherence_identical_signals() {
        let sample_rate = 256.0;
        let signal = sine_wave(10.0, sample_rate, 512);
        let config = SpectralConfig {
            window_size: 128,
            overlap: 0.5,
            window_type: WindowType::Hann,
            sample_rate,
        };
        let analyzer = SpectralAnalyzer::new(config);

        let result = analyzer.coherence(&signal, &signal);

        // Coherence of a signal with itself should be 1.0 at all frequencies
        // where there is actual power.
        let peak_bin = result
            .coherence
            .iter()
            .enumerate()
            .skip(1) // skip DC
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert!(
            result.coherence[peak_bin] > 0.99,
            "Self-coherence at peak should be ~1.0, got {}",
            result.coherence[peak_bin]
        );

        // All non-zero-power bins should have coherence very close to 1.
        for (i, &c) in result.coherence.iter().enumerate() {
            if c > 0.01 {
                // only check bins with non-negligible power
                assert!(c > 0.99, "Self-coherence at bin {i} = {c}, expected ~1.0");
            }
        }
    }

    #[test]
    fn test_spectral_entropy_sine_less_than_noise() {
        let sample_rate = 256.0;
        let sine = sine_wave(10.0, sample_rate, 1024);
        let noise = white_noise(1024);

        let config = SpectralConfig {
            window_size: 256,
            overlap: 0.5,
            window_type: WindowType::Hann,
            sample_rate,
        };
        let analyzer = SpectralAnalyzer::new(config);

        let h_sine = analyzer.spectral_entropy(&sine);
        let h_noise = analyzer.spectral_entropy(&noise);

        assert!(
            h_sine < h_noise,
            "Pure sine entropy ({h_sine:.3}) should be less than white noise entropy ({h_noise:.3})"
        );
        // Sine entropy should be well below the noise floor.
        assert!(
            h_sine < h_noise * 0.8,
            "Sine entropy ({h_sine:.3}) should be substantially less than noise ({h_noise:.3})"
        );
    }

    #[test]
    fn test_window_functions() {
        let n = 64;
        for wt in &[
            WindowType::Rectangular,
            WindowType::Hann,
            WindowType::Hamming,
            WindowType::Blackman,
        ] {
            let w = wt.generate(n);
            assert_eq!(w.len(), n);

            // All coefficients should be in [0, 1]
            for &c in &w {
                assert!((0.0..=1.0).contains(&c), "{:?} coeff {c} out of [0,1]", wt);
            }

            // Rectangular should be all 1s
            if *wt == WindowType::Rectangular {
                for &c in &w {
                    assert!((c - 1.0).abs() < 1e-12);
                }
            }

            // Non-rectangular windows should taper to ~0 at edges
            if *wt != WindowType::Rectangular {
                assert!(
                    w[0] < 0.1,
                    "{:?} first coeff {:.4} should be near 0",
                    wt,
                    w[0]
                );
                assert!(
                    w[n - 1] < 0.1,
                    "{:?} last coeff {:.4} should be near 0",
                    wt,
                    w[n - 1]
                );
            }
        }
    }

    #[test]
    fn test_fft_dc_component() {
        // A constant signal should have all energy at DC (bin 0).
        let signal = vec![3.0; 64];
        let spectrum = SpectralAnalyzer::fft(&signal);

        let dc_mag = spectrum[0].norm();
        assert!(
            (dc_mag - 3.0 * 64.0).abs() < 1e-8,
            "DC magnitude should be N*value = {}, got {dc_mag}",
            3.0 * 64.0
        );

        // All other bins should be ~0.
        for k in 1..spectrum.len() {
            assert!(
                spectrum[k].norm() < 1e-10,
                "Non-DC bin {k} has magnitude {}, expected ~0",
                spectrum[k].norm()
            );
        }
    }

    #[test]
    fn test_welch_reduces_variance() {
        // Welch's method should produce a smoother PSD than a single window.
        let sample_rate = 256.0;
        let noise = white_noise(2048);

        let config_single = SpectralConfig {
            window_size: 2048,
            overlap: 0.5,
            window_type: WindowType::Hann,
            sample_rate,
        };
        let config_welch = SpectralConfig {
            window_size: 256,
            overlap: 0.5,
            window_type: WindowType::Hann,
            sample_rate,
        };

        let analyzer_single = SpectralAnalyzer::new(config_single);
        let analyzer_welch = SpectralAnalyzer::new(config_welch);

        let psd_single = analyzer_single.psd(&noise);
        let psd_welch = analyzer_welch.welch(&noise);

        // Compute coefficient of variation (std / mean) for each.
        fn coeff_var(data: &[f64]) -> f64 {
            let n = data.len() as f64;
            let mean = data.iter().sum::<f64>() / n;
            if mean.abs() < 1e-15 {
                return 0.0;
            }
            let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
            var.sqrt() / mean
        }

        let cv_single = coeff_var(&psd_single.psd[1..psd_single.psd.len() - 1]);
        let cv_welch = coeff_var(&psd_welch.psd[1..psd_welch.psd.len() - 1]);

        assert!(
            cv_welch < cv_single,
            "Welch CV ({cv_welch:.3}) should be less than single-window CV ({cv_single:.3})"
        );
    }

    #[test]
    fn test_phase_spectrum() {
        // A cosine (zero phase shift) should have phase ~0 at its frequency.
        let sample_rate = 256.0;
        let n = 256;
        let freq = 8.0;
        let cosine: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / sample_rate).cos())
            .collect();

        let config = SpectralConfig {
            window_size: n,
            overlap: 0.5,
            window_type: WindowType::Rectangular, // no window distortion
            sample_rate,
        };
        let analyzer = SpectralAnalyzer::new(config);
        let spectrum = analyzer.psd(&cosine);

        // Find the peak bin
        let peak_bin = spectrum
            .psd
            .iter()
            .enumerate()
            .skip(1)
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;

        // Phase at the peak should be close to 0 for a cosine.
        assert!(
            spectrum.phases[peak_bin].abs() < 0.1,
            "Cosine phase at peak bin {peak_bin} = {:.4}, expected ~0",
            spectrum.phases[peak_bin]
        );
    }

    #[test]
    fn test_band_power_total() {
        let sample_rate = 256.0;
        let signal = white_noise(1024);
        let config = SpectralConfig {
            window_size: 256,
            overlap: 0.5,
            window_type: WindowType::Hann,
            sample_rate,
        };
        let analyzer = SpectralAnalyzer::new(config);
        let bp = analyzer.band_power(&signal);

        // Total should equal sum of bands.
        let sum = bp.delta + bp.theta + bp.alpha + bp.beta + bp.gamma;
        assert!(
            (bp.total - sum).abs() < 1e-12,
            "Total {:.6} != sum of bands {:.6}",
            bp.total,
            sum
        );
    }
}
