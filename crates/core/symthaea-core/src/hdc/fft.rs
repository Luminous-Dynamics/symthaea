// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Fast Fourier Transform Engine
//!
//! Cooley-Tukey radix-2 FFT and inverse FFT with HDC encoding.
//!
//! ## Capabilities
//!
//! - Forward FFT (time domain → frequency domain)
//! - Inverse FFT (frequency domain → time domain)
//! - Power spectrum
//! - Convolution via FFT (O(n log n) instead of O(n²))

use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::primitive_system::seed_from_name;
use serde::{Deserialize, Serialize};

// ─── Complex Number ──────────────────────────────────────────────────────────

/// Simple complex number for FFT operations
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    pub fn from_real(re: f64) -> Self {
        Self { re, im: 0.0 }
    }

    pub fn magnitude(&self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    pub fn magnitude_squared(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    pub fn phase(&self) -> f64 {
        self.im.atan2(self.re)
    }

    pub fn conjugate(&self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }
}

impl std::ops::Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl std::ops::Mul<f64> for Complex {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self {
            re: self.re * rhs,
            im: self.im * rhs,
        }
    }
}

// ─── FFT Result ──────────────────────────────────────────────────────────────

/// Result of an FFT operation
#[derive(Debug, Clone)]
pub struct FftResult {
    /// Complex spectrum
    pub spectrum: Vec<Complex>,
    /// Number of points
    pub n: usize,
    /// Whether this is forward (true) or inverse (false) transform
    pub is_forward: bool,
    /// Phi measurement
    pub phi: f64,
    /// HDC encoding
    pub encoding: BinaryHV,
}

impl FftResult {
    /// Power spectrum (|X[k]|²)
    pub fn power_spectrum(&self) -> Vec<f64> {
        self.spectrum
            .iter()
            .map(|c| c.magnitude_squared())
            .collect()
    }

    /// Magnitude spectrum (|X[k]|)
    pub fn magnitude_spectrum(&self) -> Vec<f64> {
        self.spectrum.iter().map(|c| c.magnitude()).collect()
    }

    /// Phase spectrum
    pub fn phase_spectrum(&self) -> Vec<f64> {
        self.spectrum.iter().map(|c| c.phase()).collect()
    }

    /// Get real parts (useful for inverse FFT result)
    pub fn real_parts(&self) -> Vec<f64> {
        self.spectrum.iter().map(|c| c.re).collect()
    }
}

// ─── FFT Engine ──────────────────────────────────────────────────────────────

/// The Hyperdimensional FFT Engine
pub struct FftEngine;

impl FftEngine {
    /// Forward FFT (Cooley-Tukey radix-2 DIT).
    ///
    /// Input length must be a power of 2, or will be zero-padded.
    pub fn fft(signal: &[f64]) -> FftResult {
        let input: Vec<Complex> = signal.iter().map(|&x| Complex::from_real(x)).collect();
        let padded = Self::pad_to_power_of_2(input);
        let n = padded.len();

        let spectrum = Self::fft_recursive(&padded, false);

        let encoding = Self::encode_fft(&spectrum, true);

        FftResult {
            spectrum,
            n,
            is_forward: true,
            phi: 0.2 + 0.1 * (n as f64).log2() / 10.0,
            encoding,
        }
    }

    /// Forward FFT on complex input
    pub fn fft_complex(signal: &[Complex]) -> FftResult {
        let padded = Self::pad_to_power_of_2(signal.to_vec());
        let n = padded.len();
        let spectrum = Self::fft_recursive(&padded, false);

        FftResult {
            spectrum,
            n,
            is_forward: true,
            phi: 0.2,
            encoding: Self::encode_fft(signal, true),
        }
    }

    /// Inverse FFT.
    ///
    /// Recovers the time-domain signal from frequency domain.
    pub fn ifft(spectrum: &[Complex]) -> FftResult {
        let padded = Self::pad_to_power_of_2(spectrum.to_vec());
        let n = padded.len();

        let mut result = Self::fft_recursive(&padded, true);

        // Divide by N
        let scale = 1.0 / n as f64;
        for c in &mut result {
            c.re *= scale;
            c.im *= scale;
        }

        FftResult {
            spectrum: result,
            n,
            is_forward: false,
            phi: 0.2,
            encoding: Self::encode_fft(spectrum, false),
        }
    }

    /// Convolution via FFT: h = f * g = IFFT(FFT(f) · FFT(g))
    pub fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
        let n = (a.len() + b.len() - 1).next_power_of_two();

        let mut a_padded: Vec<Complex> = a.iter().map(|&x| Complex::from_real(x)).collect();
        a_padded.resize(n, Complex::zero());

        let mut b_padded: Vec<Complex> = b.iter().map(|&x| Complex::from_real(x)).collect();
        b_padded.resize(n, Complex::zero());

        let fa = Self::fft_recursive(&a_padded, false);
        let fb = Self::fft_recursive(&b_padded, false);

        let product: Vec<Complex> = fa.iter().zip(fb.iter()).map(|(&a, &b)| a * b).collect();

        let mut result = Self::fft_recursive(&product, true);
        let scale = 1.0 / n as f64;
        for c in &mut result {
            c.re *= scale;
            c.im *= scale;
        }

        result
            .iter()
            .take(a.len() + b.len() - 1)
            .map(|c| c.re)
            .collect()
    }

    /// Power spectrum of a real signal
    pub fn power_spectrum(signal: &[f64]) -> Vec<f64> {
        Self::fft(signal).power_spectrum()
    }

    /// Real FFT wrapper — returns only the first N/2+1 unique bins (half-spectrum).
    ///
    /// For a real-valued signal of length N, the full FFT has conjugate symmetry:
    /// X[k] = conj(X[N-k]). This method returns only bins 0..=N/2, saving half the storage.
    pub fn rfft(signal: &[f64]) -> Vec<Complex> {
        let result = Self::fft(signal);
        let n = result.n;
        let half = n / 2 + 1;
        result.spectrum.into_iter().take(half).collect()
    }

    /// Inverse real FFT — reconstructs a real signal from the half-spectrum produced by [`rfft`].
    ///
    /// `half_spectrum` should have N/2+1 bins. The full spectrum is reconstructed via
    /// conjugate symmetry, then inverse FFT is applied.
    pub fn irfft(half_spectrum: &[Complex], original_len: usize) -> Vec<f64> {
        let n = original_len.next_power_of_two();
        let mut full: Vec<Complex> = Vec::with_capacity(n);

        // Copy the half-spectrum
        for &c in half_spectrum.iter().take(n / 2 + 1) {
            full.push(c);
        }
        // Reconstruct conjugate-symmetric bins
        for k in (1..n / 2).rev() {
            full.push(half_spectrum[k].conjugate());
        }

        // Pad if needed
        while full.len() < n {
            full.push(Complex::zero());
        }

        let result = Self::ifft(&full);
        result
            .spectrum
            .iter()
            .take(original_len)
            .map(|c| c.re)
            .collect()
    }

    // ─── Core FFT Implementation ─────────────────────────────────────────

    fn fft_recursive(x: &[Complex], inverse: bool) -> Vec<Complex> {
        let n = x.len();
        if n <= 1 {
            return x.to_vec();
        }

        // Bit-reversal permutation + iterative butterfly
        let mut a = Self::bit_reverse_copy(x);

        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let angle_sign = if inverse { 1.0 } else { -1.0 };
            let angle = angle_sign * 2.0 * std::f64::consts::PI / len as f64;
            let wlen = Complex::new(angle.cos(), angle.sin());

            let mut i = 0;
            while i < n {
                let mut w = Complex::new(1.0, 0.0);
                for j in 0..half {
                    let u = a[i + j];
                    let v = a[i + j + half] * w;
                    a[i + j] = u + v;
                    a[i + j + half] = u - v;
                    w = w * wlen;
                }
                i += len;
            }
            len *= 2;
        }

        a
    }

    fn bit_reverse_copy(x: &[Complex]) -> Vec<Complex> {
        let n = x.len();
        let bits = (n as f64).log2() as u32;
        let mut result = vec![Complex::zero(); n];
        for i in 0..n {
            let rev = Self::bit_reverse(i as u32, bits) as usize;
            result[rev] = x[i];
        }
        result
    }

    fn bit_reverse(mut x: u32, bits: u32) -> u32 {
        let mut result = 0;
        for _ in 0..bits {
            result = (result << 1) | (x & 1);
            x >>= 1;
        }
        result
    }

    fn pad_to_power_of_2(mut data: Vec<Complex>) -> Vec<Complex> {
        let n = data.len().next_power_of_two();
        data.resize(n, Complex::zero());
        data
    }

    fn encode_fft(spectrum: &[Complex], forward: bool) -> BinaryHV {
        let fft_prim = BinaryHV::random(seed_from_name(if forward {
            "FFT_FORWARD"
        } else {
            "FFT_INVERSE"
        }));
        let n_hv = BinaryHV::random(seed_from_name(&format!("FFT_N_{}", spectrum.len())));
        fft_prim.bind(&n_hv)
    }
}

// ─── Bluestein Chirp-Z Transform ─────────────────────────────────────────────

/// Chirp-Z transform: computes DFT for arbitrary N via Bluestein's algorithm.
///
/// Pads to next power of 2, applies chirp modulation, radix-2 FFT, demodulates.
/// Returns the complex spectrum as `Vec<(re, im)>`.
pub fn chirp_z_dft(input: &[f64]) -> Vec<(f64, f64)> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }

    // Chirp sequence: w[k] = e^{jπk²/N}
    let chirp: Vec<Complex> = (0..n)
        .map(|k| {
            let angle = std::f64::consts::PI * (k * k) as f64 / n as f64;
            Complex::new(angle.cos(), angle.sin())
        })
        .collect();

    // Modulate input: a[k] = x[k] * conj(chirp[k])
    let a_mod: Vec<Complex> = input
        .iter()
        .enumerate()
        .map(|(k, &x)| Complex::from_real(x) * chirp[k].conjugate())
        .collect();

    // The convolution kernel h[k] = chirp[k] for 0 ≤ k < N
    // Pad both to next power of 2 ≥ 2N-1
    let pad_len = (2 * n - 1).next_power_of_two();

    let mut a_padded: Vec<Complex> = a_mod;
    a_padded.resize(pad_len, Complex::zero());

    let mut h_padded: Vec<Complex> = chirp.clone();
    h_padded.resize(pad_len, Complex::zero());
    // Add conjugate-reflected chirp at the end for linear convolution
    for k in 1..n {
        let angle = std::f64::consts::PI * (k * k) as f64 / n as f64;
        let idx = pad_len - k;
        h_padded[idx] = Complex::new(angle.cos(), angle.sin());
    }

    // FFT both, multiply, IFFT
    let fa = FftEngine::fft_recursive(&a_padded, false);
    let fh = FftEngine::fft_recursive(&h_padded, false);
    let product: Vec<Complex> = fa.iter().zip(fh.iter()).map(|(&a, &b)| a * b).collect();
    let mut conv = FftEngine::fft_recursive(&product, true);
    let scale = 1.0 / pad_len as f64;
    for c in &mut conv {
        c.re *= scale;
        c.im *= scale;
    }

    // Demodulate output: X[k] = conv[k] * conj(chirp[k])
    (0..n)
        .map(|k| {
            let c = conv[k] * chirp[k].conjugate();
            (c.re, c.im)
        })
        .collect()
}

// ─── Window Functions ─────────────────────────────────────────────────────────

/// Hann window: w[k] = 0.5 * (1 - cos(2πk/(n-1)))
pub fn window_hann(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let denom = (n - 1) as f64;
    (0..n)
        .map(|k| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * k as f64 / denom).cos()))
        .collect()
}

/// Hamming window: w[k] = 0.54 - 0.46 * cos(2πk/(n-1))
pub fn window_hamming(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let denom = (n - 1) as f64;
    (0..n)
        .map(|k| 0.54 - 0.46 * (2.0 * std::f64::consts::PI * k as f64 / denom).cos())
        .collect()
}

/// Blackman window (3-term): w[k] = 0.42 - 0.5*cos(2πk/(n-1)) + 0.08*cos(4πk/(n-1))
pub fn window_blackman(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let denom = (n - 1) as f64;
    (0..n)
        .map(|k| {
            let t = 2.0 * std::f64::consts::PI * k as f64 / denom;
            0.42 - 0.5 * t.cos() + 0.08 * (2.0 * t).cos()
        })
        .collect()
}

/// Bartlett (triangular) window: w[k] = 1 - |2k/(n-1) - 1|
pub fn window_bartlett(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let denom = (n - 1) as f64;
    (0..n)
        .map(|k| 1.0 - (2.0 * k as f64 / denom - 1.0).abs())
        .collect()
}

/// Kaiser-Bessel window using modified Bessel function I₀.
///
/// I₀(x) ≈ 1 + Σ_{k=1}^{20} (x/2)^{2k} / (k!)²
pub fn window_kaiser(n: usize, beta: f64) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let denom = (n - 1) as f64;
    let i0_beta = bessel_i0(beta);
    (0..n)
        .map(|k| {
            let arg = beta * (1.0 - (2.0 * k as f64 / denom - 1.0).powi(2)).sqrt();
            bessel_i0(arg) / i0_beta
        })
        .collect()
}

/// Modified Bessel function of the first kind, order 0.
///
/// I₀(x) ≈ 1 + Σ_{k=1}^{25} (x/2)^{2k} / (k!)²
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let x_half = x / 2.0;
    for k in 1..=25 {
        term *= x_half * x_half / (k * k) as f64;
        sum += term;
        if term < 1e-15 * sum {
            break;
        }
    }
    sum
}

/// Apply a window to a signal (element-wise multiplication).
pub fn apply_window(signal: &[f64], window: &[f64]) -> Vec<f64> {
    signal
        .iter()
        .zip(window.iter())
        .map(|(&s, &w)| s * w)
        .collect()
}

/// Energy correction factor for a window: 1 / mean(w²).
pub fn window_energy_correction(window: &[f64]) -> f64 {
    if window.is_empty() {
        return 1.0;
    }
    let mean_sq = window.iter().map(|&w| w * w).sum::<f64>() / window.len() as f64;
    if mean_sq < 1e-15 { 1.0 } else { 1.0 / mean_sq }
}

// ─── 2D FFT ──────────────────────────────────────────────────────────────────

/// 2D FFT via row-column decomposition.
///
/// Applies 1D FFT to each row, then to each column of the result.
pub fn fft_2d(input: &[Vec<f64>]) -> Vec<Vec<(f64, f64)>> {
    if input.is_empty() {
        return Vec::new();
    }
    let rows = input.len();
    let cols = input[0].len();

    // Step 1: FFT each row
    let row_transformed: Vec<Vec<Complex>> = input
        .iter()
        .map(|row| {
            let result = FftEngine::fft(row);
            result
                .spectrum
                .into_iter()
                .take(cols.next_power_of_two())
                .collect()
        })
        .collect();

    // Step 2: FFT each column of the row-transformed result
    let n_rows_padded = rows.next_power_of_two();
    let n_cols_padded = cols.next_power_of_two();

    let mut output = vec![vec![(0.0f64, 0.0f64); n_cols_padded]; n_rows_padded];

    for col in 0..n_cols_padded {
        let col_signal: Vec<Complex> = (0..row_transformed.len())
            .map(|row| {
                row_transformed[row]
                    .get(col)
                    .copied()
                    .unwrap_or(Complex::zero())
            })
            .collect();
        let col_result = FftEngine::fft_complex(&col_signal);
        for (row, c) in col_result.spectrum.iter().take(n_rows_padded).enumerate() {
            output[row][col] = (c.re, c.im);
        }
    }

    output
}

/// 2D Inverse FFT via column-row decomposition.
pub fn ifft_2d(input: &[Vec<(f64, f64)>]) -> Vec<Vec<f64>> {
    if input.is_empty() {
        return Vec::new();
    }
    let rows = input.len();
    let cols = input[0].len();

    // Step 1: IFFT each column
    let col_transformed: Vec<Vec<Complex>> = (0..cols)
        .map(|col| {
            let col_signal: Vec<Complex> = (0..rows)
                .map(|row| {
                    let (re, im) = input[row].get(col).copied().unwrap_or((0.0, 0.0));
                    Complex::new(re, im)
                })
                .collect();
            let result = FftEngine::ifft(&col_signal);
            result.spectrum
        })
        .collect();

    // Step 2: IFFT each row
    (0..rows)
        .map(|row| {
            let row_signal: Vec<Complex> = (0..cols)
                .map(|col| {
                    col_transformed[col]
                        .get(row)
                        .copied()
                        .unwrap_or(Complex::zero())
                })
                .collect();
            let result = FftEngine::ifft(&row_signal);
            result.spectrum.iter().take(cols).map(|c| c.re).collect()
        })
        .collect()
}

// ─── Spectral Utilities ───────────────────────────────────────────────────────

/// Spectral centroid: center of mass of the magnitude spectrum.
///
/// Returns frequency in Hz: Σ(|X[k]| * f[k]) / Σ|X[k]|
pub fn spectral_centroid(spectrum: &[(f64, f64)], sample_rate: f64) -> f64 {
    let n = spectrum.len();
    if n == 0 {
        return 0.0;
    }
    let freq_res = sample_rate / n as f64;
    let (weighted_sum, total) =
        spectrum
            .iter()
            .enumerate()
            .fold((0.0, 0.0), |(ws, t), (k, (re, im))| {
                let mag = (re * re + im * im).sqrt();
                (ws + mag * k as f64 * freq_res, t + mag)
            });
    if total < 1e-15 {
        0.0
    } else {
        weighted_sum / total
    }
}

/// Peak frequency: the frequency bin with highest magnitude.
pub fn peak_frequency(spectrum: &[(f64, f64)], sample_rate: f64) -> f64 {
    let n = spectrum.len();
    if n == 0 {
        return 0.0;
    }
    let freq_res = sample_rate / n as f64;
    // Only look at first half (positive frequencies)
    let half = n / 2;
    let peak_bin = spectrum[..half]
        .iter()
        .enumerate()
        .max_by(|(_, (re1, im1)), (_, (re2, im2))| {
            let m1 = re1 * re1 + im1 * im1;
            let m2 = re2 * re2 + im2 * im2;
            m1.partial_cmp(&m2).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(k, _)| k)
        .unwrap_or(0);
    peak_bin as f64 * freq_res
}

/// Frequency resolution: bin width in Hz = sample_rate / n.
pub fn frequency_resolution(n: usize, sample_rate: f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    sample_rate / n as f64
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-8;

    #[test]
    fn test_fft_constant() {
        // FFT of constant signal [1,1,1,1] → [4, 0, 0, 0]
        let result = FftEngine::fft(&[1.0, 1.0, 1.0, 1.0]);
        assert!((result.spectrum[0].re - 4.0).abs() < TOL);
        for i in 1..4 {
            assert!(result.spectrum[i].magnitude() < TOL);
        }
    }

    #[test]
    fn test_fft_impulse() {
        // FFT of [1, 0, 0, 0] → [1, 1, 1, 1]
        let result = FftEngine::fft(&[1.0, 0.0, 0.0, 0.0]);
        for c in &result.spectrum[..4] {
            assert!((c.re - 1.0).abs() < TOL);
            assert!(c.im.abs() < TOL);
        }
    }

    #[test]
    fn test_fft_ifft_roundtrip() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let spectrum = FftEngine::fft(&signal);
        let recovered = FftEngine::ifft(&spectrum.spectrum);

        for (i, (&orig, rec)) in signal.iter().zip(recovered.spectrum.iter()).enumerate() {
            assert!(
                (orig - rec.re).abs() < TOL,
                "Sample {}: {} != {}",
                i,
                orig,
                rec.re
            );
            assert!(rec.im.abs() < TOL);
        }
    }

    #[test]
    fn test_parseval_theorem() {
        // Parseval's theorem: Σ|x[n]|² = (1/N) Σ|X[k]|²
        let signal = vec![1.0, 3.0, -2.0, 4.0, 0.0, 1.0, -1.0, 2.0];
        let n = signal.len();

        let time_energy: f64 = signal.iter().map(|x| x * x).sum();
        let spectrum = FftEngine::fft(&signal);
        let freq_energy: f64 = spectrum
            .spectrum
            .iter()
            .take(n)
            .map(|c| c.magnitude_squared())
            .sum::<f64>()
            / n as f64;

        assert!(
            (time_energy - freq_energy).abs() < TOL,
            "Parseval: {} != {}",
            time_energy,
            freq_energy
        );
    }

    #[test]
    fn test_fft_sinusoid() {
        // FFT of a pure sinusoid should have peaks at the frequency
        let n = 64;
        let freq = 4.0; // 4 cycles in n samples
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / n as f64).sin())
            .collect();

        let result = FftEngine::fft(&signal);
        let magnitudes = result.magnitude_spectrum();

        // Should have peaks at index 4 and N-4
        let peak_idx = magnitudes[1..n / 2]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i + 1)
            .unwrap();

        assert_eq!(
            peak_idx, freq as usize,
            "Peak should be at frequency {}",
            freq
        );
    }

    #[test]
    fn test_convolution() {
        // Convolution of [1, 2, 3] * [1, 1] = [1, 3, 5, 3]
        let result = FftEngine::convolve(&[1.0, 2.0, 3.0], &[1.0, 1.0]);
        let expected = vec![1.0, 3.0, 5.0, 3.0];
        assert_eq!(result.len(), expected.len());
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < TOL, "{} != {}", r, e);
        }
    }

    #[test]
    fn test_convolution_impulse() {
        // Convolution with impulse = identity
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let result = FftEngine::convolve(&signal, &[1.0]);
        for (r, s) in result.iter().zip(signal.iter()) {
            assert!((r - s).abs() < TOL);
        }
    }

    #[test]
    fn test_power_spectrum() {
        let signal = vec![1.0, 0.0, 1.0, 0.0];
        let ps = FftEngine::power_spectrum(&signal);
        assert_eq!(ps.len(), 4);
        // DC component: (1+0+1+0)² = 4
        assert!((ps[0] - 4.0).abs() < TOL);
    }

    #[test]
    fn test_complex_arithmetic() {
        let a = Complex::new(3.0, 4.0);
        assert!((a.magnitude() - 5.0).abs() < TOL);
        assert!((a.magnitude_squared() - 25.0).abs() < TOL);

        let b = Complex::new(1.0, 2.0);
        let sum = a + b;
        assert!((sum.re - 4.0).abs() < TOL);
        assert!((sum.im - 6.0).abs() < TOL);

        let product = a * b;
        // (3+4i)(1+2i) = 3+6i+4i+8i² = 3-8+10i = -5+10i
        assert!((product.re - (-5.0)).abs() < TOL);
        assert!((product.im - 10.0).abs() < TOL);
    }

    #[test]
    fn test_fft_result_methods() {
        let result = FftEngine::fft(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(result.magnitude_spectrum().len(), 4);
        assert_eq!(result.phase_spectrum().len(), 4);
        assert_eq!(result.real_parts().len(), 4);
        assert!(result.is_forward);
    }

    // ── Real FFT (half-spectrum) ─────────────────────────────────────────

    #[test]
    fn test_rfft_half_spectrum_length() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let half = FftEngine::rfft(&signal);
        // N=8 → N/2+1 = 5 unique bins
        assert_eq!(half.len(), 5, "rfft should return N/2+1 bins");
    }

    #[test]
    fn test_rfft_dc_and_nyquist() {
        // DC = sum of signal, Nyquist = alternating sum
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let half = FftEngine::rfft(&signal);
        // DC: 1+2+3+4 = 10
        assert!(
            (half[0].re - 10.0).abs() < TOL,
            "DC should be 10, got {}",
            half[0].re
        );
        assert!(half[0].im.abs() < TOL);
        // Nyquist (N/2=2): 1-2+3-4 = -2
        assert!(
            (half[2].re - (-2.0)).abs() < TOL,
            "Nyquist should be -2, got {}",
            half[2].re
        );
    }

    #[test]
    fn test_irfft_roundtrip() {
        let signal = vec![1.0, 3.0, -2.0, 4.0, 0.5, 1.5, -1.0, 2.0];
        let n = signal.len();
        let half = FftEngine::rfft(&signal);
        let recovered = FftEngine::irfft(&half, n);
        assert_eq!(recovered.len(), n);
        for (i, (&orig, &rec)) in signal.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-6,
                "Sample {}: {} != {}",
                i,
                orig,
                rec
            );
        }
    }

    // ── Convolution theorem ─────────────────────────────────────────────

    #[test]
    fn test_convolution_theorem() {
        // Convolution in time = multiplication in frequency
        // Verify: FFT(a*b) ≈ FFT(a) · FFT(b) (with proper zero-padding)
        let a = vec![1.0, 2.0, 1.0, 0.0];
        let b = vec![1.0, 1.0, 0.0, 0.0];

        let conv = FftEngine::convolve(&a, &b);

        // Manual convolution for verification
        let expected = vec![1.0, 3.0, 3.0, 1.0, 0.0, 0.0, 0.0];
        for (i, (&c, &e)) in conv.iter().zip(expected.iter()).enumerate() {
            assert!((c - e).abs() < TOL, "Convolution[{}]: {} != {}", i, c, e);
        }
    }

    // ── Power spectrum of sine wave ─────────────────────────────────────

    #[test]
    fn test_power_spectrum_sine_wave() {
        // A pure sine wave should have energy concentrated at one frequency
        let n = 128;
        let freq = 8.0; // 8 cycles in 128 samples
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / n as f64).sin())
            .collect();

        let ps = FftEngine::power_spectrum(&signal);

        // Energy at bin 8 and bin N-8 should dominate
        let total_energy: f64 = ps.iter().sum();
        let peak_energy = ps[freq as usize] + ps[n - freq as usize];
        let ratio = peak_energy / total_energy;
        assert!(
            ratio > 0.99,
            "Sine wave energy should be >99% at frequency bin, got {:.4}",
            ratio
        );
    }

    // ── Complex FFT ─────────────────────────────────────────────────────

    #[test]
    fn test_fft_complex_input() {
        // Complex exponential e^{j2πk/N} should have a single peak at bin 1
        let n = 8;
        let signal: Vec<Complex> = (0..n)
            .map(|i| {
                let angle = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                Complex::new(angle.cos(), angle.sin())
            })
            .collect();

        let result = FftEngine::fft_complex(&signal);
        let magnitudes = result.magnitude_spectrum();

        // Peak should be at bin 1
        assert!(
            magnitudes[1] > magnitudes[0] * 10.0,
            "Bin 1 should dominate: bin0={}, bin1={}",
            magnitudes[0],
            magnitudes[1]
        );
    }

    // ── Edge cases ──────────────────────────────────────────────────────

    #[test]
    fn test_fft_single_sample() {
        let result = FftEngine::fft(&[42.0]);
        assert_eq!(result.spectrum.len(), 1);
        assert!((result.spectrum[0].re - 42.0).abs() < TOL);
    }

    #[test]
    fn test_convolution_commutativity() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0];
        let ab = FftEngine::convolve(&a, &b);
        let ba = FftEngine::convolve(&b, &a);
        assert_eq!(ab.len(), ba.len());
        for (x, y) in ab.iter().zip(ba.iter()) {
            assert!((x - y).abs() < TOL, "Convolution should be commutative");
        }
    }

    #[test]
    fn test_complex_conjugate() {
        let c = Complex::new(3.0, 4.0);
        let conj = c.conjugate();
        assert!((conj.re - 3.0).abs() < TOL);
        assert!((conj.im - (-4.0)).abs() < TOL);
        // c * conj(c) = |c|²
        let product = c * conj;
        assert!((product.re - 25.0).abs() < TOL);
        assert!(product.im.abs() < TOL);
    }

    // ── Bluestein chirp-Z ───────────────────────────────────────────────

    #[test]
    fn test_chirp_z_matches_fft_on_power_of_2() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let chirp = chirp_z_dft(&signal);
        let fft_result = FftEngine::fft(&signal);
        // Compare magnitudes
        for (i, (c, f)) in chirp.iter().zip(fft_result.spectrum.iter()).enumerate() {
            let chirp_mag = (c.0 * c.0 + c.1 * c.1).sqrt();
            let fft_mag = f.magnitude();
            assert!(
                (chirp_mag - fft_mag).abs() < 1e-6,
                "Chirp-Z vs FFT mismatch at bin {}: {} vs {}",
                i,
                chirp_mag,
                fft_mag
            );
        }
    }

    // ── Window functions ─────────────────────────────────────────────────

    #[test]
    fn test_hann_window_endpoints() {
        let w = window_hann(8);
        assert_eq!(w.len(), 8);
        // First and last samples should be 0
        assert!(w[0].abs() < 1e-10, "Hann window starts at 0");
        assert!(w[7].abs() < 1e-10, "Hann window ends at 0");
    }

    #[test]
    fn test_hann_window_sum_approx_n_over_2() {
        let n = 1024;
        let w = window_hann(n);
        let sum: f64 = w.iter().sum();
        // Sum of Hann window ≈ n/2
        assert!(
            (sum - n as f64 / 2.0).abs() < 2.0,
            "Hann sum {} should be near {}",
            sum,
            n as f64 / 2.0
        );
    }

    #[test]
    fn test_blackman_window_length() {
        let w = window_blackman(64);
        assert_eq!(w.len(), 64);
        // All values in [0, 1]
        for &v in &w {
            assert!(v >= -0.01 && v <= 1.01, "Blackman value {} out of range", v);
        }
    }

    #[test]
    fn test_window_apply_length() {
        let signal = vec![1.0; 32];
        let w = window_hann(32);
        let windowed = apply_window(&signal, &w);
        assert_eq!(windowed.len(), 32);
    }

    #[test]
    fn test_kaiser_window_symmetric() {
        let w = window_kaiser(16, 5.0);
        assert_eq!(w.len(), 16);
        // Kaiser is symmetric
        for i in 0..8 {
            assert!(
                (w[i] - w[15 - i]).abs() < 1e-10,
                "Kaiser not symmetric at index {}",
                i
            );
        }
    }

    // ── 2D FFT ──────────────────────────────────────────────────────────

    #[test]
    fn test_fft_2d_roundtrip() {
        let input = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0, 12.0],
            vec![13.0, 14.0, 15.0, 16.0],
        ];
        let spectrum = fft_2d(&input);
        let recovered = ifft_2d(&spectrum);
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (input[i][j] - recovered[i][j]).abs() < 1e-6,
                    "2D FFT round-trip failed at [{},{}]: {} vs {}",
                    i,
                    j,
                    input[i][j],
                    recovered[i][j]
                );
            }
        }
    }

    // ── Spectral utilities ───────────────────────────────────────────────

    #[test]
    fn test_spectral_centroid_pure_tone() {
        // A 440 Hz tone in a 1024-sample signal at 44100 Hz sample rate
        let n = 1024;
        let sample_rate = 44100.0;
        let freq = 440.0;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / sample_rate).sin())
            .collect();
        let fft = FftEngine::fft(&signal);
        let spectrum_pairs: Vec<(f64, f64)> = fft.spectrum.iter().map(|c| (c.re, c.im)).collect();
        // Peak frequency should be near 440 Hz
        let peak = peak_frequency(&spectrum_pairs, sample_rate);
        assert!(
            (peak - freq).abs() < sample_rate / n as f64 * 2.0,
            "Peak frequency {} should be near {} Hz",
            peak,
            freq
        );
    }

    #[test]
    fn test_frequency_resolution() {
        let res = frequency_resolution(1024, 44100.0);
        assert!((res - 44100.0 / 1024.0).abs() < 1e-6);
    }
}
