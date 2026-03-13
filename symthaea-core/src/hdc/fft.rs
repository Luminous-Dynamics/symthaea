#![allow(dead_code)]

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
}
