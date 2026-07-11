// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! The Fast Fourier Transform (radix-2 Cooley-Tukey) over [`Complex`].
//!
//! `symthaea-dsp` computes a naive O(n²) DFT over `(re, im)` tuples; this is the
//! O(n log n) transform with a real complex type. Lengths must be powers of two.

use crate::complex::Complex;
use std::f64::consts::PI;

/// Is `n` a power of two (and non-zero)?
fn is_pow2(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

/// In-place iterative radix-2 FFT. `sign = -1.0` is the forward transform,
/// `+1.0` the inverse (unnormalised). `None` if the length is not a power of 2.
fn transform(data: &mut [Complex], sign: f64) -> Option<()> {
    let n = data.len();
    if !is_pow2(n) {
        return None;
    }
    if n == 1 {
        return Some(());
    }
    // Bit-reversal permutation.
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            data.swap(i, j);
        }
    }
    // Butterfly stages.
    let mut len = 2;
    while len <= n {
        let ang = sign * 2.0 * PI / len as f64;
        let wlen = Complex::from_polar(1.0, ang);
        let half = len / 2;
        let mut i = 0;
        while i < n {
            let mut w = Complex::real(1.0);
            for k in 0..half {
                let u = data[i + k];
                let v = data[i + k + half] * w;
                data[i + k] = u + v;
                data[i + k + half] = u - v;
                w = w * wlen;
            }
            i += len;
        }
        len <<= 1;
    }
    Some(())
}

/// Forward FFT of a complex signal (length must be a power of two).
pub fn fft(input: &[Complex]) -> Option<Vec<Complex>> {
    let mut data = input.to_vec();
    transform(&mut data, -1.0)?;
    Some(data)
}

/// Inverse FFT (normalised by `1/n`).
pub fn ifft(input: &[Complex]) -> Option<Vec<Complex>> {
    let mut data = input.to_vec();
    transform(&mut data, 1.0)?;
    let n = data.len() as f64;
    for z in data.iter_mut() {
        *z = *z * (1.0 / n);
    }
    Some(data)
}

/// Forward FFT of a real signal (convenience wrapper).
pub fn fft_real(input: &[f64]) -> Option<Vec<Complex>> {
    let data: Vec<Complex> = input.iter().map(|&x| Complex::real(x)).collect();
    fft(&data)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &[Complex], b: &[Complex]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!(x.approx_eq(*y, 1e-9), "{x:?} vs {y:?}");
        }
    }

    #[test]
    fn constant_signal_has_only_dc() {
        // FFT of [1,1,1,1] = [4, 0, 0, 0] (all energy in the DC bin).
        let out = fft_real(&[1.0, 1.0, 1.0, 1.0]).unwrap();
        assert!(out[0].approx_eq(Complex::real(4.0), 1e-9));
        for z in &out[1..] {
            assert!(z.approx_eq(Complex::real(0.0), 1e-9), "{z:?}");
        }
    }

    #[test]
    fn single_frequency_lands_in_one_bin() {
        // A pure cosine at bin 1 over 8 samples → energy at bins 1 and 7.
        let n = 8;
        let sig: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * i as f64 / n as f64).cos())
            .collect();
        let out = fft_real(&sig).unwrap();
        assert!((out[1].modulus() - 4.0).abs() < 1e-9, "{:?}", out[1]);
        assert!((out[7].modulus() - 4.0).abs() < 1e-9);
        assert!(out[2].modulus() < 1e-9 && out[3].modulus() < 1e-9);
    }

    #[test]
    fn ifft_inverts_fft() {
        let sig: Vec<Complex> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            .iter()
            .map(|&x| Complex::real(x))
            .collect();
        let round_trip = ifft(&fft(&sig).unwrap()).unwrap();
        close(&round_trip, &sig);
    }

    #[test]
    fn non_power_of_two_is_none() {
        assert!(fft_real(&[1.0, 2.0, 3.0]).is_none());
    }
}
