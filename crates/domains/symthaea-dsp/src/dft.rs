// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Discrete Fourier Transform (naive O(n²)) and magnitude spectrum.

use std::f64::consts::PI;

/// DFT of a real signal → complex bins `(re, im)`.
/// `X[k] = Σ x[n]·e^(−2πi·kn/N)`.
pub fn dft(signal: &[f64]) -> Vec<(f64, f64)> {
    let n = signal.len();
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let (mut re, mut im) = (0.0, 0.0);
        for (nn, &x) in signal.iter().enumerate() {
            let angle = -2.0 * PI * (k * nn) as f64 / n as f64;
            re += x * angle.cos();
            im += x * angle.sin();
        }
        out.push((re, im));
    }
    out
}

/// Magnitude spectrum `|X[k]|` from a DFT.
pub fn magnitude(spectrum: &[(f64, f64)]) -> Vec<f64> {
    spectrum
        .iter()
        .map(|(re, im)| (re * re + im * im).sqrt())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_signal_has_only_dc_bin() {
        // [1,1,1,1] → X[0]=4, rest ≈ 0.
        let x = dft(&[1.0, 1.0, 1.0, 1.0]);
        assert!((x[0].0 - 4.0).abs() < 1e-9 && x[0].1.abs() < 1e-9);
        for bin in &x[1..] {
            assert!(bin.0.abs() < 1e-9 && bin.1.abs() < 1e-9);
        }
    }

    #[test]
    fn impulse_is_flat_spectrum() {
        // δ[n] → all bins magnitude 1.
        let mag = magnitude(&dft(&[1.0, 0.0, 0.0, 0.0]));
        for m in mag {
            assert!((m - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn single_cosine_peaks_at_its_bin() {
        // cos(2π·1·n/8), N=8 → energy at k=1 and k=7 (conjugate).
        let n = 8;
        let sig: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * i as f64 / n as f64).cos())
            .collect();
        let mag = magnitude(&dft(&sig));
        assert!(mag[1] > 3.0 && mag[7] > 3.0);
        assert!(mag[3] < 1e-6);
    }
}
