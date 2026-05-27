// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Welch power spectral density estimation

use rustfft::{FftPlanner, num_complex::Complex};
use std::f32::consts::PI;

/// Compute power spectral density using Welch's method
///
/// # Arguments
/// * `signal` - Input signal samples
/// * `sample_rate` - Sampling frequency in Hz
/// * `nperseg` - Samples per segment
///
/// # Returns
/// Tuple of (frequencies, PSD values)
pub fn welch_psd(signal: &[f32], sample_rate: f32, nperseg: usize) -> (Vec<f32>, Vec<f32>) {
    let nperseg = nperseg.min(signal.len()).max(8);
    let noverlap = nperseg / 2;
    let step = nperseg - noverlap;

    // Create Hanning window
    let window: Vec<f32> = (0..nperseg)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / nperseg as f32).cos()))
        .collect();

    // Window normalization factor
    let window_sum: f32 = window.iter().map(|w| w * w).sum();

    // FFT setup
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(nperseg);

    // Accumulate PSD
    let nfreqs = nperseg / 2 + 1;
    let mut psd = vec![0.0f32; nfreqs];
    let mut n_segments = 0;

    let mut pos = 0;
    while pos + nperseg <= signal.len() {
        // Apply window
        let mut buffer: Vec<Complex<f32>> = signal[pos..pos + nperseg]
            .iter()
            .zip(window.iter())
            .map(|(&s, &w)| Complex::new(s * w, 0.0))
            .collect();

        // Compute FFT
        fft.process(&mut buffer);

        // Accumulate power
        for (i, psd_val) in psd.iter_mut().enumerate() {
            *psd_val += buffer[i].norm_sqr();
        }

        n_segments += 1;
        pos += step;
    }

    if n_segments == 0 {
        return (vec![0.0; nfreqs], vec![0.0; nfreqs]);
    }

    // Normalize
    let scale = 2.0 / (sample_rate * window_sum * n_segments as f32);
    for p in psd.iter_mut() {
        *p *= scale;
    }

    // DC and Nyquist should not be doubled
    psd[0] /= 2.0;
    if nperseg % 2 == 0 {
        psd[nfreqs - 1] /= 2.0;
    }

    // Frequency bins
    let frequencies: Vec<f32> = (0..nfreqs)
        .map(|i| i as f32 * sample_rate / nperseg as f32)
        .collect();

    (frequencies, psd)
}

/// Compute band power from PSD using trapezoidal integration
///
/// # Arguments
/// * `frequencies` - Frequency bins from welch_psd
/// * `psd` - Power spectral density values
/// * `fmin` - Lower frequency bound (Hz)
/// * `fmax` - Upper frequency bound (Hz)
pub fn band_power(frequencies: &[f32], psd: &[f32], fmin: f32, fmax: f32) -> f32 {
    let mut power = 0.0;
    let mut prev_f = None;
    let mut prev_p = None;

    for (&f, &p) in frequencies.iter().zip(psd.iter()) {
        if f >= fmin && f <= fmax {
            if let (Some(pf), Some(pp)) = (prev_f, prev_p) {
                // Trapezoidal integration
                let df = f - pf;
                power += 0.5 * (p + pp) * df;
            }
            prev_f = Some(f);
            prev_p = Some(p);
        }
    }

    power
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_welch_psd() {
        // Generate 10 Hz sine wave
        let sample_rate = 256.0;
        let duration = 4.0;
        let n_samples = (sample_rate * duration) as usize;

        let signal: Vec<f32> = (0..n_samples)
            .map(|i| {
                let t = i as f32 / sample_rate;
                (2.0 * PI * 10.0 * t).sin()
            })
            .collect();

        let (freqs, psd) = welch_psd(&signal, sample_rate, 256);

        // Find peak frequency
        let peak_idx = psd
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
            .unwrap();

        let peak_freq = freqs[peak_idx];

        // Should be close to 10 Hz
        assert!((peak_freq - 10.0).abs() < 2.0);
    }

    #[test]
    fn test_band_power() {
        let freqs: Vec<f32> = (0..=30).map(|i| i as f32).collect();
        let psd: Vec<f32> = freqs
            .iter()
            .map(|&f| {
                // Peak at 10 Hz (alpha)
                (-(f - 10.0).powi(2) / 4.0).exp()
            })
            .collect();

        let alpha_power = band_power(&freqs, &psd, 8.0, 13.0);
        let delta_power = band_power(&freqs, &psd, 0.5, 4.0);

        // Alpha should have more power
        assert!(alpha_power > delta_power);
    }
}