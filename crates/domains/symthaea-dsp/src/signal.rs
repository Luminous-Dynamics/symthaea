// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Convolution, simple filters, and sampling theory.

/// Full linear convolution `(a * b)`; length `a.len() + b.len() − 1`.
pub fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0.0; a.len() + b.len() - 1];
    for (i, &av) in a.iter().enumerate() {
        for (j, &bv) in b.iter().enumerate() {
            out[i + j] += av * bv;
        }
    }
    out
}

/// Windowed moving average of width `w` (each output = mean of the trailing `w`).
pub fn moving_average(signal: &[f64], w: usize) -> Vec<f64> {
    if w == 0 {
        return signal.to_vec();
    }
    (0..signal.len())
        .map(|i| {
            let start = i.saturating_sub(w - 1);
            let slice = &signal[start..=i];
            slice.iter().sum::<f64>() / slice.len() as f64
        })
        .collect()
}

/// Nyquist frequency `fs/2` — the highest representable frequency.
pub fn nyquist_frequency(sample_rate: f64) -> f64 {
    sample_rate / 2.0
}

/// Whether a signal frequency will alias at a given sample rate (`f > fs/2`).
pub fn will_alias(signal_freq: f64, sample_rate: f64) -> bool {
    signal_freq > nyquist_frequency(sample_rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convolution_of_boxes() {
        // [1,1] * [1,1] = [1,2,1] (triangle).
        assert_eq!(convolve(&[1.0, 1.0], &[1.0, 1.0]), vec![1.0, 2.0, 1.0]);
    }

    #[test]
    fn identity_convolution() {
        // Convolving with a unit impulse returns the signal (zero-padded).
        assert_eq!(convolve(&[3.0, 1.0, 4.0], &[1.0]), vec![3.0, 1.0, 4.0]);
    }

    #[test]
    fn moving_average_smooths() {
        let ma = moving_average(&[0.0, 2.0, 4.0, 6.0], 2);
        // [0, (0+2)/2, (2+4)/2, (4+6)/2] = [0,1,3,5].
        assert_eq!(ma, vec![0.0, 1.0, 3.0, 5.0]);
    }

    #[test]
    fn nyquist_and_aliasing() {
        assert!((nyquist_frequency(44_100.0) - 22_050.0).abs() < 1e-9);
        assert!(will_alias(25_000.0, 44_100.0)); // above Nyquist → aliases
        assert!(!will_alias(1_000.0, 44_100.0)); // audible tone is fine
    }
}
