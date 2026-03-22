// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Permutation Entropy - Time-Domain Complexity Measure
//!
//! Measures the "randomness" of ordinal patterns in a signal.
//! This is the DIFFERENTIATION component of the consciousness equation:
//!
//!   Φ = Integration (Synchrony) × Differentiation (Entropy)
//!
//! Key properties:
//! - Purely time-domain (no FFT required)
//! - Normalized to [0, 1]
//! - Robust to amplitude scaling
//! - Captures the "unpredictability" of neural dynamics
//!
//! Reference: Bandt & Pompe (2002) "Permutation Entropy: A Natural
//! Complexity Measure for Time Series"

/// Calculates Permutation Entropy of order 3 with configurable delay.
///
/// # Returns
/// - 0.0 = Perfectly predictable (monotonic signal, like deep sleep delta)
/// - 1.0 = Maximum entropy (random noise, like awake cortical activity)
///
/// # Arguments
/// - `signal`: The input signal (EEG samples)
/// - `delay`: Embedding delay (typically 1 for 100Hz, higher for faster sampling)
///
/// # Example
/// ```
/// use symthaea::perception::physio::entropy::permutation_entropy_order3;
///
/// let monotonic = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let pe = permutation_entropy_order3(&monotonic, 1);
/// assert!(pe < 0.1); // Very predictable (ascending pattern)
///
/// // More varied signal with different ordinal patterns
/// let varied = vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0, 1.0, 9.0, 2.0, 8.0];
/// let pe = permutation_entropy_order3(&varied, 1);
/// assert!(pe > 0.5); // Higher entropy than monotonic
/// ```
pub fn permutation_entropy_order3(signal: &[f64], delay: usize) -> f64 {
    let n = signal.len();
    if n < 3 * delay {
        return 0.5; // Not enough data, return neutral
    }

    // Order 3 has 3! = 6 possible permutation patterns:
    // [0,1,2] = ascending (x1 < x2 < x3)
    // [0,2,1] = x1 < x3 < x2
    // [1,0,2] = x2 < x1 < x3
    // [1,2,0] = x3 < x1 < x2
    // [2,0,1] = x2 < x3 < x1
    // [2,1,0] = descending (x3 < x2 < x1)
    let mut counts = [0usize; 6];
    let mut total = 0usize;

    for i in 0..n.saturating_sub(2 * delay) {
        let x1 = signal[i];
        let x2 = signal[i + delay];
        let x3 = signal[i + 2 * delay];

        // Map the ordinal pattern to index 0-5
        // Using comparison-based encoding for efficiency
        let pattern_idx = ordinal_pattern_index(x1, x2, x3);

        counts[pattern_idx] += 1;
        total += 1;
    }

    if total == 0 {
        return 0.5;
    }

    // Calculate Shannon entropy of the pattern distribution
    let mut entropy = 0.0;
    let total_f = total as f64;

    for &count in counts.iter() {
        if count > 0 {
            let p = count as f64 / total_f;
            entropy -= p * p.ln(); // Natural log for consistency
        }
    }

    // Normalize by maximum possible entropy: ln(6) ≈ 1.7918
    // This gives us a value in [0, 1]
    let max_entropy = 6.0_f64.ln();
    (entropy / max_entropy).clamp(0.0, 1.0)
}

/// Determine the ordinal pattern index for three values.
/// Returns 0-5 corresponding to the 6 possible orderings.
#[inline]
fn ordinal_pattern_index(x1: f64, x2: f64, x3: f64) -> usize {
    // Handle ties by treating them as "less than" (stable ordering)
    if x1 <= x2 {
        if x2 <= x3 {
            0 // x1 <= x2 <= x3 (ascending)
        } else if x1 <= x3 {
            1 // x1 <= x3 < x2
        } else {
            2 // x3 < x1 <= x2
        }
    } else {
        // x2 < x1
        if x1 <= x3 {
            3 // x2 < x1 <= x3
        } else if x2 <= x3 {
            4 // x2 <= x3 < x1
        } else {
            5 // x3 < x2 < x1 (descending)
        }
    }
}

/// Calculates Permutation Entropy of order 4 for finer discrimination.
///
/// Order 4 has 4! = 24 possible patterns, providing more resolution
/// but requiring longer signals for statistical stability.
///
/// Use this for longer epochs (>10 seconds at 100Hz).
pub fn permutation_entropy_order4(signal: &[f64], delay: usize) -> f64 {
    let n = signal.len();
    if n < 4 * delay {
        return 0.5;
    }

    // Order 4 has 4! = 24 possible permutation patterns
    let mut counts = [0usize; 24];
    let mut total = 0usize;

    for i in 0..n.saturating_sub(3 * delay) {
        let x1 = signal[i];
        let x2 = signal[i + delay];
        let x3 = signal[i + 2 * delay];
        let x4 = signal[i + 3 * delay];

        let pattern_idx = ordinal_pattern_index_4(x1, x2, x3, x4);
        counts[pattern_idx] += 1;
        total += 1;
    }

    if total == 0 {
        return 0.5;
    }

    let mut entropy = 0.0;
    let total_f = total as f64;

    for &count in counts.iter() {
        if count > 0 {
            let p = count as f64 / total_f;
            entropy -= p * p.ln();
        }
    }

    // Normalize by ln(24) ≈ 3.178
    let max_entropy = 24.0_f64.ln();
    (entropy / max_entropy).clamp(0.0, 1.0)
}

/// Compute order-4 ordinal pattern index (0-23)
fn ordinal_pattern_index_4(x1: f64, x2: f64, x3: f64, x4: f64) -> usize {
    // Create array and sort to get ranking
    let mut vals = [(x1, 0), (x2, 1), (x3, 2), (x4, 3)];
    vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Convert ranking to factorial number system
    let mut ranks = [0usize; 4];
    for (rank, &(_, orig_idx)) in vals.iter().enumerate() {
        ranks[orig_idx] = rank;
    }

    // Encode as index: r0 * 6 + r1_adj * 2 + r2_adj
    // Using Lehmer code
    let mut index = 0;
    let mut available = [true; 4];

    for i in 0..4 {
        let mut count = 0;
        for j in 0..ranks[i] {
            if available[j] {
                count += 1;
            }
        }
        index = index * (4 - i) + count;
        available[ranks[i]] = false;
    }

    index.min(23)
}

/// Weighted Permutation Entropy - accounts for amplitude differences
///
/// Standard PE only considers ordinal patterns. Weighted PE also
/// considers the magnitude of differences, giving more weight to
/// patterns with larger amplitude variations.
///
/// This is useful for EEG where both pattern AND amplitude matter.
pub fn weighted_permutation_entropy(signal: &[f64], delay: usize) -> f64 {
    let n = signal.len();
    if n < 3 * delay {
        return 0.5;
    }

    let mut weighted_counts = [0.0f64; 6];
    let mut total_weight = 0.0;

    for i in 0..n.saturating_sub(2 * delay) {
        let x1 = signal[i];
        let x2 = signal[i + delay];
        let x3 = signal[i + 2 * delay];

        let pattern_idx = ordinal_pattern_index(x1, x2, x3);

        // Weight by variance of the triplet
        let mean = (x1 + x2 + x3) / 3.0;
        let var = ((x1 - mean).powi(2) + (x2 - mean).powi(2) + (x3 - mean).powi(2)) / 3.0;
        let weight = var.sqrt().max(1e-10); // Avoid zero weights

        weighted_counts[pattern_idx] += weight;
        total_weight += weight;
    }

    if total_weight < 1e-10 {
        return 0.5;
    }

    let mut entropy = 0.0;
    for &wcount in weighted_counts.iter() {
        if wcount > 1e-10 {
            let p = wcount / total_weight;
            entropy -= p * p.ln();
        }
    }

    let max_entropy = 6.0_f64.ln();
    (entropy / max_entropy).clamp(0.0, 1.0)
}

/// Multi-scale Permutation Entropy
///
/// Calculates PE at multiple time scales and returns the mean.
/// This captures complexity across different temporal resolutions.
///
/// Useful for distinguishing states that look similar at one scale
/// but differ at others (e.g., N2 spindles vs REM sawtooth waves).
pub fn multiscale_permutation_entropy(signal: &[f64], max_scale: usize) -> f64 {
    if signal.len() < 10 * max_scale {
        return permutation_entropy_order3(signal, 1);
    }

    let mut total_pe = 0.0;
    let mut count = 0;

    for scale in 1..=max_scale {
        // Coarse-grain the signal at this scale
        let coarse: Vec<f64> = signal
            .chunks(scale)
            .map(|chunk| chunk.iter().sum::<f64>() / chunk.len() as f64)
            .collect();

        if coarse.len() >= 6 {
            total_pe += permutation_entropy_order3(&coarse, 1);
            count += 1;
        }
    }

    if count == 0 {
        0.5
    } else {
        total_pe / count as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monotonic_low_entropy() {
        // Ascending sequence = perfectly predictable = low entropy
        let ascending: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let pe = permutation_entropy_order3(&ascending, 1);
        assert!(
            pe < 0.1,
            "Monotonic signal should have near-zero entropy, got {}",
            pe
        );
    }

    #[test]
    fn test_random_high_entropy() {
        // Pseudo-random sequence = high entropy
        let random: Vec<f64> = vec![
            0.5, 0.2, 0.8, 0.1, 0.9, 0.3, 0.7, 0.4, 0.6, 0.25, 0.75, 0.15, 0.85, 0.35, 0.65, 0.45,
            0.55, 0.28, 0.72, 0.18, 0.82, 0.38, 0.62, 0.48, 0.52, 0.22, 0.78, 0.12, 0.88, 0.32,
        ];
        let pe = permutation_entropy_order3(&random, 1);
        assert!(
            pe > 0.7,
            "Random signal should have high entropy, got {}",
            pe
        );
    }

    #[test]
    fn test_sine_wave_medium_entropy() {
        // Sine wave = structured but not monotonic = medium entropy
        use std::f64::consts::PI;
        let sine: Vec<f64> = (0..200)
            .map(|x| (2.0 * PI * x as f64 / 20.0).sin())
            .collect();
        let pe = permutation_entropy_order3(&sine, 1);
        assert!(
            pe > 0.3 && pe < 0.8,
            "Sine wave should have medium entropy, got {}",
            pe
        );
    }

    #[test]
    fn test_normalized_range() {
        let signal: Vec<f64> = (0..50).map(|x| (x as f64).sin()).collect();
        let pe = permutation_entropy_order3(&signal, 1);
        assert!(
            (0.0..=1.0).contains(&pe),
            "PE should be in [0,1], got {}",
            pe
        );
    }
}
