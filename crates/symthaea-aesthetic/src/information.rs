// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Information-theoretic aesthetic measures.
//!
//! Shannon entropy quantifies the "information content" of an artifact's
//! element distribution. Moderate entropy is aesthetically pleasing
//! (Berlyne 1971): too low = boring, too high = chaotic.

/// Compute Shannon entropy of a probability distribution.
///
/// Input: slice of non-negative values (will be normalized to sum to 1).
/// Returns entropy in bits, normalized to [0, 1] where 1.0 = maximum entropy
/// (uniform distribution over n elements).
///
/// # Panics
///
/// Does not panic. Returns 0.0 for empty or all-zero distributions.
pub fn shannon_entropy_normalized(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }

    let sum: f32 = values.iter().filter(|v| **v > 0.0).sum();
    if sum <= 0.0 {
        return 0.0;
    }

    let n = values.len();
    let max_entropy = (n as f32).ln();
    if max_entropy <= 0.0 {
        return 0.0; // Single element
    }

    let entropy: f32 = values
        .iter()
        .filter(|&&v| v > 0.0)
        .map(|&v| {
            let p = v / sum;
            -p * p.ln()
        })
        .sum();

    (entropy / max_entropy).clamp(0.0, 1.0)
}

/// Compute the information balance score.
///
/// Based on Berlyne's (1971) arousal potential theory:
/// aesthetic pleasure peaks at moderate complexity/information.
///
/// Returns a value in [0, 1] where 1.0 = optimal information balance
/// (entropy near the sweet spot of ~0.6 of maximum).
pub fn information_balance(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let entropy = shannon_entropy_normalized(values);
    // Berlyne curve: Gaussian centered at 0.6 with σ=0.25
    let optimal = 0.6;
    let sigma = 0.25;
    let deviation = entropy - optimal;
    (-deviation * deviation / (2.0 * sigma * sigma)).exp()
}

/// Compute element diversity: number of distinct non-zero categories
/// divided by total categories.
///
/// Higher diversity = more variety in the artifact.
pub fn element_diversity(values: &[f32], threshold: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let distinct = values.iter().filter(|&&v| v > threshold).count();
    (distinct as f32) / (values.len() as f32)
}

/// Compute the repetition index: how much the distribution is dominated
/// by a single element.
///
/// Returns a value in [0, 1] where 0.0 = no dominant element (uniform)
/// and 1.0 = completely dominated by one element.
pub fn repetition_index(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let sum: f32 = values.iter().sum();
    if sum <= 0.0 {
        return 0.0;
    }
    let max = values.iter().cloned().fold(0.0f32, f32::max);
    let dominance = max / sum;
    // With n elements, uniform dominance = 1/n. Scale to [0, 1].
    let n = values.len() as f32;
    let uniform_dominance = 1.0 / n;
    ((dominance - uniform_dominance) / (1.0 - uniform_dominance)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_max_entropy() {
        let uniform = vec![1.0; 8];
        let e = shannon_entropy_normalized(&uniform);
        assert!((e - 1.0).abs() < 0.01, "uniform entropy = {e}, want ~1.0");
    }

    #[test]
    fn single_element_zero_entropy() {
        let single = vec![1.0, 0.0, 0.0, 0.0];
        let e = shannon_entropy_normalized(&single);
        assert!(e < 0.01, "single element entropy = {e}, want ~0.0");
    }

    #[test]
    fn empty_zero() {
        assert_eq!(shannon_entropy_normalized(&[]), 0.0);
        assert_eq!(information_balance(&[]), 0.0);
        assert_eq!(element_diversity(&[], 0.0), 0.0);
        assert_eq!(repetition_index(&[]), 0.0);
    }

    #[test]
    fn information_balance_peaks_at_moderate() {
        // Berlyne: moderate complexity most pleasing.
        // With 8 elements, entropy ~0.6 of max is the sweet spot.
        // Strongly peaked (low entropy) should score lowest.
        let peaked = vec![1.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]; // low entropy
        let moderate = vec![0.30, 0.20, 0.15, 0.12, 0.10, 0.06, 0.04, 0.03]; // ~0.6 entropy
        let uniform = vec![1.0; 8]; // max entropy

        let b_peak = information_balance(&peaked);
        let b_mod = information_balance(&moderate);
        let b_uni = information_balance(&uniform);

        // Moderate should beat both extremes
        assert!(b_mod > b_peak, "moderate {b_mod} > peaked {b_peak}");
        assert!(b_mod > b_uni, "moderate {b_mod} > uniform {b_uni}");
        assert!(b_mod > 0.3, "moderate balance = {b_mod}");
    }

    #[test]
    fn element_diversity_works() {
        let values = [0.8, 0.6, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0];
        let d = element_diversity(&values, 0.01);
        assert!((d - 3.0 / 8.0).abs() < 0.01);
    }

    #[test]
    fn repetition_index_uniform_zero() {
        let uniform = vec![1.0; 8];
        let r = repetition_index(&uniform);
        assert!(r < 0.01, "uniform repetition = {r}");
    }

    #[test]
    fn repetition_index_dominated_high() {
        let dominated = vec![100.0, 0.1, 0.1, 0.1];
        let r = repetition_index(&dominated);
        assert!(r > 0.9, "dominated repetition = {r}");
    }

    #[test]
    fn entropy_bounded() {
        for vals in [
            vec![0.5, 0.3, 0.2],
            vec![1.0],
            vec![0.0, 0.0, 1.0],
            vec![0.25; 4],
        ] {
            let e = shannon_entropy_normalized(&vals);
            assert!(
                e >= 0.0 && e <= 1.0,
                "entropy {e} out of bounds for {vals:?}"
            );
        }
    }
}
