// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Golden ratio and Fibonacci spiral detection.
//!
//! The golden ratio φ ≈ 1.618 appears in aesthetically pleasing compositions.
//! Livio (2002) documented its prevalence in art, architecture, and nature.
//! This module scores how closely an artifact's proportions approach φ.

/// The golden ratio φ = (1 + √5) / 2.
pub const PHI: f32 = 1.618_033_9;

/// Inverse golden ratio 1/φ ≈ 0.618.
pub const INV_PHI: f32 = 0.618_033_9;

/// Score how closely a ratio approaches the golden ratio.
///
/// Returns 1.0 for a perfect golden ratio, decaying to 0.0 as the ratio
/// diverges. Uses a Gaussian kernel centered on φ.
///
/// # Arguments
/// * `ratio` — The ratio to evaluate (must be > 0). Automatically considers
///   both the ratio and its reciprocal (since φ and 1/φ are both golden).
pub fn golden_ratio_score(ratio: f32) -> f32 {
    if ratio <= 0.0 || !ratio.is_finite() {
        return 0.0;
    }

    // Consider both ratio and its reciprocal
    let r1 = ratio;
    let r2 = 1.0 / ratio;

    let score1 = phi_kernel(r1);
    let score2 = phi_kernel(r2);

    score1.max(score2)
}

/// Gaussian kernel centered on φ with σ = 0.3.
fn phi_kernel(ratio: f32) -> f32 {
    let sigma = 0.3;
    let deviation = ratio - PHI;
    (-deviation * deviation / (2.0 * sigma * sigma)).exp()
}

/// Score a set of proportions for golden ratio presence.
///
/// Given a list of measurements (e.g., segment lengths in a composition),
/// computes the average golden ratio score across all consecutive pairs.
///
/// Returns 0.0 if fewer than 2 measurements.
pub fn golden_proportions_score(measurements: &[f32]) -> f32 {
    if measurements.len() < 2 {
        return 0.0;
    }

    let mut total_score = 0.0;
    let mut count = 0;

    for window in measurements.windows(2) {
        if window[0] > 0.0 && window[1] > 0.0 {
            total_score += golden_ratio_score(window[0] / window[1]);
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        total_score / count as f32
    }
}

/// Detect Fibonacci-like progression in a sequence.
///
/// Measures how closely consecutive ratios approach φ (since Fibonacci
/// ratios converge to the golden ratio). Returns average score.
pub fn fibonacci_progression_score(values: &[f32]) -> f32 {
    if values.len() < 3 {
        return 0.0;
    }

    let mut total = 0.0;
    let mut count = 0;

    for window in values.windows(3) {
        let a = window[0];
        let b = window[1];
        let c = window[2];

        if a > 0.0 && b > 0.0 {
            // In true Fibonacci: c ≈ a + b, and b/a ≈ φ
            let ratio = b / a;
            let sum_score = if (a + b) > 0.0 {
                // How close is c to a + b?
                let expected = a + b;
                let deviation = ((c - expected) / expected).abs();
                (-deviation * deviation / 0.18).exp() // σ² = 0.09
            } else {
                0.0
            };

            total += 0.5 * golden_ratio_score(ratio) + 0.5 * sum_score;
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        total / count as f32
    }
}

/// Score a rectangular composition for golden ratio divisions.
///
/// Given a viewport (width, height) and a list of division points along
/// one axis, scores how golden the divisions are.
pub fn golden_division_score(total_length: f32, divisions: &[f32]) -> f32 {
    if divisions.is_empty() || total_length <= 0.0 {
        return 0.0;
    }

    let mut scores = Vec::with_capacity(divisions.len());

    for &d in divisions {
        if d > 0.0 && d < total_length {
            let ratio = d / (total_length - d);
            scores.push(golden_ratio_score(ratio));
        }
    }

    if scores.is_empty() {
        0.0
    } else {
        scores.iter().sum::<f32>() / scores.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_golden_ratio() {
        let score = golden_ratio_score(PHI);
        assert!((score - 1.0).abs() < 0.01, "perfect ratio score = {score}");
    }

    #[test]
    fn inverse_golden_also_perfect() {
        let score = golden_ratio_score(INV_PHI);
        assert!((score - 1.0).abs() < 0.01, "inverse ratio score = {score}");
    }

    #[test]
    fn non_golden_lower() {
        let golden = golden_ratio_score(PHI);
        let square = golden_ratio_score(1.0); // 1:1 ratio
        let extreme = golden_ratio_score(5.0); // 5:1 ratio

        assert!(golden > square, "golden {golden} > square {square}");
        assert!(golden > extreme, "golden {golden} > extreme {extreme}");
    }

    #[test]
    fn zero_and_negative_safe() {
        assert_eq!(golden_ratio_score(0.0), 0.0);
        assert_eq!(golden_ratio_score(-1.0), 0.0);
        assert_eq!(golden_ratio_score(f32::NAN), 0.0);
        assert_eq!(golden_ratio_score(f32::INFINITY), 0.0);
    }

    #[test]
    fn proportions_fibonacci_sequence() {
        // Fibonacci: 1, 1, 2, 3, 5, 8, 13
        let fib = [1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0];
        let score = golden_proportions_score(&fib);
        assert!(score > 0.5, "Fibonacci proportions = {score}");
    }

    #[test]
    fn proportions_too_few() {
        assert_eq!(golden_proportions_score(&[1.0]), 0.0);
        assert_eq!(golden_proportions_score(&[]), 0.0);
    }

    #[test]
    fn fibonacci_progression() {
        let fib = [1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0];
        let score = fibonacci_progression_score(&fib);
        assert!(score > 0.7, "Fibonacci progression = {score}");
    }

    #[test]
    fn golden_division() {
        // 100px total, divided at 61.8px (golden cut)
        let score = golden_division_score(100.0, &[61.8]);
        assert!(score > 0.9, "golden division = {score}");
    }

    #[test]
    fn non_golden_division_lower() {
        let golden = golden_division_score(100.0, &[61.8]);
        let half = golden_division_score(100.0, &[50.0]);
        assert!(golden > half, "golden {golden} > half {half}");
    }

    #[test]
    fn score_bounded() {
        for ratio in [0.5, 1.0, 1.5, PHI, 2.0, 3.0, 10.0] {
            let s = golden_ratio_score(ratio);
            assert!(s >= 0.0 && s <= 1.0, "ratio {ratio} score = {s}");
        }
    }
}
