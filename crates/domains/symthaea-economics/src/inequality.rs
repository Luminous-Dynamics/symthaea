// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Income-inequality metrics.

/// Gini coefficient of a set of non-negative values, in `[0, 1]`.
///
/// `G = (2·Σ i·xᵢ − (n+1)·Σ xᵢ) / (n·Σ xᵢ)` with `xᵢ` sorted ascending and `i`
/// 1-indexed. 0 = perfect equality; approaches `(n−1)/n` as all income
/// concentrates in one holder.
pub fn gini(values: &[f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    let mut v: Vec<f64> = values.iter().copied().filter(|x| *x >= 0.0).collect();
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    let sum: f64 = v.iter().sum();
    if sum <= 0.0 {
        return 0.0;
    }
    let weighted: f64 = v
        .iter()
        .enumerate()
        .map(|(i, x)| (i as f64 + 1.0) * x)
        .sum();
    (2.0 * weighted - (n as f64 + 1.0) * sum) / (n as f64 * sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_equality_is_zero() {
        assert!(gini(&[1.0, 1.0, 1.0, 1.0]).abs() < 1e-12);
    }

    #[test]
    fn total_concentration_approaches_max() {
        // n=4 → max is (n-1)/n = 0.75.
        assert!((gini(&[0.0, 0.0, 0.0, 1.0]) - 0.75).abs() < 1e-12);
    }

    #[test]
    fn known_distribution() {
        // [1,2,3,4,5] → Gini = 0.2667.
        assert!((gini(&[1.0, 2.0, 3.0, 4.0, 5.0]) - 0.266667).abs() < 1e-5);
    }

    #[test]
    fn order_independent() {
        assert!(
            (gini(&[5.0, 1.0, 3.0, 2.0, 4.0]) - gini(&[1.0, 2.0, 3.0, 4.0, 5.0])).abs() < 1e-12
        );
    }
}
