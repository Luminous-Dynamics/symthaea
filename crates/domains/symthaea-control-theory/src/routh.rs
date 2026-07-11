// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Routh-Hurwitz stability: count right-half-plane roots of a characteristic
//! polynomial from its coefficients, without finding the roots themselves.

/// Number of right-half-plane roots of a polynomial given its coefficients
/// (highest degree first), via sign changes in the first Routh column.
///
/// A control system is stable iff this is 0. A zero in the first column is
/// nudged to a tiny epsilon (the standard degenerate-case handling).
pub fn rhp_root_count(coeffs: &[f64]) -> usize {
    let n = coeffs.len();
    if n < 2 {
        return 0;
    }
    let rows = n; // degree + 1
    let width = (n + 1) / 2;
    let mut table = vec![vec![0.0f64; width]; rows];

    for (k, &c) in coeffs.iter().enumerate() {
        table[k % 2][k / 2] = c;
    }
    for i in 2..rows {
        let mut a = table[i - 1][0];
        if a == 0.0 {
            a = 1e-12;
        }
        for j in 0..width.saturating_sub(1) {
            table[i][j] = (a * table[i - 2][j + 1] - table[i - 2][0] * table[i - 1][j + 1]) / a;
        }
    }

    let mut changes = 0;
    let mut prev = if table[0][0] == 0.0 {
        1e-12
    } else {
        table[0][0]
    };
    for row in table.iter().skip(1) {
        let cur = if row[0] == 0.0 { 1e-12 } else { row[0] };
        if prev.signum() != cur.signum() {
            changes += 1;
        }
        prev = cur;
    }
    changes
}

/// Whether the system is stable (no right-half-plane roots).
pub fn is_stable(coeffs: &[f64]) -> bool {
    rhp_root_count(coeffs) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_second_order() {
        // s² + 3s + 2 = (s+1)(s+2), both LHP.
        assert!(is_stable(&[1.0, 3.0, 2.0]));
        assert_eq!(rhp_root_count(&[1.0, 3.0, 2.0]), 0);
    }

    #[test]
    fn unstable_second_order() {
        // s² − 3s + 2 = (s−1)(s−2), both RHP.
        assert_eq!(rhp_root_count(&[1.0, -3.0, 2.0]), 2);
        assert!(!is_stable(&[1.0, -3.0, 2.0]));
    }

    #[test]
    fn stable_cubic() {
        // s³ + 6s² + 11s + 6 = (s+1)(s+2)(s+3).
        assert!(is_stable(&[1.0, 6.0, 11.0, 6.0]));
    }

    #[test]
    fn unstable_cubic_missing_damping() {
        // s³ + s + 1: a missing s² term guarantees instability.
        assert!(!is_stable(&[1.0, 0.0, 1.0, 1.0]));
    }
}
