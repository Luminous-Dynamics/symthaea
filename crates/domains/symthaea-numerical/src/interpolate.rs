// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Interpolation through sample points `(xᵢ, yᵢ)`.

/// Lagrange polynomial interpolation at `x` through the given nodes. `None` if
/// there are no nodes, lengths mismatch, or two nodes share an `x`.
pub fn lagrange(xs: &[f64], ys: &[f64], x: f64) -> Option<f64> {
    if xs.is_empty() || xs.len() != ys.len() {
        return None;
    }
    let n = xs.len();
    let mut result = 0.0;
    for i in 0..n {
        let mut term = ys[i];
        for j in 0..n {
            if i != j {
                let denom = xs[i] - xs[j];
                if denom.abs() < 1e-300 {
                    return None; // duplicate node
                }
                term *= (x - xs[j]) / denom;
            }
        }
        result += term;
    }
    Some(result)
}

/// Piecewise-linear interpolation at `x`. `xs` must be sorted ascending. Values
/// outside `[xs[0], xs[last]]` are clamped to the nearest endpoint. `None` if
/// empty or mismatched.
pub fn linear(xs: &[f64], ys: &[f64], x: f64) -> Option<f64> {
    if xs.is_empty() || xs.len() != ys.len() {
        return None;
    }
    if x <= xs[0] {
        return Some(ys[0]);
    }
    if x >= xs[xs.len() - 1] {
        return Some(ys[ys.len() - 1]);
    }
    // Find the bracketing interval.
    let i = xs.partition_point(|&xi| xi <= x) - 1;
    let (x0, x1) = (xs[i], xs[i + 1]);
    let (y0, y1) = (ys[i], ys[i + 1]);
    let t = (x - x0) / (x1 - x0);
    Some(y0 + t * (y1 - y0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lagrange_recovers_a_parabola() {
        // Through (0,0),(1,1),(2,4) — the parabola y = x². Check at x=3 → 9.
        let xs = [0.0, 1.0, 2.0];
        let ys = [0.0, 1.0, 4.0];
        assert!((lagrange(&xs, &ys, 3.0).unwrap() - 9.0).abs() < 1e-12);
        assert!((lagrange(&xs, &ys, 1.5).unwrap() - 2.25).abs() < 1e-12);
    }

    #[test]
    fn lagrange_passes_through_nodes() {
        let xs = [1.0, 2.0, 5.0];
        let ys = [3.0, -1.0, 4.0];
        for (x, y) in xs.iter().zip(&ys) {
            assert!((lagrange(&xs, &ys, *x).unwrap() - y).abs() < 1e-9);
        }
        assert!(lagrange(&[1.0, 1.0], &[2.0, 3.0], 1.0).is_none()); // duplicate node
    }

    #[test]
    fn linear_interp_and_clamp() {
        let xs = [0.0, 1.0, 2.0];
        let ys = [0.0, 10.0, 20.0];
        assert!((linear(&xs, &ys, 0.5).unwrap() - 5.0).abs() < 1e-12);
        assert!((linear(&xs, &ys, 1.5).unwrap() - 15.0).abs() < 1e-12);
        assert_eq!(linear(&xs, &ys, -1.0), Some(0.0)); // clamp low
        assert_eq!(linear(&xs, &ys, 9.0), Some(20.0)); // clamp high
    }
}
