// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Single-species logistic growth: bounded growth toward a carrying capacity.

/// Instantaneous growth rate dN/dt = r·N·(1 − N/K).
pub fn growth_rate(n: f64, r: f64, k: f64) -> f64 {
    r * n * (1.0 - n / k)
}

/// The closed-form logistic solution
/// `N(t) = K / (1 + ((K − N₀)/N₀)·e^{−r t})`.
/// Requires `n0 > 0` and `k > 0`.
pub fn population(n0: f64, r: f64, k: f64, t: f64) -> f64 {
    if n0 <= 0.0 {
        return 0.0;
    }
    k / (1.0 + ((k - n0) / n0) * (-r * t).exp())
}

/// Time at which the population reaches `target` (must lie strictly between
/// `n0` and `K`, same side of `K`). `None` if unreachable or degenerate.
pub fn time_to_reach(n0: f64, r: f64, k: f64, target: f64) -> Option<f64> {
    if n0 <= 0.0 || k <= 0.0 || r <= 0.0 || target <= 0.0 || target >= k {
        return None;
    }
    // Invert the closed form for t.
    let ratio = ((k - target) / target) / ((k - n0) / n0);
    if ratio <= 0.0 {
        return None;
    }
    Some(-ratio.ln() / r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_and_asymptote() {
        // N(0) = N₀; N(∞) → K.
        assert!((population(10.0, 1.0, 100.0, 0.0) - 10.0).abs() < 1e-12);
        assert!((population(10.0, 1.0, 100.0, 1000.0) - 100.0).abs() < 1e-6);
    }

    #[test]
    fn growth_is_zero_at_capacity_and_extinction() {
        assert!(growth_rate(100.0, 0.5, 100.0).abs() < 1e-12);
        assert!(growth_rate(0.0, 0.5, 100.0).abs() < 1e-12);
        // Maximum growth at N = K/2.
        let g_half = growth_rate(50.0, 0.5, 100.0);
        assert!(g_half > growth_rate(30.0, 0.5, 100.0));
        assert!(g_half > growth_rate(70.0, 0.5, 100.0));
    }

    #[test]
    fn time_to_reach_inverts_population() {
        let t = time_to_reach(10.0, 0.5, 100.0, 50.0).unwrap();
        assert!((population(10.0, 0.5, 100.0, t) - 50.0).abs() < 1e-9);
    }
}
