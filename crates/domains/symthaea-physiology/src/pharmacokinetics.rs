// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! One-compartment first-order pharmacokinetics.

use std::f64::consts::LN_2;

/// Elimination rate constant `kₑ = ln2/t½` (per unit time).
pub fn elimination_rate_constant(half_life: f64) -> f64 {
    LN_2 / half_life
}

/// Half-life from an elimination rate constant `t½ = ln2/kₑ`.
pub fn half_life(elimination_rate: f64) -> f64 {
    LN_2 / elimination_rate
}

/// Plasma concentration at time `t`: `C(t) = C₀·e^(−kₑ·t)`.
pub fn concentration_at(initial_concentration: f64, elimination_rate: f64, t: f64) -> f64 {
    initial_concentration * (-elimination_rate * t).exp()
}

/// Clearance `CL = kₑ·Vd` (volume per unit time).
pub fn clearance(elimination_rate: f64, volume_of_distribution: f64) -> f64 {
    elimination_rate * volume_of_distribution
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_life_roundtrip() {
        let ke = elimination_rate_constant(4.0);
        assert!((ke - 0.173287).abs() < 1e-5, "ke={ke}");
        assert!((half_life(ke) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn one_half_life_halves_concentration() {
        let ke = elimination_rate_constant(4.0);
        assert!((concentration_at(100.0, ke, 4.0) - 50.0).abs() < 1e-6);
        assert!((concentration_at(100.0, ke, 8.0) - 25.0).abs() < 1e-6);
    }

    #[test]
    fn clearance_known() {
        let ke = elimination_rate_constant(4.0);
        // Vd = 50 L → CL = ke·Vd.
        assert!((clearance(ke, 50.0) - ke * 50.0).abs() < 1e-12);
    }
}
