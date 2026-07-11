// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! AC steady-state: reactance, resonance, and series-RLC impedance.

use std::f64::consts::PI;

/// Capacitive reactance `Xc = 1/(2πfC)` (Ω).
pub fn capacitive_reactance(frequency: f64, capacitance: f64) -> f64 {
    1.0 / (2.0 * PI * frequency * capacitance)
}

/// Inductive reactance `Xl = 2πfL` (Ω).
pub fn inductive_reactance(frequency: f64, inductance: f64) -> f64 {
    2.0 * PI * frequency * inductance
}

/// Resonant frequency of an LC circuit `f₀ = 1/(2π√(LC))` (Hz).
pub fn resonant_frequency(inductance: f64, capacitance: f64) -> f64 {
    1.0 / (2.0 * PI * (inductance * capacitance).sqrt())
}

/// Series-RLC impedance magnitude `|Z| = √(R² + (Xl − Xc)²)` (Ω).
pub fn series_rlc_impedance(resistance: f64, x_l: f64, x_c: f64) -> f64 {
    (resistance * resistance + (x_l - x_c).powi(2)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reactances_known() {
        // f=60 Hz, C=1 µF → Xc ≈ 2652.6 Ω; L=1 mH → Xl ≈ 0.377 Ω.
        assert!((capacitive_reactance(60.0, 1e-6) - 2652.58).abs() < 0.1);
        assert!((inductive_reactance(60.0, 1e-3) - 0.376991).abs() < 1e-5);
    }

    #[test]
    fn resonance_known() {
        // L=1 mH, C=1 µF → f₀ ≈ 5032.9 Hz.
        assert!((resonant_frequency(1e-3, 1e-6) - 5032.92).abs() < 0.1);
    }

    #[test]
    fn impedance_is_purely_resistive_at_resonance() {
        // At f₀, Xl == Xc, so |Z| = R.
        let (l, c) = (1e-3, 1e-6);
        let f0 = resonant_frequency(l, c);
        let xl = inductive_reactance(f0, l);
        let xc = capacitive_reactance(f0, c);
        assert!((xl - xc).abs() < 1e-6);
        assert!((series_rlc_impedance(50.0, xl, xc) - 50.0).abs() < 1e-6);
    }
}
