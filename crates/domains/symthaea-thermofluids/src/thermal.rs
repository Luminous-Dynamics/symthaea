// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Heat engines and heat transfer.

/// Carnot efficiency `η = 1 − Tc/Th` (temperatures in kelvin).
pub fn carnot_efficiency(t_cold: f64, t_hot: f64) -> f64 {
    1.0 - t_cold / t_hot
}

/// Fourier conduction heat rate `q = k·A·ΔT/L` (W).
pub fn conduction_heat_rate(conductivity: f64, area: f64, delta_temp: f64, thickness: f64) -> f64 {
    conductivity * area * delta_temp / thickness
}

/// Newton's law of cooling heat rate `q = h·A·ΔT` (W).
pub fn convection_heat_rate(coefficient: f64, area: f64, delta_temp: f64) -> f64 {
    coefficient * area * delta_temp
}

/// Work output of a heat engine given heat input and thermal efficiency (J).
pub fn engine_work(heat_input: f64, efficiency: f64) -> f64 {
    heat_input * efficiency
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn carnot_known() {
        // Tc=300 K, Th=600 K → η = 0.5.
        assert!((carnot_efficiency(300.0, 600.0) - 0.5).abs() < 1e-12);
        // Efficiency rises as the hot reservoir gets hotter.
        assert!(carnot_efficiency(300.0, 900.0) > carnot_efficiency(300.0, 600.0));
    }

    #[test]
    fn conduction_known() {
        // k=200, A=1, ΔT=50, L=0.1 → q = 100 kW.
        assert!((conduction_heat_rate(200.0, 1.0, 50.0, 0.1) - 100_000.0).abs() < 1e-6);
    }

    #[test]
    fn engine_work_bounded_by_carnot() {
        let eta = carnot_efficiency(300.0, 600.0);
        assert!((engine_work(1000.0, eta) - 500.0).abs() < 1e-9);
    }
}
