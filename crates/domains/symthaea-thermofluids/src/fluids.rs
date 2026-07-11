// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Incompressible fluid mechanics: Reynolds number, Bernoulli, pipe head loss.

/// Standard gravity (m/s²).
pub const G: f64 = 9.81;

/// Flow regime by Reynolds number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Regime {
    Laminar,
    Transitional,
    Turbulent,
}

/// Reynolds number `Re = ρvD/μ` (dimensionless).
pub fn reynolds_number(density: f64, velocity: f64, diameter: f64, viscosity: f64) -> f64 {
    density * velocity * diameter / viscosity
}

/// Pipe-flow regime from Reynolds number (Re < 2300 laminar, > 4000 turbulent).
pub fn flow_regime(reynolds: f64) -> Regime {
    if reynolds < 2300.0 {
        Regime::Laminar
    } else if reynolds > 4000.0 {
        Regime::Turbulent
    } else {
        Regime::Transitional
    }
}

/// Bernoulli total head at a point: `P/(ρg) + v²/(2g) + z` (m of fluid).
pub fn bernoulli_head(pressure: f64, velocity: f64, elevation: f64, density: f64) -> f64 {
    pressure / (density * G) + velocity * velocity / (2.0 * G) + elevation
}

/// Darcy-Weisbach head loss `hf = f·(L/D)·v²/(2g)` (m).
pub fn darcy_weisbach_head_loss(
    friction_factor: f64,
    length: f64,
    diameter: f64,
    velocity: f64,
) -> f64 {
    friction_factor * (length / diameter) * velocity * velocity / (2.0 * G)
}

/// Continuity: outlet velocity from `A₁v₁ = A₂v₂` given areas and inlet velocity.
pub fn continuity_velocity(area_in: f64, velocity_in: f64, area_out: f64) -> f64 {
    area_in * velocity_in / area_out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reynolds_and_regime() {
        // water: ρ=1000, v=2, D=0.05, μ=1e-3 → Re=100000 (turbulent).
        let re = reynolds_number(1000.0, 2.0, 0.05, 1e-3);
        assert!((re - 100_000.0).abs() < 1e-6);
        assert_eq!(flow_regime(re), Regime::Turbulent);
        assert_eq!(flow_regime(1000.0), Regime::Laminar);
        assert_eq!(flow_regime(3000.0), Regime::Transitional);
    }

    #[test]
    fn bernoulli_conserved_in_ideal_flow() {
        // Narrowing pipe: as area halves, velocity doubles; total head constant
        // if pressure drops to compensate (ideal, no losses).
        let (rho, a1, v1) = (1000.0, 0.02, 1.0);
        let v2 = continuity_velocity(a1, v1, a1 / 2.0);
        assert!((v2 - 2.0).abs() < 1e-9);
        // Choose pressures so total head matches (h1 == h2).
        let p1 = 200_000.0;
        let h1 = bernoulli_head(p1, v1, 0.0, rho);
        // p2 = p1 + ½ρ(v1²−v2²).
        let p2 = p1 + 0.5 * rho * (v1 * v1 - v2 * v2);
        let h2 = bernoulli_head(p2, v2, 0.0, rho);
        assert!((h1 - h2).abs() < 1e-6);
    }

    #[test]
    fn darcy_head_loss_known() {
        // f=0.02, L=100, D=0.1, v=2 → hf ≈ 4.077 m.
        let hf = darcy_weisbach_head_loss(0.02, 100.0, 0.1, 2.0);
        assert!((hf - 4.0775).abs() < 1e-3, "hf={hf}");
    }
}
