// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Exotic-matter energy requirement for the metric perturbation this crate simulates.
//!
//! [`amplifier::alcubierre_shaping`] and the RK4 integrator move a craft through a
//! shift-vector field, but neither computes what sources that field. This module
//! answers the question the rest of the crate is silent on: how much negative
//! energy density does General Relativity require to actually produce the shift
//! vector the simulator assumes for free?
//!
//! ## Derivation
//!
//! For a single Alcubierre shift vector along one axis (the classic 1994 geometry:
//! `β^x = v_s f(r_s)`, `β^y = β^z = 0`), the energy density measured by an Eulerian
//! observer is (Alcubierre, *Class. Quantum Grav.* 11 L73 (1994); reproduced e.g. in
//! Lobo, "Wormholes, Warp Drives and Energy Conditions" (2007), eq. 10-11):
//!
//! ```text
//! ρ(r_s, θ) = -(v_s² / 32π) · sin²θ · (df/dr_s)²        [geometrized units, G=c=1]
//! ```
//!
//! where `θ` is the polar angle from the direction of travel, so `sin²θ = (y²+z²)/r_s²`.
//! This module reuses that exact formula — same shaping function
//! ([`crate::amplifier::alcubierre_shaping_derivative`]) as the simulator — rather than
//! re-deriving stress-energy for the crate's simplified three-amplifier shift field, so
//! treat the result as an order-of-magnitude physical grounding check on the single-bubble
//! geometry this crate is modeled after, not an exact solve of this crate's own metric.
//!
//! Total exotic mass-energy follows by integrating ρ over all space in spherical
//! coordinates centered on the bubble. The angular part is elementary
//! (`∫sin³θ dθ = 4/3`, `∫dφ = 2π`) and reduces the problem to a 1-D radial integral of
//! `(df/dr_s)² r_s²`, computed numerically here directly from the crate's own shaping
//! function — see [`radial_shape_integral`].

use crate::amplifier::alcubierre_shaping_derivative;
use symthaea_core::physics::constants::{C, G};

/// Reduced Planck constant, J·s (CODATA).
const HBAR: f64 = 1.054_571_817e-34;

/// `∫₀^∞ (df/dr_s)² r_s² dr_s`, evaluated numerically via Simpson's rule.
///
/// `f'` is sharply peaked near `r_s = radius` (width `~1/sigma`) and decays
/// exponentially away from the wall, so a bounded integration range with a
/// generous margin is exact to numerical precision.
pub fn radial_shape_integral(radius: f64, sigma: f64) -> f64 {
    if radius <= 0.0 || sigma <= 0.0 {
        return 0.0;
    }
    let r_max = radius + 15.0 / sigma; // f' is ~0 well before this for any reasonable sigma
    let steps = 4000usize.max((steps_for(sigma, r_max)) as usize);
    let steps = if steps % 2 == 1 { steps + 1 } else { steps }; // Simpson needs even steps
    let h = r_max / steps as f64;

    let integrand = |r_s: f64| {
        let fp = alcubierre_shaping_derivative(r_s, radius, sigma);
        fp * fp * r_s * r_s
    };

    let mut sum = integrand(0.0) + integrand(r_max);
    for i in 1..steps {
        let r_s = i as f64 * h;
        let weight = if i % 2 == 0 { 2.0 } else { 4.0 };
        sum += weight * integrand(r_s);
    }
    sum * h / 3.0
}

fn steps_for(sigma: f64, r_max: f64) -> f64 {
    // Resolve the wall (width ~1/sigma) with at least ~50 points across it.
    (r_max * sigma * 50.0).clamp(2000.0, 200_000.0)
}

/// Total exotic mass (kg) required to source this bubble at coordinate speed `v_frac`
/// (a fraction of c — pass `v_ms / C` for a velocity in m/s).
///
/// Follows from `M_geom = (v_frac² / 12) · I` (this crate's own angular-integral
/// derivation of the Alcubierre formula above), converted from geometrized units
/// (length) to kilograms via `c²/G`.
pub fn exotic_mass_kg(radius: f64, sigma: f64, v_frac: f64) -> f64 {
    let i = radial_shape_integral(radius, sigma);
    (v_frac * v_frac / 12.0) * i * C * C / G
}

/// Total exotic energy (Joules), i.e. `exotic_mass_kg() * c²`.
pub fn exotic_energy_joules(radius: f64, sigma: f64, v_frac: f64) -> f64 {
    exotic_mass_kg(radius, sigma, v_frac) * C * C
}

/// Peak local negative energy density (J/m³, magnitude) — occurs at `r_s = radius`,
/// `θ = 90°` (the bubble's equator, perpendicular to the direction of travel), where
/// both `sin²θ` and `|df/dr_s|` are maximal.
pub fn peak_energy_density_si(radius: f64, sigma: f64, v_frac: f64) -> f64 {
    let fp_peak = alcubierre_shaping_derivative(radius, radius, sigma).abs();
    let rho_geom = (v_frac * v_frac / (32.0 * std::f64::consts::PI)) * fp_peak * fp_peak;
    rho_geom * C.powi(4) / G
}

/// Casimir-effect negative energy density (J/m³, magnitude) between two parallel
/// conducting plates separated by `gap_m` — the only negative energy density humanity
/// has ever actually produced and measured. Formula: `π²ħc / (720 d⁴)`.
pub fn casimir_energy_density_si(gap_m: f64) -> f64 {
    std::f64::consts::PI.powi(2) * HBAR * C / (720.0 * gap_m.powi(4))
}

/// How many orders of magnitude larger the required peak density is than the most
/// extreme negative energy density ever measured (Casimir plates at a 1 nm gap —
/// smaller gaps are not experimentally achievable with flat plates).
pub fn feasibility_gap_orders_of_magnitude(radius: f64, sigma: f64, v_frac: f64) -> f64 {
    let required = peak_energy_density_si(radius, sigma, v_frac);
    let achieved = casimir_energy_density_si(1e-9);
    if achieved <= 0.0 || required <= 0.0 {
        return f64::NEG_INFINITY;
    }
    (required / achieved).log10()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::GravcraftConfig;

    #[test]
    fn radial_integral_positive_and_finite() {
        let cfg = GravcraftConfig::default();
        let i = radial_shape_integral(cfg.bubble_radius, cfg.wall_sigma);
        assert!(i > 0.0 && i.is_finite());
    }

    #[test]
    fn radial_integral_zero_for_degenerate_shape() {
        assert_eq!(radial_shape_integral(0.0, 1.0), 0.0);
        assert_eq!(radial_shape_integral(1.0, 0.0), 0.0);
    }

    #[test]
    fn exotic_mass_scales_with_v_squared() {
        let cfg = GravcraftConfig::default();
        let m_slow = exotic_mass_kg(cfg.bubble_radius, cfg.wall_sigma, 0.01);
        let m_fast = exotic_mass_kg(cfg.bubble_radius, cfg.wall_sigma, 0.02);
        // Doubling v_frac should ~quadruple the required mass.
        let ratio = m_fast / m_slow;
        assert!((ratio - 4.0).abs() < 0.01, "ratio={}", ratio);
    }

    #[test]
    fn exotic_mass_zero_at_rest() {
        let cfg = GravcraftConfig::default();
        assert_eq!(exotic_mass_kg(cfg.bubble_radius, cfg.wall_sigma, 0.0), 0.0);
    }

    #[test]
    fn peak_density_is_negative_scale_but_reported_as_magnitude() {
        let cfg = GravcraftConfig::default();
        // At even a tiny fraction of c, a 5m bubble already needs an enormous density.
        let rho = peak_energy_density_si(cfg.bubble_radius, cfg.wall_sigma, 0.01);
        assert!(rho > 0.0 && rho.is_finite());
    }

    #[test]
    fn casimir_density_matches_known_order_of_magnitude() {
        // At d = 1 micron, |rho_casimir| is on the order of 1e-3 J/m^3 (textbook value).
        let rho = casimir_energy_density_si(1e-6);
        assert!(
            (1e-5..1e-1).contains(&rho),
            "casimir density at 1um out of expected range: {}",
            rho
        );
    }

    #[test]
    fn feasibility_gap_is_astronomically_large_for_default_config() {
        let cfg = GravcraftConfig::default();
        // Even at 1% of c with this crate's default 5m bubble, the gap to the most
        // extreme negative energy density ever measured (Casimir at 1nm) is enormous.
        let gap = feasibility_gap_orders_of_magnitude(cfg.bubble_radius, cfg.wall_sigma, 0.01);
        assert!(
            gap > 20.0,
            "expected the exotic-matter requirement to dwarf anything ever measured, got {} orders of magnitude",
            gap
        );
    }
}
