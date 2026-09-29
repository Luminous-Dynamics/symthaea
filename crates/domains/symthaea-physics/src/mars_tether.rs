// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! First-order Mars surface-to-synchronous-tether mechanics.
//!
//! This module deliberately stops short of a full elevator solver. It provides
//! an auditable T0/T1 reference kernel: Mars constants, rotating-frame
//! effective acceleration, synchronous radius, constant-stress taper exponent,
//! and distributed-mass quadrature. Higher-fidelity orbital ephemerides,
//! flexible dynamics, climber loads, and termination/counterweight boundary
//! conditions belong in separate adapters.
//!
//! The implementation is intentionally dependency-free so its limiting cases
//! can be cross-checked against hand calculations and external solvers.

use std::f64::consts::PI;

/// Mars reference constants used by the first-order tether model.
///
/// Values follow NASA's Mars fact sheet / standard reference values used by
/// the engineering foundation document. SI units are used internally.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MarsTetherReference {
    /// Mars gravitational parameter, m^3/s^2.
    pub mu_m3_s2: f64,
    /// Mean equatorial reference radius, m.
    pub radius_m: f64,
    /// Sidereal rotation period, s.
    pub rotation_period_s: f64,
    /// Phobos mean orbital radius from Mars center, m.
    pub phobos_radius_m: f64,
    /// Deimos mean orbital radius from Mars center, m.
    pub deimos_radius_m: f64,
}

impl MarsTetherReference {
    pub const MARS: Self = Self {
        mu_m3_s2: 4.2828372e13,
        radius_m: 3_396_200.0,
        rotation_period_s: 88_642.44,
        phobos_radius_m: 9_378_000.0,
        deimos_radius_m: 23_459_000.0,
    };

    /// Mars angular rotation rate, rad/s.
    pub const fn omega_rad_s(self) -> f64 {
        2.0 * PI / self.rotation_period_s
    }

    /// Radius of the circular synchronous orbit, m.
    pub fn synchronous_radius_m(self) -> f64 {
        (self.mu_m3_s2 / self.omega_rad_s().powi(2)).cbrt()
    }

    /// Synchronous altitude above the reference radius, m.
    pub fn synchronous_altitude_m(self) -> f64 {
        self.synchronous_radius_m() - self.radius_m
    }

    /// Radial effective acceleration in the Mars-corotating frame, m/s^2.
    ///
    /// Positive means centrifugal acceleration exceeds Mars gravity;
    /// negative means gravity dominates.
    pub fn effective_radial_acceleration_m_s2(self, radius_m: f64) -> f64 {
        let r = radius_m.max(1.0);
        self.omega_rad_s().powi(2) * r - self.mu_m3_s2 / r.powi(2)
    }

    /// Negative integral of effective acceleration from r0 to r1.
    ///
    /// This is the specific-energy-like quantity appearing in the
    /// constant-stress taper exponent:
    ///   ln(A1/A0) = -integral(a_eff dr) / (sigma/rho).
    pub fn taper_integral_m2_s2(self, r0_m: f64, r1_m: f64) -> f64 {
        let r0 = r0_m.max(1.0);
        let r1 = r1_m.max(1.0);
        self.mu_m3_s2 * (1.0 / r0 - 1.0 / r1)
            - 0.5 * self.omega_rad_s().powi(2) * (r1.powi(2) - r0.powi(2))
    }
}

/// Effective material properties for the first-order tether model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TetherMaterial {
    /// Bulk density, kg/m^3.
    pub density_kg_m3: f64,
    /// Allowable axial stress after all design knock-down factors, Pa.
    pub allowable_stress_pa: f64,
}

impl TetherMaterial {
    /// Specific allowable strength, sigma/rho, m^2/s^2.
    pub fn specific_strength_m2_s2(self) -> f64 {
        self.allowable_stress_pa / self.density_kg_m3.max(f64::MIN_POSITIVE)
    }
}

/// First-order constant-stress taper result between two radii.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TaperResult {
    pub r0_m: f64,
    pub r1_m: f64,
    pub taper_integral_m2_s2: f64,
    pub specific_strength_m2_s2: f64,
    pub area_ratio_a1_a0: f64,
}

/// Compute the constant-stress area ratio A(r1)/A(r0).
///
/// This is a limiting-case sanity check, not a complete elevator design:
/// the model has no apex/counterweight boundary condition, climber loads,
/// dynamic loads, defects, bending, fatigue, or environmental degradation.
pub fn constant_stress_taper(
    reference: MarsTetherReference,
    material: TetherMaterial,
    r0_m: f64,
    r1_m: f64,
) -> TaperResult {
    let integral = reference.taper_integral_m2_s2(r0_m, r1_m);
    let specific_strength = material.specific_strength_m2_s2();
    let exponent = (integral / specific_strength).clamp(-700.0, 700.0);
    TaperResult {
        r0_m,
        r1_m,
        taper_integral_m2_s2: integral,
        specific_strength_m2_s2: specific_strength,
        area_ratio_a1_a0: exponent.exp(),
    }
}

/// Numerically integrate the tether area profile implied by constant stress.
///
/// Returns tether mass per unit anchor area, kg/m^2, over [r0, r1].
/// The result is useful for trade studies because it exposes the mass
/// consequence of material specific strength without hiding it in a single
/// feasibility score.
pub fn mass_per_anchor_area(
    reference: MarsTetherReference,
    material: TetherMaterial,
    r0_m: f64,
    r1_m: f64,
    steps: usize,
) -> f64 {
    assert!(steps > 0, "steps must be non-zero");
    let n = if steps % 2 == 0 { steps } else { steps + 1 };
    let h = (r1_m - r0_m) / n as f64;
    let mut sum = 0.0;
    for i in 0..=n {
        let r = r0_m + i as f64 * h;
        let a_ratio = constant_stress_taper(reference, material, r0_m, r).area_ratio_a1_a0;
        let weight = if i == 0 || i == n {
            1.0
        } else if i % 2 == 0 {
            2.0
        } else {
            4.0
        };
        sum += weight * material.density_kg_m3 * a_ratio;
    }
    sum * h / 3.0
}

/// Dimensionless phase-independent clearance margin for a satellite radius.
///
/// This does not replace a phase-aware ephemeris. It simply states whether
/// the tether's radial interval intersects a satellite's mean orbital radius.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadialIntersection {
    IntersectsMeanOrbit,
    ClearOfMeanOrbit,
}

pub fn mean_orbit_intersection(
    r0_m: f64,
    r1_m: f64,
    satellite_radius_m: f64,
) -> RadialIntersection {
    let lo = r0_m.min(r1_m);
    let hi = r0_m.max(r1_m);
    if (lo..=hi).contains(&satellite_radius_m) {
        RadialIntersection::IntersectsMeanOrbit
    } else {
        RadialIntersection::ClearOfMeanOrbit
    }
}

/// Compact reference envelope for the Mars surface-to-AMO problem.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MarsTetherReferenceEnvelope {
    pub surface_radius_m: f64,
    pub synchronous_radius_m: f64,
    pub synchronous_altitude_m: f64,
    pub phobos_radius_m: f64,
    pub deimos_radius_m: f64,
    pub surface_effective_accel_m_s2: f64,
}

impl MarsTetherReferenceEnvelope {
    pub fn mars() -> Self {
        let m = MarsTetherReference::MARS;
        Self {
            surface_radius_m: m.radius_m,
            synchronous_radius_m: m.synchronous_radius_m(),
            synchronous_altitude_m: m.synchronous_altitude_m(),
            phobos_radius_m: m.phobos_radius_m,
            deimos_radius_m: m.deimos_radius_m,
            surface_effective_accel_m_s2: m.effective_radial_acceleration_m_s2(m.radius_m),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS_RADIUS_M: f64 = 5.0;
    const EPS_ACCEL: f64 = 1e-6;

    #[test]
    fn synchronous_radius_matches_independent_reference() {
        let m = MarsTetherReference::MARS;
        assert!((m.synchronous_radius_m() - 20_427_650.0).abs() < EPS_RADIUS_M);
        assert!((m.synchronous_altitude_m() - 17_031_450.0).abs() < EPS_RADIUS_M);
    }

    #[test]
    fn effective_acceleration_changes_sign_at_sync() {
        let m = MarsTetherReference::MARS;
        assert!(m.effective_radial_acceleration_m_s2(m.radius_m) < -3.69);
        assert!(m.effective_radial_acceleration_m_s2(m.synchronous_radius_m()).abs() < EPS_ACCEL);
        assert!(m.effective_radial_acceleration_m_s2(m.deimos_radius_m) > 0.04 - EPS_ACCEL);
    }

    #[test]
    fn phobos_and_deimos_are_explicitly_inside_and_outside_sync() {
        let m = MarsTetherReference::MARS;
        assert!(m.phobos_radius_m < m.synchronous_radius_m());
        assert!(m.deimos_radius_m > m.synchronous_radius_m());
        assert_eq!(
            mean_orbit_intersection(m.radius_m, m.synchronous_radius_m(), m.phobos_radius_m),
            RadialIntersection::IntersectsMeanOrbit
        );
        assert_eq!(
            mean_orbit_intersection(m.radius_m, m.synchronous_radius_m(), m.deimos_radius_m),
            RadialIntersection::ClearOfMeanOrbit
        );
    }

    #[test]
    fn taper_sanity_matches_analytic_area_ratios() {
        let m = MarsTetherReference::MARS;
        let material = TetherMaterial {
            density_kg_m3: 1.0,
            allowable_stress_pa: 15.0e6,
        };
        let t = constant_stress_taper(m, material, m.radius_m, m.synchronous_radius_m());
        assert!((t.taper_integral_m2_s2 - 9.4948e6).abs() < 100.0);
        assert!((t.area_ratio_a1_a0 - 1.883).abs() < 0.005);
    }

    #[test]
    fn stronger_specific_strength_reduces_taper() {
        let m = MarsTetherReference::MARS;
        let weak = constant_stress_taper(
            m,
            TetherMaterial { density_kg_m3: 1.0, allowable_stress_pa: 5.0e6 },
            m.radius_m,
            m.synchronous_radius_m(),
        );
        let strong = constant_stress_taper(
            m,
            TetherMaterial { density_kg_m3: 1.0, allowable_stress_pa: 50.0e6 },
            m.radius_m,
            m.synchronous_radius_m(),
        );
        assert!(strong.area_ratio_a1_a0 < weak.area_ratio_a1_a0);
    }

    #[test]
    fn mass_per_area_is_finite_and_positive() {
        let m = MarsTetherReference::MARS;
        let material = TetherMaterial {
            density_kg_m3: 2260.0,
            allowable_stress_pa: 15.0e9,
        };
        let mass = mass_per_anchor_area(
            m,
            material,
            m.radius_m,
            m.synchronous_radius_m(),
            200,
        );
        assert!(mass.is_finite());
        assert!(mass > 0.0);
    }
}
