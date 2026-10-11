// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cislunar mechanics primitives.
//!
//! This module intentionally starts with small, auditable building blocks:
//! normalized Earth-Moon CR3BP geometry, collinear Lagrange-point solving,
//! rotating/inertial frame transforms, and differential third-body gravity.
//! It is not a high-fidelity lunar ephemeris and does not establish mission
//! qualification.

use std::f64::consts::TAU;

/// Earth standard gravitational parameter, km^3/s^2.
pub const EARTH_MU_KM3_S2: f64 = 398_600.441_8;
/// Moon standard gravitational parameter, km^3/s^2.
pub const MOON_MU_KM3_S2: f64 = 4_902.800_066;
/// Mean Earth-Moon center-to-center separation, km.
pub const EARTH_MOON_MEAN_DISTANCE_KM: f64 = 384_400.0;
/// Lunar sidereal orbital period, seconds (27.321661 d).
pub const MOON_SIDEREAL_PERIOD_S: f64 = 27.321_661 * 86_400.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CislunarError {
    NonFiniteInput,
    InvalidMassParameter,
    SingularGeometry,
    RootNotBracketed,
}

/// Parameters for the circular restricted three-body problem.
///
/// Coordinates are barycentric and normalized by primary separation. The
/// primary is at `x=-mu`; the secondary is at `x=1-mu`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cr3bpSystem {
    /// m2 / (m1 + m2), equivalently mu2 / (mu1 + mu2).
    pub mu: f64,
    /// Primary-secondary separation in km.
    pub separation_km: f64,
    /// Frame angular rate in rad/s.
    pub angular_rate_rad_s: f64,
}

impl Cr3bpSystem {
    pub fn new(
        primary_mu_km3_s2: f64,
        secondary_mu_km3_s2: f64,
        separation_km: f64,
        angular_rate_rad_s: f64,
    ) -> Result<Self, CislunarError> {
        if !primary_mu_km3_s2.is_finite()
            || !secondary_mu_km3_s2.is_finite()
            || !separation_km.is_finite()
            || !angular_rate_rad_s.is_finite()
        {
            return Err(CislunarError::NonFiniteInput);
        }
        if primary_mu_km3_s2 <= 0.0
            || secondary_mu_km3_s2 <= 0.0
            || separation_km <= 0.0
            || angular_rate_rad_s <= 0.0
        {
            return Err(CislunarError::InvalidMassParameter);
        }

        let mu = secondary_mu_km3_s2 / (primary_mu_km3_s2 + secondary_mu_km3_s2);
        if !(0.0..0.5).contains(&mu) {
            return Err(CislunarError::InvalidMassParameter);
        }

        Ok(Self {
            mu,
            separation_km,
            angular_rate_rad_s,
        })
    }

    pub fn earth_moon() -> Self {
        Self::new(
            EARTH_MU_KM3_S2,
            MOON_MU_KM3_S2,
            EARTH_MOON_MEAN_DISTANCE_KM,
            TAU / MOON_SIDEREAL_PERIOD_S,
        )
        .expect("hard-coded Earth-Moon constants are valid")
    }

    /// Barycentric x-coordinate of the primary in normalized units.
    pub fn primary_x(self) -> f64 {
        -self.mu
    }

    /// Barycentric x-coordinate of the secondary in normalized units.
    pub fn secondary_x(self) -> f64 {
        1.0 - self.mu
    }

    /// Solve the collinear L1/L2/L3 equilibrium points in normalized,
    /// barycentric rotating coordinates.
    pub fn collinear_lagrange_points(self) -> Result<CollinearLagrangePoints, CislunarError> {
        if !(0.0..0.5).contains(&self.mu) {
            return Err(CislunarError::InvalidMassParameter);
        }

        // Avoid the primary/secondary singularities by a generous epsilon.
        // Bisection is slower than Newton-Raphson but deterministic and robust
        // for this one-time geometry calculation.
        let eps = 1.0e-9;
        let p1 = self.primary_x();
        let p2 = self.secondary_x();

        let l1 = bisect_root(
            |x| collinear_equilibrium(x, self.mu),
            p1 + eps,
            p2 - eps,
        )?;
        let l2 = bisect_root(
            |x| collinear_equilibrium(x, self.mu),
            p2 + eps,
            p2 + 2.0,
        )?;
        let l3 = bisect_root(
            |x| collinear_equilibrium(x, self.mu),
            p1 - 2.0,
            p1 - eps,
        )?;

        Ok(CollinearLagrangePoints { l1, l2, l3 })
    }
}

impl Default for Cr3bpSystem {
    fn default() -> Self {
        Self::earth_moon()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CollinearLagrangePoints {
    /// L1 normalized barycentric x coordinate.
    pub l1: f64,
    /// L2 normalized barycentric x coordinate.
    pub l2: f64,
    /// L3 normalized barycentric x coordinate.
    pub l3: f64,
}

impl CollinearLagrangePoints {
    pub fn l1_distance_from_secondary_km(self, system: Cr3bpSystem) -> f64 {
        (system.secondary_x() - self.l1).abs() * system.separation_km
    }

    pub fn l2_distance_from_secondary_km(self, system: Cr3bpSystem) -> f64 {
        (self.l2 - system.secondary_x()).abs() * system.separation_km
    }
}

/// x-axis equilibrium equation for the normalized CR3BP rotating frame.
fn collinear_equilibrium(x: f64, mu: f64) -> f64 {
    let r1 = x + mu;
    let r2 = x - 1.0 + mu;
    x - (1.0 - mu) * r1 / r1.abs().powi(3) - mu * r2 / r2.abs().powi(3)
}

fn bisect_root<F>(f: F, mut lo: f64, mut hi: f64) -> Result<f64, CislunarError>
where
    F: Fn(f64) -> f64,
{
    let mut flo = f(lo);
    let fhi = f(hi);
    if !flo.is_finite() || !fhi.is_finite() {
        return Err(CislunarError::SingularGeometry);
    }
    if flo == 0.0 {
        return Ok(lo);
    }
    if fhi == 0.0 {
        return Ok(hi);
    }
    if flo.signum() == fhi.signum() {
        return Err(CislunarError::RootNotBracketed);
    }

    for _ in 0..160 {
        let mid = 0.5 * (lo + hi);
        let fm = f(mid);
        if !fm.is_finite() {
            return Err(CislunarError::SingularGeometry);
        }
        if fm.abs() <= 1.0e-14 || (hi - lo).abs() <= 1.0e-14 {
            return Ok(mid);
        }
        if flo.signum() == fm.signum() {
            lo = mid;
            flo = fm;
        } else {
            hi = mid;
        }
    }

    Ok(0.5 * (lo + hi))
}

fn rotate_z(v: [f64; 3], theta_rad: f64) -> [f64; 3] {
    let (s, c) = theta_rad.sin_cos();
    [c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]]
}

fn rotate_z_inverse(v: [f64; 3], theta_rad: f64) -> [f64; 3] {
    rotate_z(v, -theta_rad)
}

fn omega_cross_r(omega_rad_s: f64, r_km: [f64; 3]) -> [f64; 3] {
    [-omega_rad_s * r_km[1], omega_rad_s * r_km[0], 0.0]
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Convert a state from a frame rotating about +Z into the corresponding
/// inertial frame. Units remain km and km/s.
pub fn rotating_to_inertial_state(
    position_rotating_km: [f64; 3],
    velocity_rotating_km_s: [f64; 3],
    theta_rad: f64,
    angular_rate_rad_s: f64,
) -> Result<([f64; 3], [f64; 3]), CislunarError> {
    if !all_finite(position_rotating_km)
        || !all_finite(velocity_rotating_km_s)
        || !theta_rad.is_finite()
        || !angular_rate_rad_s.is_finite()
    {
        return Err(CislunarError::NonFiniteInput);
    }

    let position_inertial_km = rotate_z(position_rotating_km, theta_rad);
    let velocity_plus_frame = add(
        velocity_rotating_km_s,
        omega_cross_r(angular_rate_rad_s, position_rotating_km),
    );
    let velocity_inertial_km_s = rotate_z(velocity_plus_frame, theta_rad);
    Ok((position_inertial_km, velocity_inertial_km_s))
}

/// Convert an inertial state into a frame rotating about +Z.
pub fn inertial_to_rotating_state(
    position_inertial_km: [f64; 3],
    velocity_inertial_km_s: [f64; 3],
    theta_rad: f64,
    angular_rate_rad_s: f64,
) -> Result<([f64; 3], [f64; 3]), CislunarError> {
    if !all_finite(position_inertial_km)
        || !all_finite(velocity_inertial_km_s)
        || !theta_rad.is_finite()
        || !angular_rate_rad_s.is_finite()
    {
        return Err(CislunarError::NonFiniteInput);
    }

    let position_rotating_km = rotate_z_inverse(position_inertial_km, theta_rad);
    let inertial_velocity_in_rotating_axes = rotate_z_inverse(velocity_inertial_km_s, theta_rad);
    let velocity_rotating_km_s = sub(
        inertial_velocity_in_rotating_axes,
        omega_cross_r(angular_rate_rad_s, position_rotating_km),
    );
    Ok((position_rotating_km, velocity_rotating_km_s))
}

/// Differential acceleration from a third body in a primary-centered frame.
///
/// `spacecraft_from_primary_km` and `third_body_from_primary_km` are positions
/// relative to the same primary. The indirect term is included so a
/// spacecraft at the primary origin has zero relative third-body acceleration.
pub fn third_body_acceleration_km_s2(
    spacecraft_from_primary_km: [f64; 3],
    third_body_from_primary_km: [f64; 3],
    third_body_mu_km3_s2: f64,
) -> Result<[f64; 3], CislunarError> {
    if !all_finite(spacecraft_from_primary_km)
        || !all_finite(third_body_from_primary_km)
        || !third_body_mu_km3_s2.is_finite()
    {
        return Err(CislunarError::NonFiniteInput);
    }
    if third_body_mu_km3_s2 <= 0.0 {
        return Err(CislunarError::InvalidMassParameter);
    }

    let third_to_spacecraft = sub(third_body_from_primary_km, spacecraft_from_primary_km);
    let d_sc = norm(third_to_spacecraft);
    let d_primary = norm(third_body_from_primary_km);
    if d_sc <= f64::EPSILON || d_primary <= f64::EPSILON {
        return Err(CislunarError::SingularGeometry);
    }

    let direct_scale = third_body_mu_km3_s2 / d_sc.powi(3);
    let indirect_scale = third_body_mu_km3_s2 / d_primary.powi(3);
    Ok([
        direct_scale * third_to_spacecraft[0] - indirect_scale * third_body_from_primary_km[0],
        direct_scale * third_to_spacecraft[1] - indirect_scale * third_body_from_primary_km[1],
        direct_scale * third_to_spacecraft[2] - indirect_scale * third_body_from_primary_km[2],
    ])
}

fn norm(v: [f64; 3]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn all_finite(v: [f64; 3]) -> bool {
    v.into_iter().all(f64::is_finite)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "{a} !~= {b} (tol {tol})");
    }

    #[test]
    fn earth_moon_mass_parameter_matches_reference_scale() {
        let system = Cr3bpSystem::earth_moon();
        close(system.mu, 0.012_150_58, 2.0e-8);
    }

    #[test]
    fn earth_moon_collinear_points_match_standard_cr3bp_values() {
        let points = Cr3bpSystem::earth_moon()
            .collinear_lagrange_points()
            .unwrap();
        close(points.l1, 0.836_915, 2.0e-6);
        close(points.l2, 1.155_682, 2.0e-6);
        close(points.l3, -1.005_063, 2.0e-6);
    }

    #[test]
    fn earth_moon_l1_l2_are_tens_of_thousands_km_from_moon() {
        let system = Cr3bpSystem::earth_moon();
        let points = system.collinear_lagrange_points().unwrap();
        let l1_km = points.l1_distance_from_secondary_km(system);
        let l2_km = points.l2_distance_from_secondary_km(system);
        assert!((50_000.0..70_000.0).contains(&l1_km));
        assert!((50_000.0..75_000.0).contains(&l2_km));
        assert!(l2_km > l1_km);
    }

    #[test]
    fn rotating_inertial_state_round_trip() {
        let r = [1200.0, -450.0, 33.0];
        let v = [0.2, 0.8, -0.03];
        let theta = 1.234;
        let omega = Cr3bpSystem::earth_moon().angular_rate_rad_s;

        let (ri, vi) = rotating_to_inertial_state(r, v, theta, omega).unwrap();
        let (rr, vr) = inertial_to_rotating_state(ri, vi, theta, omega).unwrap();

        for i in 0..3 {
            close(rr[i], r[i], 1.0e-10);
            close(vr[i], v[i], 1.0e-12);
        }
    }

    #[test]
    fn third_body_relative_acceleration_is_zero_at_primary_origin() {
        let a = third_body_acceleration_km_s2(
            [0.0, 0.0, 0.0],
            [EARTH_MOON_MEAN_DISTANCE_KM, 0.0, 0.0],
            MOON_MU_KM3_S2,
        )
        .unwrap();
        close(norm(a), 0.0, 1.0e-18);
    }

    #[test]
    fn malformed_inputs_fail_closed() {
        assert_eq!(
            Cr3bpSystem::new(1.0, 1.0, f64::NAN, 1.0),
            Err(CislunarError::NonFiniteInput)
        );
        assert_eq!(
            third_body_acceleration_km_s2([0.0; 3], [0.0; 3], 1.0),
            Err(CislunarError::SingularGeometry)
        );
    }
}
