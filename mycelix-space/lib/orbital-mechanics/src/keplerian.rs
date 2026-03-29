// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Keplerian Orbital Elements and Maneuver Planning
//!
//! Provides conversion between state vectors and classical orbital elements,
//! orbital mechanics computations, and impulsive maneuver planning.
//!
//! # Coordinate Convention
//! All angles are in radians. Positions in km, velocities in km/s.
//! Uses WGS-84 gravitational parameter (μ = 398600.4418 km³/s²).

use chrono::{DateTime, Utc};
use nalgebra::Vector3;

use crate::coordinates::wgs84::MU;
use crate::state::StateVector;

/// Classical Keplerian orbital elements
#[derive(Clone, Debug, PartialEq)]
pub struct KeplerianElements {
    /// Semi-major axis (km)
    pub semi_major_axis: f64,
    /// Eccentricity (0 = circular, 0..1 = elliptical)
    pub eccentricity: f64,
    /// Inclination (rad)
    pub inclination: f64,
    /// Right ascension of the ascending node (rad)
    pub raan: f64,
    /// Argument of periapsis (rad)
    pub arg_periapsis: f64,
    /// True anomaly (rad)
    pub true_anomaly: f64,
}

/// Error type for Keplerian conversions
#[derive(Debug, thiserror::Error)]
pub enum KeplerianError {
    #[error("Degenerate orbit: {0}")]
    DegenerateOrbit(String),
    #[error("Hyperbolic orbit (e >= 1.0): e = {0}")]
    HyperbolicOrbit(f64),
    #[error("Invalid semi-major axis: {0} km")]
    InvalidSemiMajorAxis(f64),
    #[error("Negative radius: {0} km")]
    NegativeRadius(f64),
}

impl KeplerianElements {
    /// Convert from a state vector (position + velocity) to Keplerian elements.
    ///
    /// Uses the classical algorithm via angular momentum vector h = r × v,
    /// node vector n = k × h, and eccentricity vector.
    pub fn from_state_vector(sv: &StateVector) -> Result<Self, KeplerianError> {
        let r_vec = sv.position();
        let v_vec = sv.velocity();
        let r = r_vec.norm();
        let v = v_vec.norm();

        if r < 1e-10 {
            return Err(KeplerianError::NegativeRadius(r));
        }

        // Angular momentum vector: h = r × v
        let h_vec = r_vec.cross(&v_vec);
        let h = h_vec.norm();

        if h < 1e-10 {
            return Err(KeplerianError::DegenerateOrbit(
                "Zero angular momentum (rectilinear orbit)".into(),
            ));
        }

        // Node vector: n = k × h (k = [0,0,1])
        let k = Vector3::new(0.0, 0.0, 1.0);
        let n_vec = k.cross(&h_vec);
        let n = n_vec.norm();

        // Eccentricity vector: e = (v × h)/μ - r̂
        let e_vec = v_vec.cross(&h_vec) / MU - r_vec / r;
        let e = e_vec.norm();

        if e >= 1.0 {
            return Err(KeplerianError::HyperbolicOrbit(e));
        }

        // Semi-major axis: a = -μ/(2ε) where ε = v²/2 - μ/r
        let energy = v * v / 2.0 - MU / r;
        if energy.abs() < 1e-14 {
            return Err(KeplerianError::DegenerateOrbit("Parabolic orbit".into()));
        }
        let a = -MU / (2.0 * energy);

        if a <= 0.0 {
            return Err(KeplerianError::InvalidSemiMajorAxis(a));
        }

        // Inclination: i = acos(hz / |h|)
        let inc = (h_vec.z / h).clamp(-1.0, 1.0).acos();

        // RAAN: Ω = acos(nx / |n|)
        let raan = if n > 1e-10 {
            let cos_raan = (n_vec.x / n).clamp(-1.0, 1.0);
            let mut omega = cos_raan.acos();
            if n_vec.y < 0.0 {
                omega = 2.0 * std::f64::consts::PI - omega;
            }
            omega
        } else {
            0.0 // Equatorial orbit — RAAN undefined, set to 0
        };

        // Argument of periapsis: ω = acos(n · e / (|n| |e|))
        let arg_p = if e > 1e-10 && n > 1e-10 {
            let cos_w = (n_vec.dot(&e_vec) / (n * e)).clamp(-1.0, 1.0);
            let mut w = cos_w.acos();
            if e_vec.z < 0.0 {
                w = 2.0 * std::f64::consts::PI - w;
            }
            w
        } else if e > 1e-10 {
            // Equatorial orbit: measure from x-axis
            let mut w = e_vec.y.atan2(e_vec.x);
            if w < 0.0 {
                w += 2.0 * std::f64::consts::PI;
            }
            w
        } else {
            0.0 // Circular orbit — ω undefined, set to 0
        };

        // True anomaly: ν = acos(e · r / (|e| |r|))
        let nu = if e > 1e-10 {
            let cos_nu = (e_vec.dot(&r_vec) / (e * r)).clamp(-1.0, 1.0);
            let mut nu = cos_nu.acos();
            if r_vec.dot(&v_vec) < 0.0 {
                nu = 2.0 * std::f64::consts::PI - nu;
            }
            nu
        } else if n > 1e-10 {
            // Circular inclined: measure from ascending node
            let cos_u = (n_vec.dot(&r_vec) / (n * r)).clamp(-1.0, 1.0);
            let mut u = cos_u.acos();
            if r_vec.z < 0.0 {
                u = 2.0 * std::f64::consts::PI - u;
            }
            u
        } else {
            // Circular equatorial: measure from x-axis
            let mut l = r_vec.y.atan2(r_vec.x);
            if l < 0.0 {
                l += 2.0 * std::f64::consts::PI;
            }
            l
        };

        Ok(KeplerianElements {
            semi_major_axis: a,
            eccentricity: e,
            inclination: inc,
            raan,
            arg_periapsis: arg_p,
            true_anomaly: nu,
        })
    }

    /// Convert Keplerian elements back to a state vector.
    ///
    /// Computes position and velocity in the perifocal frame,
    /// then rotates to the inertial (ECI/TEME) frame.
    pub fn to_state_vector(&self) -> StateVector {
        let a = self.semi_major_axis;
        let e = self.eccentricity;
        let nu = self.true_anomaly;

        // Semi-latus rectum
        let p = a * (1.0 - e * e);

        // Distance
        let r = p / (1.0 + e * nu.cos());

        // Position in perifocal frame (PQW)
        let r_pqw = Vector3::new(r * nu.cos(), r * nu.sin(), 0.0);

        // Velocity in perifocal frame
        let sqrt_mu_p = (MU / p).sqrt();
        let v_pqw = Vector3::new(-sqrt_mu_p * nu.sin(), sqrt_mu_p * (e + nu.cos()), 0.0);

        // Rotation from perifocal to inertial frame
        let r_eci = self.perifocal_to_inertial(r_pqw);
        let v_eci = self.perifocal_to_inertial(v_pqw);

        StateVector::from_vectors(r_eci, v_eci)
    }

    /// Rotate a vector from perifocal (PQW) to inertial frame using Ω, i, ω.
    fn perifocal_to_inertial(&self, v: Vector3<f64>) -> Vector3<f64> {
        let co = self.raan.cos();
        let so = self.raan.sin();
        let ci = self.inclination.cos();
        let si = self.inclination.sin();
        let cw = self.arg_periapsis.cos();
        let sw = self.arg_periapsis.sin();

        Vector3::new(
            (co * cw - so * sw * ci) * v.x + (-co * sw - so * cw * ci) * v.y,
            (so * cw + co * sw * ci) * v.x + (-so * sw + co * cw * ci) * v.y,
            (sw * si) * v.x + (cw * si) * v.y,
        )
    }

    /// Orbital period (seconds).
    ///
    /// T = 2π √(a³/μ)
    pub fn orbital_period(&self) -> f64 {
        2.0 * std::f64::consts::PI * (self.semi_major_axis.powi(3) / MU).sqrt()
    }

    /// Specific orbital energy (km²/s²).
    ///
    /// ε = -μ / (2a)
    pub fn orbital_energy(&self) -> f64 {
        -MU / (2.0 * self.semi_major_axis)
    }

    /// Mean anomaly (rad) from true anomaly via eccentric anomaly.
    pub fn mean_anomaly(&self) -> f64 {
        let e = self.eccentricity;
        let nu = self.true_anomaly;

        // Eccentric anomaly from true anomaly
        let cos_e = (e + nu.cos()) / (1.0 + e * nu.cos());
        let sin_e = ((1.0 - e * e).sqrt() * nu.sin()) / (1.0 + e * nu.cos());
        let ea = sin_e.atan2(cos_e);

        // Mean anomaly from Kepler's equation: M = E - e * sin(E)
        let m = ea - e * ea.sin();
        if m < 0.0 {
            m + 2.0 * std::f64::consts::PI
        } else {
            m
        }
    }

    /// Apoapsis altitude above Earth surface (km).
    pub fn apoapsis_altitude(&self) -> f64 {
        self.semi_major_axis * (1.0 + self.eccentricity) - 6378.137
    }

    /// Periapsis altitude above Earth surface (km).
    pub fn periapsis_altitude(&self) -> f64 {
        self.semi_major_axis * (1.0 - self.eccentricity) - 6378.137
    }
}

/// An impulsive maneuver (instantaneous velocity change).
#[derive(Clone, Debug)]
pub struct ImpulsiveManeuver {
    /// Delta-V vector in inertial frame (km/s)
    pub delta_v: Vector3<f64>,
    /// Time of the burn
    pub time: DateTime<Utc>,
}

impl ImpulsiveManeuver {
    /// Magnitude of the delta-V (km/s)
    pub fn magnitude(&self) -> f64 {
        self.delta_v.norm()
    }

    /// Apply this maneuver to a state vector, returning the post-burn state.
    pub fn apply(&self, sv: &StateVector) -> StateVector {
        let new_vel = sv.velocity() + self.delta_v;
        StateVector::from_vectors(sv.position(), new_vel)
    }
}

/// Compute a Hohmann transfer between two circular orbits.
///
/// Returns (burn1_dv, burn2_dv) where burn1 is at departure and burn2 at arrival.
/// Both delta-V magnitudes are in km/s. Assumes prograde burns.
///
/// `r1` and `r2` are orbital radii (km) — not altitudes. Add Earth radius (6378.137 km) to altitudes.
pub fn hohmann_transfer(r1: f64, r2: f64) -> (f64, f64) {
    let a_transfer = (r1 + r2) / 2.0;

    // Circular velocities
    let v_circ1 = (MU / r1).sqrt();
    let v_circ2 = (MU / r2).sqrt();

    // Transfer orbit velocities at each apse
    let v_transfer_periapsis = (MU * (2.0 / r1 - 1.0 / a_transfer)).sqrt();
    let v_transfer_apoapsis = (MU * (2.0 / r2 - 1.0 / a_transfer)).sqrt();

    let dv1 = (v_transfer_periapsis - v_circ1).abs();
    let dv2 = (v_circ2 - v_transfer_apoapsis).abs();

    (dv1, dv2)
}

/// Compute a minimum-ΔV collision avoidance maneuver.
///
/// Applies a radial impulse to increase the miss distance to at least `min_distance_km`.
/// Returns the required delta-V vector in the radial direction.
///
/// This is a simplified model — a radial burn at the current position changes the
/// in-track separation at TCA.
pub fn collision_avoidance_maneuver(
    state: &StateVector,
    threat: &StateVector,
    min_distance_km: f64,
    burn_time: DateTime<Utc>,
) -> ImpulsiveManeuver {
    let miss_vec = threat.position() - state.position();
    let current_miss = miss_vec.norm();

    if current_miss >= min_distance_km {
        // Already safe — zero delta-V
        return ImpulsiveManeuver {
            delta_v: Vector3::zeros(),
            time: burn_time,
        };
    }

    // Compute radial unit vector (away from Earth center)
    let r_hat = state.position().normalize();

    // Required additional displacement
    let delta_miss = min_distance_km - current_miss;

    // Rough scaling: a radial burn of Δv creates ~Δv * T_encounter displacement
    // For a typical LEO encounter (T ~ 100s relative pass time), scale factor ~100
    let encounter_time_scale = 100.0; // seconds (typical short-range encounter)
    let dv_magnitude = delta_miss / encounter_time_scale;

    // Direction: away from threat in the radial direction
    let away_from_threat = if miss_vec.dot(&r_hat) > 0.0 {
        -r_hat // Burn inward (toward Earth) to move away from threat above
    } else {
        r_hat // Burn outward to move away from threat below
    };

    ImpulsiveManeuver {
        delta_v: away_from_threat * dv_magnitude,
        time: burn_time,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Test round-trip conversion: state vector → Keplerian → state vector
    #[test]
    fn test_state_keplerian_roundtrip() {
        // ISS-like orbit
        let sv = StateVector::new(
            -2500.0, 6500.0, 1000.0, // position (km)
            -5.5, -3.5, 2.0, // velocity (km/s)
        );

        let kep = KeplerianElements::from_state_vector(&sv).unwrap();
        let sv_recovered = kep.to_state_vector();

        assert_relative_eq!(sv.x, sv_recovered.x, epsilon = 1e-6);
        assert_relative_eq!(sv.y, sv_recovered.y, epsilon = 1e-6);
        assert_relative_eq!(sv.z, sv_recovered.z, epsilon = 1e-6);
        assert_relative_eq!(sv.vx, sv_recovered.vx, epsilon = 1e-6);
        assert_relative_eq!(sv.vy, sv_recovered.vy, epsilon = 1e-6);
        assert_relative_eq!(sv.vz, sv_recovered.vz, epsilon = 1e-6);
    }

    /// Test with a near-circular LEO orbit
    #[test]
    fn test_circular_orbit_roundtrip() {
        let r = 6778.0; // ~400 km altitude
        let v = (MU / r).sqrt(); // circular velocity
        let sv = StateVector::new(r, 0.0, 0.0, 0.0, v, 0.0);

        let kep = KeplerianElements::from_state_vector(&sv).unwrap();
        assert!(
            kep.eccentricity < 0.001,
            "Should be near-circular, e = {}",
            kep.eccentricity
        );
        assert_relative_eq!(kep.semi_major_axis, r, epsilon = 1.0);

        let sv_back = kep.to_state_vector();
        assert_relative_eq!(sv.x, sv_back.x, epsilon = 1e-4);
        assert_relative_eq!(sv.vy, sv_back.vy, epsilon = 1e-4);
    }

    /// Test ISS orbital period (~92 minutes)
    #[test]
    fn test_iss_orbital_period() {
        let sv = StateVector::new(6778.0, 0.0, 0.0, 0.0, 7.668, 0.0);
        let kep = KeplerianElements::from_state_vector(&sv).unwrap();
        let period_min = kep.orbital_period() / 60.0;

        assert!(
            (period_min - 92.0).abs() < 3.0,
            "ISS period should be ~92 min, got {} min",
            period_min
        );
    }

    /// Test orbital energy consistency between state vector and Keplerian
    #[test]
    fn test_energy_conservation() {
        let sv = StateVector::new(-2500.0, 6500.0, 1000.0, -5.5, -3.5, 2.0);
        let kep = KeplerianElements::from_state_vector(&sv).unwrap();

        // Energy from state vector: ε = v²/2 - μ/r
        let r = sv.radius();
        let v = sv.speed();
        let energy_sv = v * v / 2.0 - MU / r;

        // Energy from Keplerian: ε = -μ/(2a)
        let energy_kep = kep.orbital_energy();

        assert_relative_eq!(energy_sv, energy_kep, epsilon = 1e-6);
    }

    /// Test Hohmann transfer LEO → GEO
    #[test]
    fn test_hohmann_leo_to_geo() {
        let r_leo = 6578.0; // ~200 km altitude
        let r_geo = 42164.0; // GEO radius

        let (dv1, dv2) = hohmann_transfer(r_leo, r_geo);

        // Well-known values: ~2.46 km/s first burn, ~1.48 km/s second burn
        let total_dv = dv1 + dv2;
        assert!(
            (total_dv - 3.94).abs() < 0.1,
            "LEO→GEO Hohmann total ΔV should be ~3.94 km/s, got {}",
            total_dv
        );
        assert!(dv1 > dv2, "First burn should be larger than second");
    }

    /// Test Hohmann transfer between circular LEO orbits
    #[test]
    fn test_hohmann_small_transfer() {
        let r1 = 6778.0; // 400 km
        let r2 = 6878.0; // 500 km

        let (dv1, dv2) = hohmann_transfer(r1, r2);

        // Small transfer, ΔV should be small
        let total = dv1 + dv2;
        assert!(
            total < 0.1,
            "100 km altitude change should need < 0.1 km/s, got {}",
            total
        );
        assert!(total > 0.01, "Should need some ΔV, got {}", total);
    }

    /// Test collision avoidance maneuver produces non-zero ΔV when needed
    #[test]
    fn test_cam_needed() {
        let state = StateVector::new(7000.0, 0.0, 0.0, 0.0, 7.5, 0.0);
        let threat = StateVector::new(7000.5, 0.0, 0.0, 0.0, -7.5, 0.0);

        let maneuver = collision_avoidance_maneuver(&state, &threat, 5.0, Utc::now());

        assert!(
            maneuver.magnitude() > 0.0,
            "Should need a maneuver for 0.5 km miss with 5 km threshold"
        );
    }

    /// Test collision avoidance maneuver is zero when already safe
    #[test]
    fn test_cam_not_needed() {
        let state = StateVector::new(7000.0, 0.0, 0.0, 0.0, 7.5, 0.0);
        let threat = StateVector::new(7100.0, 0.0, 0.0, 0.0, -7.5, 0.0);

        let maneuver = collision_avoidance_maneuver(&state, &threat, 5.0, Utc::now());

        assert!(
            maneuver.magnitude() < 1e-10,
            "Should not need a maneuver for 100 km miss"
        );
    }

    /// Test impulsive maneuver application
    #[test]
    fn test_maneuver_apply() {
        let sv = StateVector::new(7000.0, 0.0, 0.0, 0.0, 7.5, 0.0);
        let maneuver = ImpulsiveManeuver {
            delta_v: Vector3::new(0.0, 0.1, 0.0),
            time: Utc::now(),
        };

        let post_burn = maneuver.apply(&sv);
        assert_relative_eq!(post_burn.x, 7000.0, epsilon = 1e-10);
        assert_relative_eq!(post_burn.vy, 7.6, epsilon = 1e-10);
    }

    /// Test mean anomaly at periapsis is 0
    #[test]
    fn test_mean_anomaly_at_periapsis() {
        let kep = KeplerianElements {
            semi_major_axis: 7000.0,
            eccentricity: 0.1,
            inclination: 0.5,
            raan: 1.0,
            arg_periapsis: 0.5,
            true_anomaly: 0.0, // At periapsis
        };

        let m = kep.mean_anomaly();
        assert!(
            m.abs() < 1e-10 || (m - 2.0 * std::f64::consts::PI).abs() < 1e-10,
            "Mean anomaly should be ~0 at periapsis, got {}",
            m
        );
    }

    /// Test apoapsis and periapsis altitudes
    #[test]
    fn test_apse_altitudes() {
        let kep = KeplerianElements {
            semi_major_axis: 6778.0,
            eccentricity: 0.01,
            inclination: 0.9,
            raan: 0.0,
            arg_periapsis: 0.0,
            true_anomaly: 0.0,
        };

        let peri = kep.periapsis_altitude();
        let apo = kep.apoapsis_altitude();

        assert!(
            peri > 300.0 && peri < 500.0,
            "Periapsis altitude unexpected: {}",
            peri
        );
        assert!(apo > peri, "Apoapsis should be above periapsis");
        assert!(
            (apo - peri) < 200.0,
            "For e=0.01, apse difference should be small"
        );
    }
}
