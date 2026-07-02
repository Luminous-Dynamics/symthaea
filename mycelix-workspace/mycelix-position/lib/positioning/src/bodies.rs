// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Celestial body definitions for body-agnostic positioning.
//!
//! The same trilateration and EKF algorithms work on any body —
//! only the coordinate transform (geodetic ↔ Cartesian) changes.

use serde::{Deserialize, Serialize};

/// Trait for celestial bodies providing geodetic ↔ Cartesian transforms.
pub trait CelestialBody {
    fn name(&self) -> &str;
    fn equatorial_radius_m(&self) -> f64;
    fn polar_radius_m(&self) -> f64;
    fn flattening(&self) -> f64 {
        1.0 - self.polar_radius_m() / self.equatorial_radius_m()
    }
    fn eccentricity_sq(&self) -> f64 {
        let f = self.flattening();
        2.0 * f - f * f
    }

    /// Convert geodetic (lat°, lon°, alt_m) to body-fixed Cartesian (x, y, z) meters.
    fn geodetic_to_cartesian(&self, lat_deg: f64, lon_deg: f64, alt_m: f64) -> [f64; 3] {
        let lat = lat_deg.to_radians();
        let lon = lon_deg.to_radians();
        let a = self.equatorial_radius_m();
        let e2 = self.eccentricity_sq();
        let sin_lat = lat.sin();
        let cos_lat = lat.cos();
        let n = a / (1.0 - e2 * sin_lat * sin_lat).sqrt();
        [
            (n + alt_m) * cos_lat * lon.cos(),
            (n + alt_m) * cos_lat * lon.sin(),
            (n * (1.0 - e2) + alt_m) * sin_lat,
        ]
    }

    /// Convert body-fixed Cartesian (x, y, z) meters to geodetic (lat°, lon°, alt_m).
    /// Uses Bowring's iterative method (converges in 2-3 iterations).
    fn cartesian_to_geodetic(&self, xyz: [f64; 3]) -> (f64, f64, f64) {
        let [x, y, z] = xyz;
        let a = self.equatorial_radius_m();
        let e2 = self.eccentricity_sq();
        let lon = y.atan2(x);
        let p = (x * x + y * y).sqrt();

        // Bowring's iterative method
        let mut lat = z.atan2(p * (1.0 - e2));
        for _ in 0..10 {
            let sin_lat = lat.sin();
            let n = a / (1.0 - e2 * sin_lat * sin_lat).sqrt();
            lat = z.atan2(p - e2 * n * lat.cos() / (n + 0.0).max(1.0));
            // Simplified: use direct formula
            let new_lat = (z + e2 * n * sin_lat).atan2(p);
            if (new_lat - lat).abs() < 1e-12 {
                lat = new_lat;
                break;
            }
            lat = new_lat;
        }
        let sin_lat = lat.sin();
        let n = a / (1.0 - e2 * sin_lat * sin_lat).sqrt();
        let alt = if lat.cos().abs() > 1e-10 {
            p / lat.cos() - n
        } else {
            z.abs() / sin_lat.abs() - n * (1.0 - e2)
        };
        (lat.to_degrees(), lon.to_degrees(), alt)
    }
}

// ============================================================================
// EARTH (WGS-84)
// ============================================================================

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Earth;

impl CelestialBody for Earth {
    fn name(&self) -> &str {
        "Earth"
    }
    fn equatorial_radius_m(&self) -> f64 {
        6_378_137.0
    }
    fn polar_radius_m(&self) -> f64 {
        6_356_752.314245
    }
}

// ============================================================================
// MOON (IAU sphere, slight triaxial ignored)
// ============================================================================

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Moon;

impl CelestialBody for Moon {
    fn name(&self) -> &str {
        "Moon"
    }
    fn equatorial_radius_m(&self) -> f64 {
        1_737_400.0
    }
    fn polar_radius_m(&self) -> f64 {
        1_737_400.0
    } // Sphere approximation
}

// ============================================================================
// MARS (IAU 2000 ellipsoid)
// ============================================================================

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Mars;

impl CelestialBody for Mars {
    fn name(&self) -> &str {
        "Mars"
    }
    fn equatorial_radius_m(&self) -> f64 {
        3_396_190.0
    }
    fn polar_radius_m(&self) -> f64 {
        3_376_200.0
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn earth_wgs84_constants() {
        let e = Earth;
        assert!((e.flattening() - 1.0 / 298.257223563).abs() < 1e-10);
    }

    #[test]
    fn moon_is_spherical() {
        let m = Moon;
        assert!((m.flattening()).abs() < 1e-10);
        assert!((m.eccentricity_sq()).abs() < 1e-10);
    }

    #[test]
    fn earth_geodetic_cartesian_roundtrip() {
        let e = Earth;
        let cases = [
            (0.0, 0.0, 0.0),       // Equator, prime meridian
            (90.0, 0.0, 0.0),      // North pole
            (-33.9, 18.4, 100.0),  // Cape Town
            (28.6, -80.6, 5.0),    // Cape Canaveral
            (-1.0, -78.5, 2800.0), // Ecuador highlands (spaceport!)
        ];
        for (lat, lon, alt) in cases {
            let xyz = e.geodetic_to_cartesian(lat, lon, alt);
            let (lat2, lon2, alt2) = e.cartesian_to_geodetic(xyz);
            assert!((lat2 - lat).abs() < 1e-6, "lat: {} vs {}", lat, lat2);
            assert!((lon2 - lon).abs() < 1e-6, "lon: {} vs {}", lon, lon2);
            assert!((alt2 - alt).abs() < 0.01, "alt: {} vs {}", alt, alt2);
        }
    }

    #[test]
    fn moon_geodetic_cartesian_roundtrip() {
        let m = Moon;
        let xyz = m.geodetic_to_cartesian(0.0, 0.0, 0.0);
        assert!((xyz[0] - 1_737_400.0).abs() < 1.0);
        let (lat, lon, alt) = m.cartesian_to_geodetic(xyz);
        assert!(lat.abs() < 1e-6);
        assert!(lon.abs() < 1e-6);
        assert!(alt.abs() < 1.0);
    }

    #[test]
    fn mars_has_nonzero_flattening() {
        let m = Mars;
        assert!(m.flattening() > 0.005);
        assert!(m.flattening() < 0.01);
    }
}
