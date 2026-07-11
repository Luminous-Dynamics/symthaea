// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Spherical-earth navigation. Coordinates in decimal degrees; distances in km.

/// Mean Earth radius (km).
pub const EARTH_RADIUS_KM: f64 = 6371.0;

/// Great-circle distance between two lat/lon points (haversine formula), km.
pub fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let (phi1, phi2) = (lat1.to_radians(), lat2.to_radians());
    let dphi = (lat2 - lat1).to_radians();
    let dlambda = (lon2 - lon1).to_radians();
    let a = (dphi / 2.0).sin().powi(2) + phi1.cos() * phi2.cos() * (dlambda / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();
    EARTH_RADIUS_KM * c
}

/// Initial bearing (forward azimuth) from point 1 to point 2, in degrees
/// clockwise from true north `[0, 360)`.
pub fn initial_bearing(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let (phi1, phi2) = (lat1.to_radians(), lat2.to_radians());
    let dlambda = (lon2 - lon1).to_radians();
    let y = dlambda.sin() * phi2.cos();
    let x = phi1.cos() * phi2.sin() - phi1.sin() * phi2.cos() * dlambda.cos();
    (y.atan2(x).to_degrees() + 360.0) % 360.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quarter_equator() {
        // (0,0) → (0,90) is a quarter of the equator = πR/2 ≈ 10007.5 km.
        let d = haversine_distance(0.0, 0.0, 0.0, 90.0);
        assert!((d - 10007.543).abs() < 0.1, "d={d}");
    }

    #[test]
    fn london_to_paris() {
        // ≈ 343 km.
        let d = haversine_distance(51.5074, -0.1278, 48.8566, 2.3522);
        assert!((d - 343.5).abs() < 2.0, "d={d}");
    }

    #[test]
    fn bearing_due_east_on_equator() {
        // Heading from (0,0) toward (0,10) is due east = 90°.
        assert!((initial_bearing(0.0, 0.0, 0.0, 10.0) - 90.0).abs() < 1e-6);
        // Toward the north pole is 0°.
        assert!(initial_bearing(0.0, 0.0, 89.0, 0.0).abs() < 1e-6);
    }

    #[test]
    fn distance_is_symmetric() {
        let a = haversine_distance(40.0, -74.0, 34.0, -118.0);
        let b = haversine_distance(34.0, -118.0, 40.0, -74.0);
        assert!((a - b).abs() < 1e-9);
    }
}
