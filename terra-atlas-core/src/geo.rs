// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Geodetic coordinate math: lat/lon to 3D, marker sizing, arc heights.

use std::f64::consts::PI;

/// Convert latitude/longitude (degrees) to 3D position on a unit-radius sphere,
/// scaled by `radius`. Y-up, matching WebGL/Bevy convention.
///
/// Matches the original TerraGlobeWithSites.tsx coordinate system exactly.
pub fn lat_lon_to_xyz(lat: f64, lon: f64, radius: f64) -> [f32; 3] {
    let phi = (90.0 - lat) * (PI / 180.0);
    let theta = (lon + 180.0) * (PI / 180.0);
    [
        -(radius * phi.sin() * theta.cos()) as f32,
        (radius * phi.cos()) as f32,
        (radius * phi.sin() * theta.sin()) as f32,
    ]
}

/// Inverse: 3D position back to (lat, lon) in degrees.
/// Assumes the position is on a sphere of given radius.
pub fn xyz_to_lat_lon(pos: [f32; 3], _radius: f64) -> (f64, f64) {
    let x = pos[0] as f64;
    let y = pos[1] as f64;
    let z = pos[2] as f64;
    let r = (x * x + y * y + z * z).sqrt().max(1e-10);
    let lat = 90.0 - (y / r).acos() * (180.0 / PI);
    let lon = z.atan2(-x) * (180.0 / PI) - 180.0;
    // Normalize longitude to [-180, 180]
    let lon = if lon < -180.0 { lon + 360.0 } else if lon > 180.0 { lon - 360.0 } else { lon };
    (lat, lon)
}

/// Marker size from capacity (MW). Logarithmic scaling, clamped.
pub fn marker_size_from_capacity(capacity_mw: f64) -> f32 {
    let size = (capacity_mw + 1.0).ln() as f32 * 0.003;
    size.clamp(0.012, 0.030)
}

/// Peak height for great-circle arc elevation above the globe surface.
/// Short arcs hug the surface; long arcs arc higher.
pub fn arc_peak_height(distance_km: f64) -> f32 {
    (0.015 + distance_km / 200_000.0) as f32
}

/// Haversine distance in km between two lat/lon points.
pub fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const EARTH_RADIUS_KM: f64 = 6371.0;
    let d_lat = (lat2 - lat1).to_radians();
    let d_lon = (lon2 - lon1).to_radians();
    let a = (d_lat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (d_lon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();
    EARTH_RADIUS_KM * c
}

/// Carbon emission halo radius from annual production and fuel type.
/// Log-scaled, clamped. Returns a globe-relative radius for the translucent red sphere.
pub fn emission_halo_radius(annual_production_mboe: f64, fuel_type: &crate::types::FuelType) -> f32 {
    let co2_mt = annual_production_mboe * crate::economics::emission_factor(fuel_type);
    let radius = (co2_mt + 1.0).ln() as f32 * 0.015;
    radius.clamp(0.02, 0.12)
}

/// Marker size from proven reserves (million barrels oil equivalent). Logarithmic, clamped.
pub fn marker_size_from_reserves(reserves_mboe: f64) -> f32 {
    let size = (reserves_mboe + 1.0).ln() as f32 * 0.003;
    size.clamp(0.012, 0.030)
}

/// Emissive multiplier based on fossil deposit status (transition narrative).
pub fn fossil_emissive_factor(status: &str) -> f32 {
    match status {
        "producing" => 1.5,
        "declining" => 0.8,
        "depleted" => 0.3,
        "undeveloped" => 0.5,
        _ => 1.0,
    }
}

/// Scale multiplier based on fossil deposit status.
pub fn fossil_scale_factor(status: &str) -> f32 {
    match status {
        "depleted" => 0.7,
        _ => 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lat_lon_roundtrip() {
        let cases = [
            (0.0, 0.0),
            (45.0, 90.0),
            (-33.9, 18.4),   // Cape Town
            (40.7, -74.0),   // NYC
            (90.0, 0.0),     // North pole
            (-90.0, 0.0),    // South pole
        ];
        for (lat, lon) in cases {
            let pos = lat_lon_to_xyz(lat, lon, 1.0);
            let (rlat, rlon) = xyz_to_lat_lon(pos, 1.0);
            assert!(
                (rlat - lat).abs() < 0.1 && (rlon - lon).abs() < 0.1,
                "roundtrip failed for ({lat}, {lon}): got ({rlat}, {rlon})"
            );
        }
    }

    #[test]
    fn test_north_pole_is_y_up() {
        let pos = lat_lon_to_xyz(90.0, 0.0, 1.0);
        assert!((pos[1] - 1.0).abs() < 1e-5, "north pole should be Y=1");
        assert!(pos[0].abs() < 1e-5);
        assert!(pos[2].abs() < 1e-5);
    }

    #[test]
    fn test_marker_size_bounds() {
        assert_eq!(marker_size_from_capacity(0.0), 0.012); // below min clamp
        assert_eq!(marker_size_from_capacity(1e6), 0.030); // above max clamp
    }

    #[test]
    fn test_haversine_known_distance() {
        // London to Paris ~ 344 km
        let d = haversine_km(51.5, -0.12, 48.85, 2.35);
        assert!((d - 344.0).abs() < 5.0, "London-Paris: {d} km");
    }

    #[test]
    fn test_emission_halo_radius_bounds() {
        use crate::types::FuelType;
        let small = emission_halo_radius(1.0, &FuelType::Gas);
        let large = emission_halo_radius(5000.0, &FuelType::Coal);
        assert!(small >= 0.02);
        assert!(large <= 0.12);
        assert!(large > small);
    }

    #[test]
    fn test_marker_size_from_reserves_bounds() {
        assert_eq!(marker_size_from_reserves(0.0), 0.012);
        assert_eq!(marker_size_from_reserves(1e9), 0.030);
        let mid = marker_size_from_reserves(1000.0);
        assert!(mid > 0.010 && mid < 0.030);
    }

    #[test]
    fn test_fossil_emissive_factors() {
        assert_eq!(fossil_emissive_factor("producing"), 1.5);
        assert_eq!(fossil_emissive_factor("declining"), 0.8);
        assert_eq!(fossil_emissive_factor("depleted"), 0.3);
        assert_eq!(fossil_emissive_factor("undeveloped"), 0.5);
        assert_eq!(fossil_emissive_factor("unknown"), 1.0);
    }

    #[test]
    fn test_fossil_scale_factors() {
        assert_eq!(fossil_scale_factor("depleted"), 0.7);
        assert_eq!(fossil_scale_factor("producing"), 1.0);
    }

    #[test]
    fn test_arc_peak_height_increases_with_distance() {
        let short = arc_peak_height(100.0);
        let long = arc_peak_height(10_000.0);
        assert!(long > short);
    }
}
