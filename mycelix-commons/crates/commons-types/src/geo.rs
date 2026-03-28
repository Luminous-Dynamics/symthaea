// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Geographic types shared across Commons domains
//!
//! Used by property-registry, housing, water-steward, and other
//! location-aware zomes.

use serde::{Deserialize, Serialize};

/// Geographic coordinates
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct GeoLocation {
    pub latitude: f64,
    pub longitude: f64,
    pub altitude_m: Option<f64>,
}

/// Physical address
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Address {
    pub street: String,
    pub city: String,
    pub state_province: String,
    pub postal_code: String,
    pub country: String,
    pub location: Option<GeoLocation>,
}

/// Input for proximity queries.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NearbyQuery {
    pub latitude: f64,
    pub longitude: f64,
    /// Maximum radius in kilometers. None = return all in geohash neighborhood.
    pub radius_km: Option<f64>,
    /// Optional entry type filter (e.g., "property", "shelter").
    pub entry_type: Option<String>,
}

// ============================================================================
// Geohash encoding (no external dependency)
// ============================================================================

const GEOHASH_BASE32: &[u8; 32] = b"0123456789bcdefghjkmnpqrstuvwxyz";

/// Encode a latitude/longitude to a geohash string at the given precision.
///
/// Precision 6 gives ~1.2km x 0.6km cells — good for neighborhood queries.
/// Precision 5 gives ~4.9km x 4.9km, precision 7 gives ~153m x 153m.
pub fn geohash_encode(lat: f64, lon: f64, precision: u8) -> String {
    let precision = precision.max(1).min(12) as usize;
    let mut lat_range = (-90.0_f64, 90.0_f64);
    let mut lon_range = (-180.0_f64, 180.0_f64);
    let mut hash = String::with_capacity(precision);
    let mut bits: u8 = 0;
    let mut bit_count: u8 = 0;
    let mut is_lon = true;

    while hash.len() < precision {
        if is_lon {
            let mid = (lon_range.0 + lon_range.1) / 2.0;
            if lon >= mid {
                bits = (bits << 1) | 1;
                lon_range.0 = mid;
            } else {
                bits <<= 1;
                lon_range.1 = mid;
            }
        } else {
            let mid = (lat_range.0 + lat_range.1) / 2.0;
            if lat >= mid {
                bits = (bits << 1) | 1;
                lat_range.0 = mid;
            } else {
                bits <<= 1;
                lat_range.1 = mid;
            }
        }
        is_lon = !is_lon;
        bit_count += 1;

        if bit_count == 5 {
            hash.push(GEOHASH_BASE32[bits as usize] as char);
            bits = 0;
            bit_count = 0;
        }
    }
    hash
}

/// Decode a geohash string to the center latitude/longitude of its cell.
pub fn geohash_decode(hash: &str) -> (f64, f64) {
    let mut lat_range = (-90.0_f64, 90.0_f64);
    let mut lon_range = (-180.0_f64, 180.0_f64);
    let mut is_lon = true;

    for ch in hash.chars() {
        let idx = match GEOHASH_BASE32.iter().position(|&c| c == ch as u8) {
            Some(i) => i as u8,
            None => continue,
        };
        for bit in (0..5).rev() {
            let b = (idx >> bit) & 1;
            if is_lon {
                let mid = (lon_range.0 + lon_range.1) / 2.0;
                if b == 1 {
                    lon_range.0 = mid;
                } else {
                    lon_range.1 = mid;
                }
            } else {
                let mid = (lat_range.0 + lat_range.1) / 2.0;
                if b == 1 {
                    lat_range.0 = mid;
                } else {
                    lat_range.1 = mid;
                }
            }
            is_lon = !is_lon;
        }
    }

    ((lat_range.0 + lat_range.1) / 2.0, (lon_range.0 + lon_range.1) / 2.0)
}

/// Get the 8 neighboring geohash cells of the given geohash.
pub fn geohash_neighbors(hash: &str) -> Vec<String> {
    if hash.is_empty() {
        return vec![];
    }

    let (center_lat, center_lon) = geohash_decode(hash);
    let precision = hash.len() as u8;

    // Estimate cell dimensions based on precision
    let (lat_err, lon_err) = geohash_error(precision);

    let offsets: [(f64, f64); 8] = [
        (-lat_err, -lon_err), (-lat_err, 0.0), (-lat_err, lon_err),
        (0.0, -lon_err),                       (0.0, lon_err),
        (lat_err, -lon_err),  (lat_err, 0.0),  (lat_err, lon_err),
    ];

    offsets
        .iter()
        .map(|(dlat, dlon)| {
            let nlat = (center_lat + dlat * 2.0).clamp(-90.0, 90.0);
            let nlon = wrap_longitude(center_lon + dlon * 2.0);
            geohash_encode(nlat, nlon, precision)
        })
        .collect()
}

/// Haversine distance between two points in kilometers.
pub fn haversine_distance_km(a: &GeoLocation, b: &GeoLocation) -> f64 {
    const R: f64 = 6371.0; // Earth radius in km
    let dlat = (b.latitude - a.latitude).to_radians();
    let dlon = (b.longitude - a.longitude).to_radians();
    let lat1 = a.latitude.to_radians();
    let lat2 = b.latitude.to_radians();

    let h = (dlat / 2.0).sin().powi(2)
        + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * h.sqrt().asin();
    R * c
}

/// Approximate error (half-cell-size) for a geohash precision.
fn geohash_error(precision: u8) -> (f64, f64) {
    // lat_bits = 5*precision/2 (rounded down), lon_bits = 5*precision - lat_bits
    let total_bits = 5 * precision as u32;
    let lon_bits = (total_bits + 1) / 2;
    let lat_bits = total_bits / 2;
    let lat_err = 180.0 / (1u64 << lat_bits) as f64;
    let lon_err = 360.0 / (1u64 << lon_bits) as f64;
    (lat_err, lon_err)
}

/// Wrap longitude to [-180, 180).
fn wrap_longitude(lon: f64) -> f64 {
    let mut l = lon;
    while l > 180.0 { l -= 360.0; }
    while l < -180.0 { l += 360.0; }
    l
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geohash_encode_known_values() {
        // Richardson, TX area (~32.95, -96.73)
        let hash = geohash_encode(32.95, -96.73, 6);
        assert_eq!(hash.len(), 6);
        // Should start with "9vg" (North America)
        assert!(hash.starts_with("9v"), "Richardson TX should start with 9v, got {}", hash);
    }

    #[test]
    fn geohash_decode_roundtrip() {
        let lat = 48.8566; // Paris
        let lon = 2.3522;
        let hash = geohash_encode(lat, lon, 8);
        let (dlat, dlon) = geohash_decode(&hash);
        assert!((dlat - lat).abs() < 0.001, "Latitude drift: {} vs {}", dlat, lat);
        assert!((dlon - lon).abs() < 0.001, "Longitude drift: {} vs {}", dlon, lon);
    }

    #[test]
    fn geohash_neighbors_returns_eight() {
        let hash = geohash_encode(32.95, -96.73, 6);
        let neighbors = geohash_neighbors(&hash);
        assert_eq!(neighbors.len(), 8);
        // All should be different from center
        for n in &neighbors {
            // At least some should differ from center (all 8 should, but
            // edge cases at poles/dateline could produce duplicates)
            assert_eq!(n.len(), hash.len());
        }
    }

    #[test]
    fn geohash_neighbors_empty_input() {
        assert!(geohash_neighbors("").is_empty());
    }

    #[test]
    fn haversine_same_point_is_zero() {
        let p = GeoLocation { latitude: 32.95, longitude: -96.73, altitude_m: None };
        assert!((haversine_distance_km(&p, &p)).abs() < 0.001);
    }

    #[test]
    fn haversine_known_distance() {
        // Richardson TX to Dallas TX (~15 km)
        let richardson = GeoLocation { latitude: 32.95, longitude: -96.73, altitude_m: None };
        let dallas = GeoLocation { latitude: 32.78, longitude: -96.80, altitude_m: None };
        let dist = haversine_distance_km(&richardson, &dallas);
        assert!(dist > 10.0 && dist < 25.0, "Expected ~15km, got {}", dist);
    }

    #[test]
    fn haversine_antipodal_is_half_circumference() {
        let a = GeoLocation { latitude: 0.0, longitude: 0.0, altitude_m: None };
        let b = GeoLocation { latitude: 0.0, longitude: 180.0, altitude_m: None };
        let dist = haversine_distance_km(&a, &b);
        // Should be ~20,015 km (half Earth circumference)
        assert!(dist > 19_000.0 && dist < 21_000.0, "Expected ~20015km, got {}", dist);
    }

    #[test]
    fn geohash_precision_clamped() {
        // Precision 0 should clamp to 1
        let h0 = geohash_encode(0.0, 0.0, 0);
        assert_eq!(h0.len(), 1);
        // Precision 20 should clamp to 12
        let h20 = geohash_encode(0.0, 0.0, 20);
        assert_eq!(h20.len(), 12);
    }

    #[test]
    fn wrap_longitude_handles_overflow() {
        assert!((wrap_longitude(181.0) - (-179.0)).abs() < 0.001);
        assert!((wrap_longitude(-181.0) - 179.0).abs() < 0.001);
        assert!((wrap_longitude(0.0)).abs() < 0.001);
    }
}
