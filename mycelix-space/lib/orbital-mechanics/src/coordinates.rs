//! Coordinate Transformations
//!
//! Converts between different reference frames:
//! - TEME: True Equator Mean Equinox (SGP4 native output)
//! - ECI: Earth-Centered Inertial (J2000)
//! - ECEF: Earth-Centered Earth-Fixed (rotating with Earth)
//! - Geodetic: Latitude, Longitude, Altitude
//!
//! # Why This Matters
//!
//! Different systems use different frames:
//! - SGP4 outputs TEME
//! - CDMs typically use ECI
//! - Ground stations use geodetic
//! - Radar observations use ECEF
//!
//! Proper transformations are essential for accurate conjunction analysis.

use chrono::{DateTime, Utc, Datelike, Timelike};
use nalgebra::{Vector3, Matrix3};

use crate::state::StateVector;

/// WGS-84 Earth constants
pub mod wgs84 {
    /// Semi-major axis (equatorial radius) in km
    pub const A: f64 = 6378.137;

    /// Flattening
    pub const F: f64 = 1.0 / 298.257223563;

    /// Semi-minor axis (polar radius) in km
    pub const B: f64 = A * (1.0 - F);

    /// First eccentricity squared
    pub const E2: f64 = 2.0 * F - F * F;

    /// Second eccentricity squared
    pub const EP2: f64 = (A * A - B * B) / (B * B);

    /// Earth's angular velocity (rad/s)
    pub const OMEGA_EARTH: f64 = 7.292115e-5;

    /// Gravitational parameter (km³/s²)
    pub const MU: f64 = 398600.4418;
}

/// Geodetic coordinates (on Earth's surface)
#[derive(Clone, Debug, PartialEq)]
pub struct GeodeticCoord {
    /// Latitude (degrees, -90 to +90, positive North)
    pub latitude_deg: f64,

    /// Longitude (degrees, -180 to +180, positive East)
    pub longitude_deg: f64,

    /// Altitude above WGS-84 ellipsoid (km)
    pub altitude_km: f64,
}

impl GeodeticCoord {
    pub fn new(lat: f64, lon: f64, alt: f64) -> Self {
        Self {
            latitude_deg: lat,
            longitude_deg: lon,
            altitude_km: alt,
        }
    }

    /// Convert to ECEF position (km)
    pub fn to_ecef(&self) -> Vector3<f64> {
        let lat = self.latitude_deg.to_radians();
        let lon = self.longitude_deg.to_radians();

        let sin_lat = lat.sin();
        let cos_lat = lat.cos();
        let sin_lon = lon.sin();
        let cos_lon = lon.cos();

        // Radius of curvature in prime vertical
        let n = wgs84::A / (1.0 - wgs84::E2 * sin_lat * sin_lat).sqrt();

        let x = (n + self.altitude_km) * cos_lat * cos_lon;
        let y = (n + self.altitude_km) * cos_lat * sin_lon;
        let z = (n * (1.0 - wgs84::E2) + self.altitude_km) * sin_lat;

        Vector3::new(x, y, z)
    }

    /// Create from ECEF position
    pub fn from_ecef(ecef: &Vector3<f64>) -> Self {
        let x = ecef.x;
        let y = ecef.y;
        let z = ecef.z;

        // Longitude is straightforward
        let lon = y.atan2(x);

        // Latitude requires iteration (Bowring's method)
        let p = (x * x + y * y).sqrt();
        let mut lat = z.atan2(p * (1.0 - wgs84::E2));

        // Iterate for more accurate latitude
        for _ in 0..5 {
            let sin_lat = lat.sin();
            let n = wgs84::A / (1.0 - wgs84::E2 * sin_lat * sin_lat).sqrt();
            lat = z.atan2(p - wgs84::E2 * n * lat.cos());
        }

        // Calculate altitude
        let sin_lat = lat.sin();
        let cos_lat = lat.cos();
        let n = wgs84::A / (1.0 - wgs84::E2 * sin_lat * sin_lat).sqrt();

        let alt = if cos_lat.abs() > 1e-10 {
            p / cos_lat - n
        } else {
            z.abs() / sin_lat.abs() - n * (1.0 - wgs84::E2)
        };

        Self {
            latitude_deg: lat.to_degrees(),
            longitude_deg: lon.to_degrees(),
            altitude_km: alt,
        }
    }
}

/// Greenwich Mean Sidereal Time (GMST) in radians
pub fn gmst(time: DateTime<Utc>) -> f64 {
    // Julian date at midnight
    let year = time.year() as f64;
    let month = time.month() as f64;
    let day = time.day() as f64;

    let jd_midnight = julian_date(year as i32, month as u32, day as u32, 0.0);

    // Centuries from J2000
    let t = (jd_midnight - 2451545.0) / 36525.0;

    // GMST at midnight (degrees)
    let gmst_midnight = 100.4606184 + 36000.77004 * t + 0.000387933 * t * t;

    // Add time of day
    let hours = time.hour() as f64
        + time.minute() as f64 / 60.0
        + time.second() as f64 / 3600.0
        + time.nanosecond() as f64 / 3600.0e9;

    let gmst_deg = gmst_midnight + hours * 15.04106864;

    // Normalize to 0-360
    let gmst_normalized = gmst_deg.rem_euclid(360.0);

    gmst_normalized.to_radians()
}

/// Compute Julian Date
pub fn julian_date(year: i32, month: u32, day: u32, ut_hours: f64) -> f64 {
    let y = if month <= 2 { year - 1 } else { year } as f64;
    let m = if month <= 2 { month + 12 } else { month } as f64;

    let a = (y / 100.0).floor();
    let b = 2.0 - a + (a / 4.0).floor();

    (365.25 * (y + 4716.0)).floor()
        + (30.6001 * (m + 1.0)).floor()
        + day as f64
        + b
        - 1524.5
        + ut_hours / 24.0
}

/// Convert TEME to ECEF at given time
/// Accounts for Earth rotation only (ignores polar motion and nutation)
pub fn teme_to_ecef(state: &StateVector, time: DateTime<Utc>) -> StateVector {
    let theta = gmst(time);

    let rot = rotation_z(-theta);

    let pos = rot * state.position();

    // Velocity needs correction for Earth rotation
    let omega = Vector3::new(0.0, 0.0, wgs84::OMEGA_EARTH);
    let vel = rot * state.velocity() - omega.cross(&pos);

    StateVector::from_vectors(pos, vel)
}

/// Convert ECEF to TEME at given time
pub fn ecef_to_teme(state: &StateVector, time: DateTime<Utc>) -> StateVector {
    let theta = gmst(time);

    let rot = rotation_z(theta);

    let pos = rot * state.position();

    // Velocity correction for Earth rotation
    let omega = Vector3::new(0.0, 0.0, wgs84::OMEGA_EARTH);
    let pos_ecef = state.position();
    let vel = rot * (state.velocity() + omega.cross(&pos_ecef));

    StateVector::from_vectors(pos, vel)
}

/// Convert TEME to ECI (J2000)
/// Note: Full conversion requires nutation and precession matrices
/// This simplified version ignores small corrections (<0.1 deg)
pub fn teme_to_eci(state: &StateVector, _time: DateTime<Utc>) -> StateVector {
    // For most conjunction analysis, TEME ≈ ECI is acceptable
    // Full transformation would require IAU-76/FK5 precession-nutation
    state.clone()
}

/// Convert ECI to TEME
pub fn eci_to_teme(state: &StateVector, _time: DateTime<Utc>) -> StateVector {
    // Simplified (inverse of above)
    state.clone()
}

/// Convert state from TEME to geodetic sub-satellite point
pub fn subsatellite_point(state: &StateVector, time: DateTime<Utc>) -> GeodeticCoord {
    let ecef = teme_to_ecef(state, time);
    GeodeticCoord::from_ecef(&ecef.position())
}

/// Rotation matrix around Z-axis
fn rotation_z(angle: f64) -> Matrix3<f64> {
    let c = angle.cos();
    let s = angle.sin();

    Matrix3::new(
        c, s, 0.0,
        -s, c, 0.0,
        0.0, 0.0, 1.0,
    )
}

/// Calculate azimuth and elevation from ground station to satellite
pub fn look_angles(
    station: &GeodeticCoord,
    satellite_ecef: &Vector3<f64>,
) -> (f64, f64, f64) {  // (azimuth_deg, elevation_deg, range_km)
    let station_ecef = station.to_ecef();
    let range_vec = satellite_ecef - station_ecef;
    let range = range_vec.norm();

    // Convert to topocentric (East-North-Up)
    let lat = station.latitude_deg.to_radians();
    let lon = station.longitude_deg.to_radians();

    let sin_lat = lat.sin();
    let cos_lat = lat.cos();
    let sin_lon = lon.sin();
    let cos_lon = lon.cos();

    // Rotation matrix from ECEF to ENU
    let rot = Matrix3::new(
        -sin_lon, cos_lon, 0.0,
        -sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat,
        cos_lat * cos_lon, cos_lat * sin_lon, sin_lat,
    );

    let enu = rot * range_vec;
    let east = enu.x;
    let north = enu.y;
    let up = enu.z;

    // Azimuth (from North, clockwise)
    let azimuth = east.atan2(north).to_degrees().rem_euclid(360.0);

    // Elevation (from horizontal)
    let horizontal = (east * east + north * north).sqrt();
    let elevation = up.atan2(horizontal).to_degrees();

    (azimuth, elevation, range)
}

/// Check if satellite is visible from ground station
pub fn is_visible(
    station: &GeodeticCoord,
    satellite_ecef: &Vector3<f64>,
    min_elevation_deg: f64,
) -> bool {
    let (_, elevation, _) = look_angles(station, satellite_ecef);
    elevation >= min_elevation_deg
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_geodetic_to_ecef_equator() {
        // Point on equator at prime meridian, sea level
        let geo = GeodeticCoord::new(0.0, 0.0, 0.0);
        let ecef = geo.to_ecef();

        // Should be at equatorial radius on x-axis
        assert_relative_eq!(ecef.x, wgs84::A, epsilon = 0.01);
        assert_relative_eq!(ecef.y, 0.0, epsilon = 0.01);
        assert_relative_eq!(ecef.z, 0.0, epsilon = 0.01);
    }

    #[test]
    fn test_geodetic_to_ecef_pole() {
        // North pole, sea level
        let geo = GeodeticCoord::new(90.0, 0.0, 0.0);
        let ecef = geo.to_ecef();

        // Should be at polar radius on z-axis
        assert_relative_eq!(ecef.x, 0.0, epsilon = 0.01);
        assert_relative_eq!(ecef.y, 0.0, epsilon = 0.01);
        assert_relative_eq!(ecef.z, wgs84::B, epsilon = 0.01);
    }

    #[test]
    fn test_ecef_geodetic_roundtrip() {
        let original = GeodeticCoord::new(45.0, -75.0, 100.0);
        let ecef = original.to_ecef();
        let recovered = GeodeticCoord::from_ecef(&ecef);

        assert_relative_eq!(original.latitude_deg, recovered.latitude_deg, epsilon = 0.0001);
        assert_relative_eq!(original.longitude_deg, recovered.longitude_deg, epsilon = 0.0001);
        assert_relative_eq!(original.altitude_km, recovered.altitude_km, epsilon = 0.001);
    }

    #[test]
    fn test_teme_ecef_roundtrip() {
        let state = StateVector::new(7000.0, 0.0, 0.0, 0.0, 7.5, 0.0);
        let time = Utc::now();

        let ecef = teme_to_ecef(&state, time);
        let recovered = ecef_to_teme(&ecef, time);

        assert_relative_eq!(state.x, recovered.x, epsilon = 0.01);
        assert_relative_eq!(state.y, recovered.y, epsilon = 0.01);
        assert_relative_eq!(state.z, recovered.z, epsilon = 0.01);
    }

    #[test]
    fn test_look_angles_overhead() {
        // Satellite directly overhead
        let station = GeodeticCoord::new(0.0, 0.0, 0.0);
        let satellite = station.to_ecef() + Vector3::new(400.0, 0.0, 0.0);

        let (az, el, range) = look_angles(&station, &satellite);

        // Elevation should be ~90 degrees (directly overhead)
        // Range should be ~400 km
        assert!(el > 80.0, "Elevation should be near 90 deg, got {}", el);
        assert_relative_eq!(range, 400.0, epsilon = 1.0);
    }

    #[test]
    fn test_visibility() {
        let station = GeodeticCoord::new(0.0, 0.0, 0.0);

        // Satellite overhead
        let overhead = station.to_ecef() + Vector3::new(400.0, 0.0, 0.0);
        assert!(is_visible(&station, &overhead, 10.0));

        // Satellite on opposite side of Earth
        let opposite = Vector3::new(-wgs84::A - 400.0, 0.0, 0.0);
        assert!(!is_visible(&station, &opposite, 10.0));
    }
}
