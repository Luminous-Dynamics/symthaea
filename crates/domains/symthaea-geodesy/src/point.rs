// Copyright (C) 2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validated geographic coordinates for the spherical geodesy API.
//!
//! This type validates coordinate representation, not datum accuracy or
//! physical geolocation. Distance calculations remain spherical-Earth
//! estimates, not WGS84 ellipsoidal inverse geodesics.

use crate::sphere::haversine_distance;
use std::error::Error;
use std::fmt;

/// Failure to construct a GeoPoint from untrusted coordinate values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeoPointError {
    /// Latitude was NaN or infinite.
    NonFiniteLatitude,
    /// Longitude was NaN or infinite.
    NonFiniteLongitude,
    /// Latitude was outside the closed interval [-90, 90] degrees.
    LatitudeOutOfRange,
    /// Longitude was outside the closed interval [-180, 180] degrees.
    LongitudeOutOfRange,
}

impl fmt::Display for GeoPointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteLatitude => f.write_str("latitude must be finite"),
            Self::NonFiniteLongitude => f.write_str("longitude must be finite"),
            Self::LatitudeOutOfRange => {
                f.write_str("latitude must be in the inclusive range [-90, 90] degrees")
            }
            Self::LongitudeOutOfRange => {
                f.write_str("longitude must be in the inclusive range [-180, 180] degrees")
            }
        }
    }
}

impl Error for GeoPointError {}

/// A latitude/longitude pair validated to conventional degree ranges.
///
/// Fields are private so a successfully constructed point cannot be mutated
/// into an invalid coordinate. Longitude is range-checked, not normalized:
/// callers retain the coordinate they supplied, including either dateline
/// representation (-180 or +180).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeoPoint {
    latitude_degrees: f64,
    longitude_degrees: f64,
}

impl GeoPoint {
    /// Construct a point from latitude/longitude in decimal degrees.
    ///
    /// Latitude must be finite and in [-90, 90]; longitude must be finite
    /// and in [-180, 180]. Invalid values return an error rather than being
    /// clamped or silently normalized.
    pub fn new(
        latitude_degrees: f64,
        longitude_degrees: f64,
    ) -> Result<Self, GeoPointError> {
        if !latitude_degrees.is_finite() {
            return Err(GeoPointError::NonFiniteLatitude);
        }
        if !longitude_degrees.is_finite() {
            return Err(GeoPointError::NonFiniteLongitude);
        }
        if !(-90.0..=90.0).contains(&latitude_degrees) {
            return Err(GeoPointError::LatitudeOutOfRange);
        }
        if !(-180.0..=180.0).contains(&longitude_degrees) {
            return Err(GeoPointError::LongitudeOutOfRange);
        }

        Ok(Self {
            latitude_degrees,
            longitude_degrees,
        })
    }

    /// Latitude in decimal degrees, guaranteed to be finite and in [-90, 90].
    pub fn latitude_degrees(self) -> f64 {
        self.latitude_degrees
    }

    /// Longitude in decimal degrees, guaranteed to be finite and in [-180, 180].
    pub fn longitude_degrees(self) -> f64 {
        self.longitude_degrees
    }

    /// Spherical great-circle distance to another validated point, in km.
    ///
    /// This is an O(1) Haversine calculation using the crate's declared mean
    /// Earth radius. It is not a WGS84 ellipsoidal distance and provides no
    /// altitude, datum, or geolocation-uncertainty semantics.
    pub fn distance_to(self, other: Self) -> f64 {
        haversine_distance(
            self.latitude_degrees,
            self.longitude_degrees,
            other.latitude_degrees,
            other.longitude_degrees,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_inclusive_coordinate_boundaries() {
        assert!(GeoPoint::new(-90.0, -180.0).is_ok());
        assert!(GeoPoint::new(90.0, 180.0).is_ok());
        assert!(GeoPoint::new(0.0, 0.0).is_ok());
    }

    #[test]
    fn rejects_non_finite_coordinates() {
        assert_eq!(
            GeoPoint::new(f64::NAN, 0.0),
            Err(GeoPointError::NonFiniteLatitude)
        );
        assert_eq!(
            GeoPoint::new(0.0, f64::INFINITY),
            Err(GeoPointError::NonFiniteLongitude)
        );
        assert_eq!(
            GeoPoint::new(f64::NEG_INFINITY, 0.0),
            Err(GeoPointError::NonFiniteLatitude)
        );
    }

    #[test]
    fn rejects_out_of_range_coordinates_without_normalizing() {
        assert_eq!(
            GeoPoint::new(90.000_001, 0.0),
            Err(GeoPointError::LatitudeOutOfRange)
        );
        assert_eq!(
            GeoPoint::new(0.0, 180.000_001),
            Err(GeoPointError::LongitudeOutOfRange)
        );
        assert_eq!(
            GeoPoint::new(0.0, -180.000_001),
            Err(GeoPointError::LongitudeOutOfRange)
        );
    }

    #[test]
    fn validated_points_preserve_short_dateline_crossing() {
        let east = GeoPoint::new(0.0, 179.9).expect("valid point");
        let west = GeoPoint::new(0.0, -179.9).expect("valid point");
        let distance = east.distance_to(west);

        assert!(distance.is_finite());
        assert!((distance - 22.239).abs() < 0.1, "distance={distance}");
        assert!((distance - west.distance_to(east)).abs() < 1e-9);
    }

    #[test]
    fn validated_nearly_antipodal_distance_remains_finite() {
        let a = GeoPoint::new(-65.33345275784633, -60.105346073003815)
            .expect("valid point");
        let b = GeoPoint::new(65.33345223272934, 119.89465306676738)
            .expect("valid point");
        let distance = a.distance_to(b);

        assert!(distance.is_finite());
        assert!(
            (distance - std::f64::consts::PI * crate::sphere::EARTH_RADIUS_KM).abs() < 1e-6
        );
    }
}
