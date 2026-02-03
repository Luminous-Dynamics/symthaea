//! Orbital State Vectors
//!
//! Represents the complete state of an orbital object: position, velocity,
//! and uncertainty (covariance matrix).

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use crate::covariance::CovarianceMatrix;

/// Reference frame for state vectors
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ReferenceFrame {
    /// True Equator Mean Equinox (J2000)
    TEME,
    /// Earth-Centered Inertial (J2000)
    ECI,
    /// Earth-Centered Earth-Fixed
    ECEF,
    /// International Celestial Reference Frame
    ICRF,
}

impl Default for ReferenceFrame {
    fn default() -> Self {
        ReferenceFrame::TEME  // SGP4 native frame
    }
}

/// 6-element state vector (position + velocity)
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct StateVector {
    /// Position X component (km)
    pub x: f64,
    /// Position Y component (km)
    pub y: f64,
    /// Position Z component (km)
    pub z: f64,
    /// Velocity X component (km/s)
    pub vx: f64,
    /// Velocity Y component (km/s)
    pub vy: f64,
    /// Velocity Z component (km/s)
    pub vz: f64,
}

impl StateVector {
    pub fn new(x: f64, y: f64, z: f64, vx: f64, vy: f64, vz: f64) -> Self {
        Self { x, y, z, vx, vy, vz }
    }

    /// Create from position and velocity vectors
    pub fn from_vectors(position: Vector3<f64>, velocity: Vector3<f64>) -> Self {
        Self {
            x: position.x,
            y: position.y,
            z: position.z,
            vx: velocity.x,
            vy: velocity.y,
            vz: velocity.z,
        }
    }

    /// Get position as 3-vector (km)
    pub fn position(&self) -> Vector3<f64> {
        Vector3::new(self.x, self.y, self.z)
    }

    /// Get velocity as 3-vector (km/s)
    pub fn velocity(&self) -> Vector3<f64> {
        Vector3::new(self.vx, self.vy, self.vz)
    }

    /// Get as 6-element array
    pub fn to_array(&self) -> [f64; 6] {
        [self.x, self.y, self.z, self.vx, self.vy, self.vz]
    }

    /// Distance from Earth center (km)
    pub fn radius(&self) -> f64 {
        self.position().norm()
    }

    /// Speed (km/s)
    pub fn speed(&self) -> f64 {
        self.velocity().norm()
    }

    /// Altitude above Earth surface (km)
    pub fn altitude_km(&self) -> f64 {
        const EARTH_RADIUS: f64 = 6378.137;
        self.radius() - EARTH_RADIUS
    }

    /// Distance between two state vectors (position only, km)
    pub fn distance_to(&self, other: &StateVector) -> f64 {
        (self.position() - other.position()).norm()
    }

    /// Relative velocity between two state vectors (km/s)
    pub fn relative_velocity(&self, other: &StateVector) -> f64 {
        (self.velocity() - other.velocity()).norm()
    }
}

/// Complete orbital state with uncertainty
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OrbitalState {
    /// NORAD catalog ID
    pub norad_id: u32,

    /// Object name (if known)
    pub name: Option<String>,

    /// Epoch time (when this state is valid)
    pub epoch: DateTime<Utc>,

    /// Reference frame
    pub frame: ReferenceFrame,

    /// State vector (position + velocity)
    pub state: StateVector,

    /// Covariance matrix (6x6 uncertainty)
    pub covariance: Option<CovarianceMatrix>,

    /// Data source
    pub source: DataSource,

    /// Quality indicator (0.0 - 1.0)
    pub quality: f32,
}

/// Source of orbital data
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum DataSource {
    /// NORAD/Space-Track TLE
    SpaceTrack,
    /// Commercial provider (LeoLabs, ExoAnalytic, etc.)
    Commercial(String),
    /// Ground-based observation
    GroundObservation {
        sensor_id: String,
        sensor_type: SensorType,
    },
    /// Space-based observation
    SpaceObservation {
        observer_norad_id: u32,
    },
    /// Operator-provided ephemeris
    OperatorEphemeris,
    /// Fused from multiple sources
    Fused {
        source_count: u32,
    },
    /// Mycelix network consensus
    NetworkConsensus {
        observation_count: u32,
        node_count: u32,
    },
}

/// Sensor types for observations
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum SensorType {
    /// Optical telescope
    Optical,
    /// Phased array radar
    Radar,
    /// Radio frequency tracking
    RfTracking,
    /// Laser ranging
    LaserRanging,
    /// Passive RF (signal detection)
    PassiveRf,
}

impl OrbitalState {
    /// Create a new orbital state from a state vector
    pub fn new(
        norad_id: u32,
        epoch: DateTime<Utc>,
        state: StateVector,
        source: DataSource,
    ) -> Self {
        Self {
            norad_id,
            name: None,
            epoch,
            frame: ReferenceFrame::TEME,
            state,
            covariance: None,
            source,
            quality: 1.0,
        }
    }

    /// Add covariance matrix
    pub fn with_covariance(mut self, cov: CovarianceMatrix) -> Self {
        self.covariance = Some(cov);
        self
    }

    /// Add name
    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    /// Get position uncertainty (1-sigma, km)
    pub fn position_uncertainty_km(&self) -> Option<f64> {
        self.covariance.as_ref().map(|c| c.position_sigma())
    }

    /// Get velocity uncertainty (1-sigma, km/s)
    pub fn velocity_uncertainty_kms(&self) -> Option<f64> {
        self.covariance.as_ref().map(|c| c.velocity_sigma())
    }

    /// Check if state has meaningful uncertainty data
    pub fn has_covariance(&self) -> bool {
        self.covariance.is_some()
    }

    /// Age of the state (time since epoch)
    pub fn age(&self, now: DateTime<Utc>) -> chrono::Duration {
        now - self.epoch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_vector_distance() {
        let s1 = StateVector::new(7000.0, 0.0, 0.0, 0.0, 7.5, 0.0);
        let s2 = StateVector::new(7010.0, 0.0, 0.0, 0.0, 7.5, 0.0);

        assert!((s1.distance_to(&s2) - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_altitude() {
        let s = StateVector::new(6778.0, 0.0, 0.0, 0.0, 7.8, 0.0);
        let alt = s.altitude_km();

        // Should be about 400 km
        assert!(alt > 390.0 && alt < 410.0);
    }
}
