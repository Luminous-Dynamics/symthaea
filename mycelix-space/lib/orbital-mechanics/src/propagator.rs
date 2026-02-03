//! Orbital Propagator
//!
//! Wraps SGP4/SDP4 for propagating TLE orbits forward/backward in time.
//! SGP4 is the standard NORAD algorithm for near-Earth objects.
//! SDP4 is used for deep-space objects (period > 225 minutes).
//!
//! # Usage
//! ```ignore
//! let tle = TwoLineElement::parse(tle_string)?;
//! let propagator = Propagator::from_tle(&tle)?;
//! let state = propagator.propagate_to(future_time)?;
//! ```

use chrono::{DateTime, Utc, Duration};
use thiserror::Error;
use crate::tle::TwoLineElement;
use crate::state::{OrbitalState, StateVector, DataSource, ReferenceFrame};
use crate::covariance::CovarianceMatrix;

#[derive(Error, Debug)]
pub enum PropagationError {
    #[error("SGP4 initialization failed: {0}")]
    InitializationFailed(String),

    #[error("Propagation failed at time {time}: {message}")]
    PropagationFailed {
        time: String,
        message: String,
    },

    #[error("Object has decayed (below Earth surface)")]
    ObjectDecayed,

    #[error("Propagation time out of bounds: {0} days from epoch")]
    TimeOutOfBounds(f64),
}

/// Orbital propagator using SGP4/SDP4 algorithms
pub struct Propagator {
    /// The underlying SGP4 constants (from sgp4 crate)
    elements: sgp4::Elements,

    /// Original TLE for reference
    tle: TwoLineElement,

    /// Maximum propagation time from epoch (days)
    /// Default 30 days forward, 7 days backward
    max_forward_days: f64,
    max_backward_days: f64,
}

impl Propagator {
    /// Create a propagator from a TLE
    pub fn from_tle(tle: &TwoLineElement) -> Result<Self, PropagationError> {
        // Parse TLE into SGP4 elements
        let elements = sgp4::Elements::from_tle(
            tle.name.clone(),
            tle.line1.as_bytes(),
            tle.line2.as_bytes(),
        ).map_err(|e| PropagationError::InitializationFailed(format!("{:?}", e)))?;

        Ok(Self {
            elements,
            tle: tle.clone(),
            max_forward_days: 30.0,
            max_backward_days: 7.0,
        })
    }

    /// Set maximum propagation bounds
    pub fn with_bounds(mut self, forward_days: f64, backward_days: f64) -> Self {
        self.max_forward_days = forward_days;
        self.max_backward_days = backward_days;
        self
    }

    /// Get the TLE epoch
    pub fn epoch(&self) -> DateTime<Utc> {
        self.tle.epoch
    }

    /// Get the NORAD ID
    pub fn norad_id(&self) -> u32 {
        self.tle.norad_id
    }

    /// Propagate to a specific time
    pub fn propagate_to(&self, time: DateTime<Utc>) -> Result<OrbitalState, PropagationError> {
        let minutes_since_epoch = (time - self.tle.epoch).num_seconds() as f64 / 60.0;
        let days_since_epoch = minutes_since_epoch / 1440.0;

        // Check bounds
        if days_since_epoch > self.max_forward_days {
            return Err(PropagationError::TimeOutOfBounds(days_since_epoch));
        }
        if days_since_epoch < -self.max_backward_days {
            return Err(PropagationError::TimeOutOfBounds(days_since_epoch));
        }

        // Create SGP4 constants (WGS84)
        let constants = sgp4::Constants::from_elements(&self.elements)
            .map_err(|e| PropagationError::InitializationFailed(format!("{:?}", e)))?;

        // Propagate
        let prediction = constants.propagate(minutes_since_epoch)
            .map_err(|e| PropagationError::PropagationFailed {
                time: time.to_rfc3339(),
                message: format!("{:?}", e),
            })?;

        // Extract position and velocity (in km and km/s)
        let state = StateVector::new(
            prediction.position[0],
            prediction.position[1],
            prediction.position[2],
            prediction.velocity[0],
            prediction.velocity[1],
            prediction.velocity[2],
        );

        // Check for decay
        if state.altitude_km() < 0.0 {
            return Err(PropagationError::ObjectDecayed);
        }

        // Estimate covariance based on TLE age
        let age_hours = days_since_epoch.abs() * 24.0;
        let covariance = CovarianceMatrix::from_tle_age(age_hours);

        Ok(OrbitalState {
            norad_id: self.tle.norad_id,
            name: self.tle.name.clone(),
            epoch: time,
            frame: ReferenceFrame::TEME,  // SGP4 outputs TEME
            state,
            covariance: Some(covariance),
            source: DataSource::SpaceTrack,
            quality: Self::estimate_quality(days_since_epoch),
        })
    }

    /// Propagate for a duration from epoch
    pub fn propagate_minutes(&self, minutes: f64) -> Result<OrbitalState, PropagationError> {
        let time = self.tle.epoch + Duration::seconds((minutes * 60.0) as i64);
        self.propagate_to(time)
    }

    /// Generate ephemeris (series of states) over a time range
    pub fn ephemeris(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        step_seconds: i64,
    ) -> Vec<Result<OrbitalState, PropagationError>> {
        let mut results = Vec::new();
        let mut current = start;

        while current <= end {
            results.push(self.propagate_to(current));
            current = current + Duration::seconds(step_seconds);
        }

        results
    }

    /// Find the next time the object crosses a given altitude
    /// Useful for pass prediction, atmospheric entry, etc.
    pub fn find_altitude_crossing(
        &self,
        start: DateTime<Utc>,
        target_altitude_km: f64,
        max_search_hours: f64,
        rising: bool,
    ) -> Option<DateTime<Utc>> {
        let step_seconds = 60;  // 1-minute steps
        let max_steps = (max_search_hours * 3600.0 / step_seconds as f64) as usize;

        let mut prev_alt = None;

        for i in 0..max_steps {
            let time = start + Duration::seconds(i as i64 * step_seconds);

            if let Ok(state) = self.propagate_to(time) {
                let alt = state.state.altitude_km();

                if let Some(prev) = prev_alt {
                    let crossed = if rising {
                        prev < target_altitude_km && alt >= target_altitude_km
                    } else {
                        prev > target_altitude_km && alt <= target_altitude_km
                    };

                    if crossed {
                        // Linear interpolation for better accuracy
                        let frac = (target_altitude_km - prev) / (alt - prev);
                        let precise_time = time - Duration::seconds(step_seconds)
                            + Duration::milliseconds((frac * step_seconds as f64 * 1000.0) as i64);
                        return Some(precise_time);
                    }
                }

                prev_alt = Some(alt);
            }
        }

        None
    }

    /// Estimate quality score based on propagation distance from epoch
    fn estimate_quality(days_from_epoch: f64) -> f32 {
        // Quality degrades with distance from epoch
        // Fresh TLE: 1.0, 7 days: 0.5, 30 days: 0.1
        let abs_days = days_from_epoch.abs();
        if abs_days < 1.0 {
            1.0
        } else if abs_days < 7.0 {
            (1.0 - (abs_days - 1.0) / 12.0) as f32
        } else {
            (0.5 * (-0.1 * (abs_days - 7.0)).exp()) as f32
        }
    }
}

/// Batch propagator for multiple objects
pub struct BatchPropagator {
    propagators: Vec<(u32, Propagator)>,
}

impl BatchPropagator {
    pub fn new() -> Self {
        Self {
            propagators: Vec::new(),
        }
    }

    pub fn add(&mut self, tle: &TwoLineElement) -> Result<(), PropagationError> {
        let prop = Propagator::from_tle(tle)?;
        self.propagators.push((tle.norad_id, prop));
        Ok(())
    }

    /// Propagate all objects to a given time
    pub fn propagate_all(&self, time: DateTime<Utc>) -> Vec<(u32, Result<OrbitalState, PropagationError>)> {
        self.propagators
            .iter()
            .map(|(id, prop)| (*id, prop.propagate_to(time)))
            .collect()
    }

    /// Get states for all objects that successfully propagate
    pub fn get_states(&self, time: DateTime<Utc>) -> Vec<OrbitalState> {
        self.propagators
            .iter()
            .filter_map(|(_, prop)| prop.propagate_to(time).ok())
            .collect()
    }

    pub fn len(&self) -> usize {
        self.propagators.len()
    }

    pub fn is_empty(&self) -> bool {
        self.propagators.is_empty()
    }
}

impl Default for BatchPropagator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ISS_TLE: &str = "ISS (ZARYA)
1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9997
2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391424577";

    #[test]
    fn test_propagator_creation() {
        let tle = TwoLineElement::parse(ISS_TLE).unwrap();
        let prop = Propagator::from_tle(&tle);
        assert!(prop.is_ok());
    }

    #[test]
    fn test_propagation_at_epoch() {
        let tle = TwoLineElement::parse(ISS_TLE).unwrap();
        let prop = Propagator::from_tle(&tle).unwrap();

        let state = prop.propagate_to(tle.epoch).unwrap();

        // At epoch, should be at approximately the right altitude (LEO range)
        // Note: synthetic TLE may give slightly different values
        let alt = state.state.altitude_km();
        assert!(alt > 330.0 && alt < 460.0, "ISS altitude should be LEO range, got {}", alt);
    }

    #[test]
    fn test_propagation_period() {
        let tle = TwoLineElement::parse(ISS_TLE).unwrap();
        let prop = Propagator::from_tle(&tle).unwrap();

        // Propagate one orbital period (~92 minutes for ISS)
        let period_minutes = tle.period_minutes();
        let state_epoch = prop.propagate_minutes(0.0).unwrap();
        let state_period = prop.propagate_minutes(period_minutes).unwrap();

        // Position should be similar after one orbit
        let distance = state_epoch.state.distance_to(&state_period.state);
        assert!(distance < 100.0, "After one orbit, should be near start. Distance: {} km", distance);
    }

    #[test]
    fn test_ephemeris_generation() {
        let tle = TwoLineElement::parse(ISS_TLE).unwrap();
        let prop = Propagator::from_tle(&tle).unwrap();

        let start = tle.epoch;
        let end = tle.epoch + Duration::hours(1);
        let eph = prop.ephemeris(start, end, 600);  // 10-minute steps

        // Should have 7 points (0, 10, 20, 30, 40, 50, 60 minutes)
        assert_eq!(eph.len(), 7);
        assert!(eph.iter().all(|r| r.is_ok()));
    }
}
