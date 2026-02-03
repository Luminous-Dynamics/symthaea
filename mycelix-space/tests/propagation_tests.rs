//! Tests for orbital propagation
//!
//! Tests SGP4/SDP4 propagation accuracy and ephemeris generation.

use chrono::{Duration, Utc};
use orbital_mechanics::{
    tle::TwoLineElement,
    propagator::{Propagator, BatchPropagator},
};

/// Sample ISS TLE for testing
const ISS_TLE: &str = "ISS (ZARYA)
1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9997
2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391424577";

/// Test propagator creation from TLE
#[test]
fn test_propagator_creation() {
    let tle = TwoLineElement::parse(ISS_TLE).unwrap();
    let prop = Propagator::from_tle(&tle);

    assert!(prop.is_ok());
}

/// Test propagation at epoch (should match TLE)
#[test]
fn test_propagation_at_epoch() {
    let tle = TwoLineElement::parse(ISS_TLE).unwrap();
    let prop = Propagator::from_tle(&tle).unwrap();

    let state = prop.propagate_to(tle.epoch).unwrap();

    // At epoch, should be at approximately the right altitude
    let alt = state.state.altitude_km();
    assert!(alt > 330.0 && alt < 460.0, "ISS altitude at epoch: {} km", alt);

    // Speed should be approximately 7.6-7.8 km/s
    let speed = state.state.speed();
    assert!(speed > 7.4 && speed < 8.0, "ISS speed at epoch: {} km/s", speed);
}

/// Test propagation forward in time
#[test]
fn test_propagation_forward() {
    let tle = TwoLineElement::parse(ISS_TLE).unwrap();
    let prop = Propagator::from_tle(&tle).unwrap();

    // Propagate 1 hour forward
    let future = tle.epoch + Duration::hours(1);
    let state = prop.propagate_to(future);

    assert!(state.is_ok());
    let state = state.unwrap();

    // Should still be in valid orbit
    let alt = state.state.altitude_km();
    assert!(alt > 330.0 && alt < 460.0, "Altitude after 1 hour: {} km", alt);
}

/// Test propagation backward in time
#[test]
fn test_propagation_backward() {
    let tle = TwoLineElement::parse(ISS_TLE).unwrap();
    let prop = Propagator::from_tle(&tle).unwrap();

    // Propagate 1 hour backward
    let past = tle.epoch - Duration::hours(1);
    let state = prop.propagate_to(past);

    assert!(state.is_ok());
}

/// Test propagation period consistency
#[test]
fn test_propagation_period() {
    let tle = TwoLineElement::parse(ISS_TLE).unwrap();
    let prop = Propagator::from_tle(&tle).unwrap();

    // Propagate one orbital period (~92 minutes for ISS)
    let period_minutes = tle.period_minutes();
    let state_epoch = prop.propagate_minutes(0.0).unwrap();
    let state_period = prop.propagate_minutes(period_minutes).unwrap();

    // Position should be similar after one orbit (within ~100km due to perturbations)
    let distance = state_epoch.state.distance_to(&state_period.state);
    assert!(distance < 100.0, "After one orbit, distance from start: {} km", distance);
}

/// Test ephemeris generation
#[test]
fn test_ephemeris_generation() {
    let tle = TwoLineElement::parse(ISS_TLE).unwrap();
    let prop = Propagator::from_tle(&tle).unwrap();

    let start = tle.epoch;
    let end = tle.epoch + Duration::hours(2);

    // Generate ephemeris with 10-minute steps
    let eph = prop.ephemeris(start, end, 600); // 600 seconds = 10 minutes

    // Should have 13 points (0, 10, 20, ..., 120 minutes)
    assert_eq!(eph.len(), 13);

    // All should succeed
    assert!(eph.iter().all(|r| r.is_ok()));

    // Verify reasonable trajectory
    for result in &eph {
        let state = result.as_ref().unwrap();
        let alt = state.state.altitude_km();
        assert!(alt > 330.0 && alt < 460.0);
    }
}

/// Test propagation bounds
#[test]
fn test_propagation_bounds() {
    let tle = TwoLineElement::parse(ISS_TLE).unwrap();
    let prop = Propagator::from_tle(&tle).unwrap();

    // Default bounds are 30 days forward, 7 days backward
    // Propagating too far should fail
    let too_far = tle.epoch + Duration::days(60);
    let result = prop.propagate_to(too_far);

    assert!(result.is_err());
}

/// Test propagation with custom bounds
#[test]
fn test_custom_bounds() {
    let tle = TwoLineElement::parse(ISS_TLE).unwrap();
    let prop = Propagator::from_tle(&tle)
        .unwrap()
        .with_bounds(60.0, 14.0); // 60 days forward, 14 days backward

    // Should now succeed with extended bounds
    let future = tle.epoch + Duration::days(45);
    let result = prop.propagate_to(future);

    assert!(result.is_ok());
}

/// Test batch propagator
#[test]
fn test_batch_propagator() {
    let tle = TwoLineElement::parse(ISS_TLE).unwrap();

    let mut batch = BatchPropagator::new();
    batch.add(&tle).unwrap();

    assert_eq!(batch.len(), 1);
    assert!(!batch.is_empty());

    // Propagate all to current time relative to epoch
    let states = batch.get_states(tle.epoch);
    assert_eq!(states.len(), 1);
    assert_eq!(states[0].norad_id, 25544);
}

/// Test state vector components
#[test]
fn test_state_vector_components() {
    let tle = TwoLineElement::parse(ISS_TLE).unwrap();
    let prop = Propagator::from_tle(&tle).unwrap();
    let state = prop.propagate_to(tle.epoch).unwrap();

    // Position components
    let pos = state.state.position();
    assert!(pos.norm() > 6000.0); // Should be at least Earth radius

    // Velocity components
    let vel = state.state.velocity();
    assert!(vel.norm() > 7.0 && vel.norm() < 8.5); // LEO orbital velocity

    // As array
    let arr = state.state.to_array();
    assert_eq!(arr.len(), 6);
}

/// Test distance calculation
#[test]
fn test_state_distance() {
    let tle = TwoLineElement::parse(ISS_TLE).unwrap();
    let prop = Propagator::from_tle(&tle).unwrap();

    let state1 = prop.propagate_minutes(0.0).unwrap();
    let state2 = prop.propagate_minutes(1.0).unwrap(); // 1 minute later

    let distance = state1.state.distance_to(&state2.state);

    // ISS travels at ~7.6 km/s, so 1 minute = ~456 km
    assert!(distance > 400.0 && distance < 500.0, "Distance in 1 minute: {} km", distance);
}

/// Test relative velocity calculation
#[test]
fn test_relative_velocity() {
    let tle = TwoLineElement::parse(ISS_TLE).unwrap();
    let prop = Propagator::from_tle(&tle).unwrap();

    let state1 = prop.propagate_minutes(0.0).unwrap();
    let state2 = prop.propagate_minutes(0.0).unwrap(); // Same time = same velocity

    let rel_vel = state1.state.relative_velocity(&state2.state);

    // Same object at same time should have zero relative velocity
    assert!(rel_vel < 0.001, "Relative velocity should be ~0: {} km/s", rel_vel);
}

/// Test propagate_minutes convenience method
#[test]
fn test_propagate_minutes() {
    let tle = TwoLineElement::parse(ISS_TLE).unwrap();
    let prop = Propagator::from_tle(&tle).unwrap();

    // Propagate 30 minutes
    let state = prop.propagate_minutes(30.0).unwrap();

    assert!(state.state.altitude_km() > 350.0);
    assert!(state.state.altitude_km() < 450.0);
}

/// Test covariance estimation from TLE age
#[test]
fn test_covariance_estimation() {
    let tle = TwoLineElement::parse(ISS_TLE).unwrap();
    let prop = Propagator::from_tle(&tle).unwrap();

    let state = prop.propagate_minutes(0.0).unwrap();

    // Propagated state should have covariance
    assert!(state.covariance.is_some());

    let cov = state.covariance.unwrap();
    let pos_sigma = cov.position_sigma();
    let vel_sigma = cov.velocity_sigma();

    // Should have meaningful uncertainty values
    assert!(pos_sigma > 0.0);
    assert!(vel_sigma > 0.0);
}

/// Test quality estimation
#[test]
fn test_quality_estimation() {
    let tle = TwoLineElement::parse(ISS_TLE).unwrap();
    let prop = Propagator::from_tle(&tle).unwrap();

    // At epoch, quality should be high
    let state_epoch = prop.propagate_minutes(0.0).unwrap();
    assert!(state_epoch.quality > 0.9, "Quality at epoch: {}", state_epoch.quality);

    // Quality should degrade with time from epoch
    let state_7d = prop.propagate_minutes(7.0 * 24.0 * 60.0).unwrap(); // 7 days
    assert!(state_7d.quality < state_epoch.quality, "Quality at 7 days: {}", state_7d.quality);
}

/// Test NORAD ID preservation
#[test]
fn test_norad_id_preserved() {
    let tle = TwoLineElement::parse(ISS_TLE).unwrap();
    let prop = Propagator::from_tle(&tle).unwrap();

    assert_eq!(prop.norad_id(), 25544);

    let state = prop.propagate_to(tle.epoch).unwrap();
    assert_eq!(state.norad_id, 25544);
}

/// Test epoch accessor
#[test]
fn test_epoch_accessor() {
    let tle = TwoLineElement::parse(ISS_TLE).unwrap();
    let prop = Propagator::from_tle(&tle).unwrap();

    assert_eq!(prop.epoch(), tle.epoch);
}
