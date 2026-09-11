// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! LL-003B spherical two-body forward ballistic propagation.
//!
//! This module is a research dynamics model for uncrewed cargo studies. It has
//! no launch-release authority, hardware command path, terrain model, or safe-
//! corridor semantics. All physical constants and frame provenance are supplied
//! by the caller.

use std::f64::consts::PI;

/// Provenance copied into every ballistic result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BallisticModelMetadata {
    pub model_id: String,
    pub frame: String,
    pub constants_ref: String,
}

impl BallisticModelMetadata {
    fn is_well_formed(&self) -> bool {
        !self.model_id.trim().is_empty()
            && !self.frame.trim().is_empty()
            && !self.constants_ref.trim().is_empty()
    }
}

/// Initial state and numerical contract for one spherical two-body surface arc.
#[derive(Debug, Clone, PartialEq)]
pub struct SphericalBallisticInput {
    /// Central-body gravitational parameter [m^3/s^2].
    pub mu_m3_s2: f64,
    /// Declared spherical reference radius [m].
    pub radius_m: f64,
    /// Launch speed relative to the non-rotating body-fixed surface proxy [m/s].
    pub speed_m_s: f64,
    /// Elevation above the local tangent plane [deg], strictly between 0 and 90.
    pub elevation_deg: f64,
    /// Azimuth clockwise from local north toward east [deg].
    pub azimuth_deg: f64,
    pub latitude_deg: f64,
    pub longitude_deg: f64,
    /// Fixed RK4 integration step [s].
    pub dt_s: f64,
    /// Maximum propagation time before returning a bounded no-return result [s].
    pub max_time_s: f64,
    pub metadata: BallisticModelMetadata,
}

impl SphericalBallisticInput {
    pub fn validate(&self) -> Result<(), BallisticError> {
        let scalars = [
            self.mu_m3_s2,
            self.radius_m,
            self.speed_m_s,
            self.elevation_deg,
            self.azimuth_deg,
            self.latitude_deg,
            self.longitude_deg,
            self.dt_s,
            self.max_time_s,
        ];
        if scalars.iter().any(|value| !value.is_finite()) {
            return Err(BallisticError::NonFiniteInput);
        }
        if self.mu_m3_s2 <= 0.0
            || self.radius_m <= 0.0
            || self.speed_m_s <= 0.0
            || self.dt_s <= 0.0
            || self.max_time_s <= 0.0
        {
            return Err(BallisticError::InvalidInput);
        }
        if !(0.0 < self.elevation_deg && self.elevation_deg < 90.0) {
            return Err(BallisticError::InvalidElevation);
        }
        if !(-90.0..=90.0).contains(&self.latitude_deg) {
            return Err(BallisticError::InvalidLatitude);
        }
        if !self.metadata.is_well_formed() {
            return Err(BallisticError::InvalidMetadata);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BallisticOutcome {
    Reimpact,
    Escape,
    NoReturnWithinWindow,
}

/// Deterministic forward-propagation result.
#[derive(Debug, Clone, PartialEq)]
pub struct SphericalBallisticResult {
    pub outcome: BallisticOutcome,
    pub flight_time_s: Option<f64>,
    pub ground_range_m: Option<f64>,
    pub max_altitude_m: f64,
    pub arrival_speed_m_s: Option<f64>,
    /// Signed angle relative to the local tangent plane; descending arrival is negative.
    pub arrival_flight_path_angle_deg: Option<f64>,
    pub arrival_latitude_deg: Option<f64>,
    pub arrival_longitude_deg: Option<f64>,
    pub max_relative_energy_error: f64,
    pub max_relative_angular_momentum_error: f64,
    pub metadata: BallisticModelMetadata,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BallisticError {
    NonFiniteInput,
    InvalidInput,
    InvalidElevation,
    InvalidLatitude,
    InvalidMetadata,
    NumericalFailure,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3 {
    const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    fn scale(self, scalar: f64) -> Self {
        Self::new(self.x * scalar, self.y * scalar, self.z * scalar)
    }

    fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }

    fn cross(self, other: Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

fn acceleration(position: Vec3, mu: f64) -> Result<Vec3, BallisticError> {
    let radius = position.norm();
    if !radius.is_finite() || radius <= 0.0 {
        return Err(BallisticError::NumericalFailure);
    }
    let factor = -mu / radius.powi(3);
    let result = position.scale(factor);
    result
        .is_finite()
        .then_some(result)
        .ok_or(BallisticError::NumericalFailure)
}

fn rk4(
    position: Vec3,
    velocity: Vec3,
    dt_s: f64,
    mu: f64,
) -> Result<(Vec3, Vec3), BallisticError> {
    let k1r = velocity;
    let k1v = acceleration(position, mu)?;

    let r2 = position.add(k1r.scale(0.5 * dt_s));
    let v2 = velocity.add(k1v.scale(0.5 * dt_s));
    let k2r = v2;
    let k2v = acceleration(r2, mu)?;

    let r3 = position.add(k2r.scale(0.5 * dt_s));
    let v3 = velocity.add(k2v.scale(0.5 * dt_s));
    let k3r = v3;
    let k3v = acceleration(r3, mu)?;

    let r4 = position.add(k3r.scale(dt_s));
    let v4 = velocity.add(k3v.scale(dt_s));
    let k4r = v4;
    let k4v = acceleration(r4, mu)?;

    let position_next = position.add(
        k1r.add(k2r.scale(2.0))
            .add(k3r.scale(2.0))
            .add(k4r)
            .scale(dt_s / 6.0),
    );
    let velocity_next = velocity.add(
        k1v.add(k2v.scale(2.0))
            .add(k3v.scale(2.0))
            .add(k4v)
            .scale(dt_s / 6.0),
    );

    if !position_next.is_finite() || !velocity_next.is_finite() {
        return Err(BallisticError::NumericalFailure);
    }
    Ok((position_next, velocity_next))
}

fn specific_energy(position: Vec3, velocity: Vec3, mu: f64) -> f64 {
    0.5 * velocity.dot(velocity) - mu / position.norm()
}

fn specific_angular_momentum(position: Vec3, velocity: Vec3) -> f64 {
    position.cross(velocity).norm()
}

fn degrees(value_rad: f64) -> f64 {
    value_rad * 180.0 / PI
}

fn radians(value_deg: f64) -> f64 {
    value_deg * PI / 180.0
}

fn surface_frame(latitude_deg: f64, longitude_deg: f64) -> (Vec3, Vec3, Vec3) {
    let latitude = radians(latitude_deg);
    let longitude = radians(longitude_deg);
    let up = Vec3::new(
        latitude.cos() * longitude.cos(),
        latitude.cos() * longitude.sin(),
        latitude.sin(),
    );
    let east = Vec3::new(-longitude.sin(), longitude.cos(), 0.0);
    let north = Vec3::new(
        -latitude.sin() * longitude.cos(),
        -latitude.sin() * longitude.sin(),
        latitude.cos(),
    );
    (north, east, up)
}

fn initial_state(input: &SphericalBallisticInput) -> (Vec3, Vec3) {
    let (north, east, up) = surface_frame(input.latitude_deg, input.longitude_deg);
    let azimuth = radians(input.azimuth_deg);
    let elevation = radians(input.elevation_deg);
    let horizontal = north.scale(azimuth.cos()).add(east.scale(azimuth.sin()));
    let direction = horizontal
        .scale(elevation.cos())
        .add(up.scale(elevation.sin()));
    (up.scale(input.radius_m), direction.scale(input.speed_m_s))
}

fn latitude_longitude(position: Vec3) -> Result<(f64, f64), BallisticError> {
    let radius = position.norm();
    if !radius.is_finite() || radius <= 0.0 {
        return Err(BallisticError::NumericalFailure);
    }
    let latitude = degrees((position.z / radius).clamp(-1.0, 1.0).asin());
    let longitude = degrees(position.y.atan2(position.x));
    Ok((latitude, longitude))
}

fn relative_error(value: f64, reference: f64) -> f64 {
    (value - reference).abs() / reference.abs().max(1.0e-30)
}

/// Propagate one ideal spherical two-body ballistic arc.
///
/// This function is a dynamics study primitive only. A `Reimpact` result does
/// not imply terrain clearance, acceptable miss consequences, receiver capture,
/// or permission to release a payload.
pub fn propagate_spherical_two_body(
    input: &SphericalBallisticInput,
) -> Result<SphericalBallisticResult, BallisticError> {
    input.validate()?;

    let (initial_position, initial_velocity) = initial_state(input);
    let mut position = initial_position;
    let mut velocity = initial_velocity;
    let initial_energy = specific_energy(position, velocity, input.mu_m3_s2);
    let initial_h = specific_angular_momentum(position, velocity);

    if !initial_energy.is_finite() || !initial_h.is_finite() {
        return Err(BallisticError::NumericalFailure);
    }

    if initial_energy >= 0.0 && position.dot(velocity) > 0.0 {
        return Ok(SphericalBallisticResult {
            outcome: BallisticOutcome::Escape,
            flight_time_s: None,
            ground_range_m: None,
            max_altitude_m: 0.0,
            arrival_speed_m_s: None,
            arrival_flight_path_angle_deg: None,
            arrival_latitude_deg: None,
            arrival_longitude_deg: None,
            max_relative_energy_error: 0.0,
            max_relative_angular_momentum_error: 0.0,
            metadata: input.metadata.clone(),
        });
    }

    let mut time_s = 0.0;
    let mut max_altitude_m: f64 = 0.0;
    let mut max_energy_error: f64 = 0.0;
    let mut max_h_error: f64 = 0.0;
    let mut lifted = false;
    let lift_threshold_m = 1.0e-10 * input.radius_m;

    while time_s < input.max_time_s {
        let remaining = input.max_time_s - time_s;
        let step_s = input.dt_s.min(remaining);
        if step_s <= 0.0 {
            break;
        }

        let current_altitude_m = position.norm() - input.radius_m;
        let (next_position, next_velocity) = rk4(position, velocity, step_s, input.mu_m3_s2)?;
        let next_altitude_m = next_position.norm() - input.radius_m;
        max_altitude_m = max_altitude_m.max(next_altitude_m);
        if next_altitude_m > lift_threshold_m {
            lifted = true;
        }

        let energy = specific_energy(next_position, next_velocity, input.mu_m3_s2);
        let h = specific_angular_momentum(next_position, next_velocity);
        if !energy.is_finite() || !h.is_finite() {
            return Err(BallisticError::NumericalFailure);
        }
        max_energy_error = max_energy_error.max(relative_error(energy, initial_energy));
        max_h_error = max_h_error.max(relative_error(h, initial_h));

        if lifted && current_altitude_m > 0.0 && next_altitude_m <= 0.0 {
            let mut low_s = 0.0;
            let mut high_s = step_s;
            for _ in 0..60 {
                let mid_s = 0.5 * (low_s + high_s);
                let (mid_position, _) = rk4(position, velocity, mid_s, input.mu_m3_s2)?;
                if mid_position.norm() - input.radius_m > 0.0 {
                    low_s = mid_s;
                } else {
                    high_s = mid_s;
                }
            }
            let impact_offset_s = 0.5 * (low_s + high_s);
            let (impact_position, impact_velocity) =
                rk4(position, velocity, impact_offset_s, input.mu_m3_s2)?;
            let impact_time_s = time_s + impact_offset_s;

            let impact_radius = impact_position.norm();
            if !impact_radius.is_finite() || impact_radius <= 0.0 {
                return Err(BallisticError::NumericalFailure);
            }
            let initial_unit = initial_position.scale(1.0 / input.radius_m);
            let impact_unit = impact_position.scale(1.0 / impact_radius);
            let central_angle = initial_unit.dot(impact_unit).clamp(-1.0, 1.0).acos();
            let ground_range_m = input.radius_m * central_angle;
            let arrival_speed_m_s = impact_velocity.norm();
            let radial_speed_m_s = impact_velocity.dot(impact_unit);
            let tangential_speed_m_s =
                (arrival_speed_m_s.powi(2) - radial_speed_m_s.powi(2))
                    .max(0.0)
                    .sqrt();
            let flight_path_angle_deg = degrees(radial_speed_m_s.atan2(tangential_speed_m_s));
            let (arrival_latitude_deg, arrival_longitude_deg) = latitude_longitude(impact_position)?;

            let terminal_energy = specific_energy(impact_position, impact_velocity, input.mu_m3_s2);
            let terminal_h = specific_angular_momentum(impact_position, impact_velocity);
            max_energy_error = max_energy_error.max(relative_error(terminal_energy, initial_energy));
            max_h_error = max_h_error.max(relative_error(terminal_h, initial_h));

            return Ok(SphericalBallisticResult {
                outcome: BallisticOutcome::Reimpact,
                flight_time_s: Some(impact_time_s),
                ground_range_m: Some(ground_range_m),
                max_altitude_m,
                arrival_speed_m_s: Some(arrival_speed_m_s),
                arrival_flight_path_angle_deg: Some(flight_path_angle_deg),
                arrival_latitude_deg: Some(arrival_latitude_deg),
                arrival_longitude_deg: Some(arrival_longitude_deg),
                max_relative_energy_error: max_energy_error,
                max_relative_angular_momentum_error: max_h_error,
                metadata: input.metadata.clone(),
            });
        }

        position = next_position;
        velocity = next_velocity;
        time_s += step_s;
    }

    Ok(SphericalBallisticResult {
        outcome: BallisticOutcome::NoReturnWithinWindow,
        flight_time_s: None,
        ground_range_m: None,
        max_altitude_m,
        arrival_speed_m_s: None,
        arrival_flight_path_angle_deg: None,
        arrival_latitude_deg: None,
        arrival_longitude_deg: None,
        max_relative_energy_error: max_energy_error,
        max_relative_angular_momentum_error: max_h_error,
        metadata: input.metadata.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata() -> BallisticModelMetadata {
        BallisticModelMetadata {
            model_id: "ll003b-test".into(),
            frame: "synthetic-body-fixed/non-rotating-proxy".into(),
            constants_ref: "synthetic-mu-radius-fixture-v1".into(),
        }
    }

    fn fixture(dt_s: f64) -> SphericalBallisticInput {
        SphericalBallisticInput {
            mu_m3_s2: 2.0e12,
            radius_m: 1.0e6,
            speed_m_s: 100.0,
            elevation_deg: 45.0,
            azimuth_deg: 90.0,
            latitude_deg: 0.0,
            longitude_deg: 0.0,
            dt_s,
            max_time_s: 1_000.0,
            metadata: metadata(),
        }
    }

    fn relative(actual: f64, expected: f64) -> f64 {
        (actual - expected).abs() / expected.abs().max(1.0e-30)
    }

    #[test]
    fn matches_independent_python_oracle_fixture() {
        let result = propagate_spherical_two_body(&fixture(0.05)).unwrap();
        assert_eq!(result.outcome, BallisticOutcome::Reimpact);
        assert!(relative(result.flight_time_s.unwrap(), 71.00636995976859) < 2.0e-9);
        assert!(relative(result.ground_range_m.unwrap(), 5012.520833156787) < 2.0e-9);
        assert!(relative(result.max_altitude_m, 1254.7031614161097) < 2.0e-6);
        assert!(relative(result.arrival_speed_m_s.unwrap(), 100.0) < 1.0e-10);
        assert!(
            (result.arrival_flight_path_angle_deg.unwrap() + 45.0).abs() < 1.0e-8
        );
        assert!(result.max_relative_energy_error < 1.0e-10);
        assert!(result.max_relative_angular_momentum_error < 1.0e-10);
    }

    #[test]
    fn short_arc_agrees_with_flat_limit() {
        let input = fixture(0.05);
        let result = propagate_spherical_two_body(&input).unwrap();
        let surface_g = input.mu_m3_s2 / input.radius_m.powi(2);
        let flat_range = input.speed_m_s.powi(2)
            * (2.0 * radians(input.elevation_deg)).sin()
            / surface_g;
        assert!(relative(result.ground_range_m.unwrap(), flat_range) < 2.0e-3);
    }

    #[test]
    fn east_west_equatorial_symmetry_holds() {
        let east = propagate_spherical_two_body(&fixture(0.05)).unwrap();
        let mut west_input = fixture(0.05);
        west_input.azimuth_deg = 270.0;
        let west = propagate_spherical_two_body(&west_input).unwrap();

        assert!(relative(east.flight_time_s.unwrap(), west.flight_time_s.unwrap()) < 1.0e-12);
        assert!(relative(east.ground_range_m.unwrap(), west.ground_range_m.unwrap()) < 1.0e-12);
        assert!(
            (east.arrival_longitude_deg.unwrap() + west.arrival_longitude_deg.unwrap()).abs()
                < 1.0e-10
        );
    }

    #[test]
    fn outward_nonnegative_energy_is_escape() {
        let mut input = fixture(0.1);
        input.speed_m_s = 2_100.0;
        let result = propagate_spherical_two_body(&input).unwrap();
        assert_eq!(result.outcome, BallisticOutcome::Escape);
        assert!(result.flight_time_s.is_none());
        assert!(result.ground_range_m.is_none());
    }

    #[test]
    fn ordinary_suborbital_solution_refines_with_step_size() {
        let coarse = propagate_spherical_two_body(&fixture(0.1)).unwrap();
        let fine = propagate_spherical_two_body(&fixture(0.05)).unwrap();
        assert!(relative(coarse.flight_time_s.unwrap(), fine.flight_time_s.unwrap()) < 1.0e-5);
        assert!(relative(coarse.ground_range_m.unwrap(), fine.ground_range_m.unwrap()) < 1.0e-5);
    }

    #[test]
    fn malformed_inputs_fail_closed() {
        let mut bad = fixture(0.05);
        bad.mu_m3_s2 = f64::NAN;
        assert_eq!(bad.validate(), Err(BallisticError::NonFiniteInput));

        let mut bad_elevation = fixture(0.05);
        bad_elevation.elevation_deg = 90.0;
        assert_eq!(bad_elevation.validate(), Err(BallisticError::InvalidElevation));

        let mut bad_metadata = fixture(0.05);
        bad_metadata.metadata.constants_ref.clear();
        assert_eq!(bad_metadata.validate(), Err(BallisticError::InvalidMetadata));
    }
}
