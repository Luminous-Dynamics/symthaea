// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Calibrated multi-sensor bearing-only triangulation.
//!
//! Inputs are sensor origins in ECEF metres, active sensor-body-to-ECEF
//! quaternions, azimuth/elevation bearings, and one-sigma angular errors.
//! Sensor-body convention: +X forward, +Y to the right, +Z upward; azimuth
//! increases from +X toward +Y and elevation increases toward +Z.
//!
//! The estimator first finds the least-squares intersection of infinite lines,
//! then performs uncertainty-weighted Gauss-Newton refinement in angular
//! residual space. It rejects poorly conditioned geometry and rays whose
//! solution lies behind a sensor. The returned covariance is the local
//! linearized inverse information matrix, not a complete instrument model.
//!
//! This is an optical/radar bearing geometry primitive. It does not perform
//! time synchronization, platform trajectory smoothing, track association,
//! atmospheric refraction correction, radar range fusion or object identity
//! inference. Those must be supplied and validated by upstream systems.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

const VECTOR_EPSILON: f64 = 1.0e-12;
const PIVOT_RELATIVE_TOLERANCE: f64 = 1.0e-13;
const MIN_AZIMUTH_COSINE: f64 = 1.0e-6;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vector3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vector3 {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn norm(self) -> f64 {
        self.x.hypot(self.y).hypot(self.z)
    }

    pub fn scale(self, factor: f64) -> Self {
        Self::new(self.x * factor, self.y * factor, self.z * factor)
    }

    pub fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    pub fn subtract(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }

    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

/// Active rotation from sensor-body coordinates into ECEF coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quaternion {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quaternion {
    pub const IDENTITY: Self = Self {
        w: 1.0,
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    pub fn normalized(self) -> Result<Self, TriangulationError> {
        if !self.w.is_finite() || !self.x.is_finite() || !self.y.is_finite() || !self.z.is_finite()
        {
            return Err(TriangulationError::InvalidQuaternion);
        }
        let norm = self.w.hypot(self.x).hypot(self.y).hypot(self.z);
        if !norm.is_finite() || norm <= VECTOR_EPSILON {
            return Err(TriangulationError::InvalidQuaternion);
        }
        Ok(Self {
            w: self.w / norm,
            x: self.x / norm,
            y: self.y / norm,
            z: self.z / norm,
        })
    }

    /// Transform an ECEF vector back into sensor-body coordinates.
    pub fn inverse_rotate_vector(self, vector: Vector3) -> Result<Vector3, TriangulationError> {
        let q = self.normalized()?;
        Self {
            w: q.w,
            x: -q.x,
            y: -q.y,
            z: -q.z,
        }
        .rotate_vector(vector)
    }

    pub fn rotate_vector(self, vector: Vector3) -> Result<Vector3, TriangulationError> {
        let q = self.normalized()?;
        let imaginary = Vector3::new(q.x, q.y, q.z);
        let rotated = imaginary
            .scale(2.0 * imaginary.dot(vector))
            .add(vector.scale(q.w * q.w - imaginary.dot(imaginary)))
            .add(imaginary.cross(vector).scale(2.0 * q.w));
        if rotated.is_finite() {
            Ok(rotated)
        } else {
            Err(TriangulationError::NonFiniteIntermediate)
        }
    }
}

/// One calibrated bearing observation at a particular synchronized epoch.
#[derive(Clone, Debug, PartialEq)]
pub struct RayObservation {
    pub sensor_id: String,
    pub origin_ecef_m: Vector3,
    pub sensor_to_ecef: Quaternion,
    pub azimuth_rad: f64,
    pub elevation_rad: f64,
    pub sigma_azimuth_rad: f64,
    pub sigma_elevation_rad: f64,
}

impl RayObservation {
    fn prepare(&self) -> Result<PreparedRay, TriangulationError> {
        if self.sensor_id.trim().is_empty() {
            return Err(TriangulationError::EmptySensorId);
        }
        if !self.origin_ecef_m.is_finite() {
            return Err(TriangulationError::NonFiniteVector);
        }
        if !self.azimuth_rad.is_finite() || !self.elevation_rad.is_finite() {
            return Err(TriangulationError::NonFiniteAngle);
        }
        if !(-std::f64::consts::FRAC_PI_2..=std::f64::consts::FRAC_PI_2)
            .contains(&self.elevation_rad)
        {
            return Err(TriangulationError::ElevationOutOfBounds);
        }
        if !self.sigma_azimuth_rad.is_finite()
            || self.sigma_azimuth_rad <= 0.0
            || !self.sigma_elevation_rad.is_finite()
            || self.sigma_elevation_rad <= 0.0
        {
            return Err(TriangulationError::InvalidAngularUncertainty);
        }

        let cos_elevation = self.elevation_rad.cos();
        if cos_elevation.abs() < MIN_AZIMUTH_COSINE {
            return Err(TriangulationError::AzimuthSingularity);
        }

        let azimuth = self.azimuth_rad;
        let elevation = self.elevation_rad;
        let direction_body = Vector3::new(
            cos_elevation * azimuth.cos(),
            cos_elevation * azimuth.sin(),
            elevation.sin(),
        );
        let azimuth_tangent_body = Vector3::new(-azimuth.sin(), azimuth.cos(), 0.0);
        let elevation_tangent_body = Vector3::new(
            -elevation.sin() * azimuth.cos(),
            -elevation.sin() * azimuth.sin(),
            cos_elevation,
        );

        let direction = self.sensor_to_ecef.rotate_vector(direction_body)?;
        let tangent_azimuth = self.sensor_to_ecef.rotate_vector(azimuth_tangent_body)?;
        let tangent_elevation = self.sensor_to_ecef.rotate_vector(elevation_tangent_body)?;

        Ok(PreparedRay {
            sensor_id: self.sensor_id.clone(),
            origin: self.origin_ecef_m,
            direction,
            tangent_azimuth,
            tangent_elevation,
            cos_elevation,
            sigma_azimuth: self.sigma_azimuth_rad,
            sigma_elevation: self.sigma_elevation_rad,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TriangulationConfig {
    pub max_iterations: usize,
    pub position_tolerance_m: f64,
    pub max_condition_number: f64,
}

impl Default for TriangulationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 32,
            position_tolerance_m: 1.0e-6,
            max_condition_number: 1.0e10,
        }
    }
}

impl TriangulationConfig {
    fn validate(self) -> Result<(), TriangulationError> {
        if self.max_iterations == 0 || self.max_iterations > 256 {
            return Err(TriangulationError::InvalidConfiguration);
        }
        if !self.position_tolerance_m.is_finite() || self.position_tolerance_m <= 0.0 {
            return Err(TriangulationError::InvalidConfiguration);
        }
        if !self.max_condition_number.is_finite() || self.max_condition_number <= 1.0 {
            return Err(TriangulationError::InvalidConfiguration);
        }
        Ok(())
    }
}

/// Bearing-only triangulation with a local linearized covariance estimate.
#[derive(Clone, Debug, PartialEq)]
pub struct TriangulatedPosition {
    pub position_ecef_m: Vector3,
    pub covariance_ecef_m2: [[f64; 3]; 3],
    pub residual_chi_square: f64,
    pub degrees_of_freedom: usize,
    pub rms_angular_residual_rad: f64,
    pub condition_number: f64,
    pub observations: usize,
    pub iterations: usize,
}

/// Candidate sensor placement and orientation for a planned bearing observation.
#[derive(Clone, Debug, PartialEq)]
pub struct SensorPlacementCandidate {
    pub sensor_id: String,
    pub origin_ecef_m: Vector3,
    pub sensor_to_ecef: Quaternion,
    pub sigma_azimuth_rad: f64,
    pub sigma_elevation_rad: f64,
}

/// Local Fisher-information / Cramer-Rao design scores for adding one sensor.
#[derive(Clone, Debug, PartialEq)]
pub struct SensorPlacementScore {
    pub sensor_id: String,
    /// D-optimal score: log(det(Fisher information)).
    pub log_determinant_information: f64,
    /// A-optimal score: trace of the local covariance lower bound.
    pub a_optimal_variance_sum_m2: f64,
    /// Ratio of largest to smallest Fisher-information eigenvalue.
    pub condition_number: f64,
    pub covariance_lower_bound_m2: [[f64; 3]; 3],
    pub observations_after_placement: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TriangulationError {
    InsufficientObservations,
    EmptySensorId,
    DuplicateSensorId(String),
    DuplicateCandidateId(String),
    NonFiniteVector,
    NonFiniteAngle,
    InvalidAngularUncertainty,
    ElevationOutOfBounds,
    AzimuthSingularity,
    InvalidQuaternion,
    InvalidConfiguration,
    ParallelOrIllConditioned { condition_number: f64 },
    BehindSensor(String),
    ZeroRange(String),
    NonConvergence,
    NonFiniteIntermediate,
}

impl fmt::Display for TriangulationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientObservations => {
                write!(f, "at least two independent sensor bearings are required")
            }
            Self::EmptySensorId => write!(f, "sensor id must not be empty"),
            Self::DuplicateSensorId(id) => write!(f, "sensor id {id} appears more than once"),
            Self::DuplicateCandidateId(id) => write!(
                f,
                "candidate sensor id {id} is duplicated or already observed"
            ),
            Self::NonFiniteVector => write!(f, "sensor ECEF origin is not finite"),
            Self::NonFiniteAngle => write!(f, "bearing angle is not finite"),
            Self::InvalidAngularUncertainty => {
                write!(
                    f,
                    "one-sigma angular uncertainty must be finite and positive"
                )
            }
            Self::ElevationOutOfBounds => {
                write!(f, "elevation must lie between -pi/2 and +pi/2")
            }
            Self::AzimuthSingularity => {
                write!(f, "azimuth is ill-defined at this near-vertical elevation")
            }
            Self::InvalidQuaternion => {
                write!(f, "sensor-to-ECEF quaternion must be finite and nonzero")
            }
            Self::InvalidConfiguration => write!(f, "triangulation configuration is invalid"),
            Self::ParallelOrIllConditioned { condition_number } => write!(
                f,
                "bearing geometry is parallel or ill-conditioned (condition={condition_number:.3e})"
            ),
            Self::BehindSensor(id) => {
                write!(f, "estimated target lies behind sensor {id}")
            }
            Self::ZeroRange(id) => write!(f, "estimated target coincides with sensor {id}"),
            Self::NonConvergence => write!(f, "bearing triangulation did not converge"),
            Self::NonFiniteIntermediate => write!(f, "triangulation produced a non-finite value"),
        }
    }
}

impl Error for TriangulationError {}

#[derive(Clone, Debug)]
struct PreparedRay {
    sensor_id: String,
    origin: Vector3,
    direction: Vector3,
    tangent_azimuth: Vector3,
    tangent_elevation: Vector3,
    cos_elevation: f64,
    sigma_azimuth: f64,
    sigma_elevation: f64,
}

/// Triangulate a synchronized set of calibrated bearings using default gates.
pub fn triangulate_bearings(
    observations: &[RayObservation],
) -> Result<TriangulatedPosition, TriangulationError> {
    triangulate_bearings_with_config(observations, TriangulationConfig::default())
}

/// Triangulate a synchronized set of calibrated bearings.
///
/// The linear line-intersection estimate initializes a weighted Gauss-Newton
/// solve in angular residual space. Covariance is approximated by the inverse
/// final normal matrix. The caller remains responsible for establishing that
/// observations are time-aligned and that sensor poses/covariances are valid.
pub fn triangulate_bearings_with_config(
    observations: &[RayObservation],
    config: TriangulationConfig,
) -> Result<TriangulatedPosition, TriangulationError> {
    config.validate()?;
    if observations.len() < 2 {
        return Err(TriangulationError::InsufficientObservations);
    }

    let mut seen_sensor_ids = BTreeSet::new();
    let mut rays = Vec::with_capacity(observations.len());
    for observation in observations {
        if !seen_sensor_ids.insert(observation.sensor_id.as_str()) {
            return Err(TriangulationError::DuplicateSensorId(
                observation.sensor_id.clone(),
            ));
        }
        rays.push(observation.prepare()?);
    }

    let (initial_normal, initial_rhs) = line_intersection_normal_equations(&rays);
    let initial_inverse = invert_well_conditioned(initial_normal, config.max_condition_number)?;
    let mut position = matrix_vector(initial_inverse, initial_rhs);

    let mut converged = false;
    let mut iterations = 0;
    for iteration in 0..config.max_iterations {
        iterations = iteration + 1;
        let (normal, gradient, _, _) = angular_normal_equations(&rays, position)?;
        let inverse = invert_well_conditioned(normal, config.max_condition_number)?;
        let delta = matrix_vector(inverse, gradient.scale(-1.0));
        position = position.add(delta);
        if !position.is_finite() {
            return Err(TriangulationError::NonFiniteIntermediate);
        }
        if delta.norm() <= config.position_tolerance_m {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(TriangulationError::NonConvergence);
    }

    let (normal, _, chi_square, sum_residual_squared) = angular_normal_equations(&rays, position)?;
    let covariance = invert_well_conditioned(normal, config.max_condition_number)?;
    let condition_number = symmetric_condition_number(normal)?;
    let degrees_of_freedom = 2 * rays.len() - 3;

    Ok(TriangulatedPosition {
        position_ecef_m: position,
        covariance_ecef_m2: covariance,
        residual_chi_square: chi_square,
        degrees_of_freedom,
        rms_angular_residual_rad: (sum_residual_squared / (2 * rays.len()) as f64).sqrt(),
        condition_number,
        observations: rays.len(),
        iterations,
    })
}

type Matrix3 = [[f64; 3]; 3];

/// Compute the local bearing Fisher information matrix at an assumed ECEF target.
pub fn bearing_fisher_information(
    observations: &[RayObservation],
    assumed_target_ecef_m: Vector3,
) -> Result<[[f64; 3]; 3], TriangulationError> {
    if observations.is_empty() || !assumed_target_ecef_m.is_finite() {
        return Err(TriangulationError::InsufficientObservations);
    }
    let mut seen = BTreeSet::new();
    let mut rays = Vec::with_capacity(observations.len());
    for observation in observations {
        if !seen.insert(observation.sensor_id.as_str()) {
            return Err(TriangulationError::DuplicateSensorId(
                observation.sensor_id.clone(),
            ));
        }
        rays.push(observation.prepare()?);
    }
    angular_information_matrix(&rays, assumed_target_ecef_m)
}

/// Rank candidate placements by D-optimal information at an assumed target.
///
/// Each candidate's bearing is predicted from the supplied target and pose.
/// Scores are local design calculations: target point and angular standard
/// deviations are treated as known. Pose uncertainty, target-prior spread,
/// visibility, occlusion, weather, cost and platform constraints are omitted.
pub fn rank_sensor_placements(
    existing_observations: &[RayObservation],
    assumed_target_ecef_m: Vector3,
    candidates: &[SensorPlacementCandidate],
) -> Result<Vec<SensorPlacementScore>, TriangulationError> {
    if !assumed_target_ecef_m.is_finite() {
        return Err(TriangulationError::NonFiniteVector);
    }

    let mut seen_ids = BTreeSet::new();
    let mut rays = Vec::with_capacity(existing_observations.len() + 1);
    for observation in existing_observations {
        if !seen_ids.insert(observation.sensor_id.as_str()) {
            return Err(TriangulationError::DuplicateSensorId(
                observation.sensor_id.clone(),
            ));
        }
        rays.push(observation.prepare()?);
    }

    let mut candidate_ids = BTreeSet::new();
    let mut scores = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        if candidate.sensor_id.trim().is_empty() {
            return Err(TriangulationError::EmptySensorId);
        }
        if seen_ids.contains(candidate.sensor_id.as_str())
            || !candidate_ids.insert(candidate.sensor_id.as_str())
        {
            return Err(TriangulationError::DuplicateCandidateId(
                candidate.sensor_id.clone(),
            ));
        }

        let observation = predicted_candidate_bearing(candidate, assumed_target_ecef_m)?;
        let mut candidate_rays = rays.clone();
        candidate_rays.push(observation.prepare()?);
        let information = angular_information_matrix(&candidate_rays, assumed_target_ecef_m)?;
        let condition_number = symmetric_condition_number(information)?;
        let covariance_lower_bound = invert_well_conditioned(information, 1.0e14)?;
        let determinant = determinant3(information);
        if !determinant.is_finite() || determinant <= 0.0 {
            return Err(TriangulationError::ParallelOrIllConditioned { condition_number });
        }
        let log_determinant_information = determinant.ln();
        let a_optimal_variance_sum_m2 = covariance_lower_bound[0][0]
            + covariance_lower_bound[1][1]
            + covariance_lower_bound[2][2];

        scores.push(SensorPlacementScore {
            sensor_id: candidate.sensor_id.clone(),
            log_determinant_information,
            a_optimal_variance_sum_m2,
            condition_number,
            covariance_lower_bound_m2: covariance_lower_bound,
            observations_after_placement: candidate_rays.len(),
        });
    }

    scores.sort_by(|left, right| {
        right
            .log_determinant_information
            .total_cmp(&left.log_determinant_information)
            .then_with(|| {
                left.a_optimal_variance_sum_m2
                    .total_cmp(&right.a_optimal_variance_sum_m2)
            })
            .then_with(|| left.sensor_id.cmp(&right.sensor_id))
    });
    Ok(scores)
}

fn predicted_candidate_bearing(
    candidate: &SensorPlacementCandidate,
    target: Vector3,
) -> Result<RayObservation, TriangulationError> {
    if !candidate.origin_ecef_m.is_finite() {
        return Err(TriangulationError::NonFiniteVector);
    }
    let displacement = target.subtract(candidate.origin_ecef_m);
    let range = displacement.norm();
    if !range.is_finite() {
        return Err(TriangulationError::NonFiniteIntermediate);
    }
    if range <= VECTOR_EPSILON {
        return Err(TriangulationError::ZeroRange(candidate.sensor_id.clone()));
    }
    let direction_ecef = displacement.scale(1.0 / range);
    let direction_body = candidate
        .sensor_to_ecef
        .inverse_rotate_vector(direction_ecef)?;
    let horizontal = direction_body.x.hypot(direction_body.y);
    Ok(RayObservation {
        sensor_id: candidate.sensor_id.clone(),
        origin_ecef_m: candidate.origin_ecef_m,
        sensor_to_ecef: candidate.sensor_to_ecef,
        azimuth_rad: direction_body.y.atan2(direction_body.x),
        elevation_rad: direction_body.z.atan2(horizontal),
        sigma_azimuth_rad: candidate.sigma_azimuth_rad,
        sigma_elevation_rad: candidate.sigma_elevation_rad,
    })
}

fn angular_information_matrix(
    rays: &[PreparedRay],
    position: Vector3,
) -> Result<Matrix3, TriangulationError> {
    if rays.is_empty() {
        return Err(TriangulationError::InsufficientObservations);
    }
    let mut normal = [[0.0; 3]; 3];
    for ray in rays {
        let displacement = position.subtract(ray.origin);
        let range = displacement.norm();
        if !range.is_finite() {
            return Err(TriangulationError::NonFiniteIntermediate);
        }
        if range <= VECTOR_EPSILON {
            return Err(TriangulationError::ZeroRange(ray.sensor_id.clone()));
        }
        if displacement.dot(ray.direction) <= 0.0 {
            return Err(TriangulationError::BehindSensor(ray.sensor_id.clone()));
        }
        let jacobian_azimuth = ray.tangent_azimuth.scale(1.0 / (range * ray.cos_elevation));
        let jacobian_elevation = ray.tangent_elevation.scale(1.0 / range);
        matrix_add_outer(
            &mut normal,
            jacobian_azimuth,
            jacobian_azimuth,
            1.0 / ray.sigma_azimuth.powi(2),
        );
        matrix_add_outer(
            &mut normal,
            jacobian_elevation,
            jacobian_elevation,
            1.0 / ray.sigma_elevation.powi(2),
        );
    }
    if normal.iter().flatten().any(|value| !value.is_finite()) {
        return Err(TriangulationError::NonFiniteIntermediate);
    }
    Ok(normal)
}

fn determinant3(matrix: Matrix3) -> f64 {
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
}

fn line_intersection_normal_equations(rays: &[PreparedRay]) -> (Matrix3, Vector3) {
    let mut normal = [[0.0; 3]; 3];
    let mut rhs = Vector3::ZERO;
    for ray in rays {
        let direction = ray.direction;
        let projection = identity_minus_outer(direction);
        matrix_add_assign(&mut normal, projection);
        rhs = rhs.add(matrix_vector(projection, ray.origin));
    }
    (normal, rhs)
}

fn angular_normal_equations(
    rays: &[PreparedRay],
    position: Vector3,
) -> Result<(Matrix3, Vector3, f64, f64), TriangulationError> {
    let mut normal = [[0.0; 3]; 3];
    let mut gradient = Vector3::ZERO;
    let mut chi_square = 0.0;
    let mut sum_residual_squared = 0.0;

    for ray in rays {
        let displacement = position.subtract(ray.origin);
        let range = displacement.norm();
        if !range.is_finite() {
            return Err(TriangulationError::NonFiniteIntermediate);
        }
        if range <= VECTOR_EPSILON {
            return Err(TriangulationError::ZeroRange(ray.sensor_id.clone()));
        }
        if displacement.dot(ray.direction) <= 0.0 {
            return Err(TriangulationError::BehindSensor(ray.sensor_id.clone()));
        }

        let residual_azimuth = displacement.dot(ray.tangent_azimuth) / (range * ray.cos_elevation);
        let residual_elevation = displacement.dot(ray.tangent_elevation) / range;
        let jacobian_azimuth = ray.tangent_azimuth.scale(1.0 / (range * ray.cos_elevation));
        let jacobian_elevation = ray.tangent_elevation.scale(1.0 / range);
        let weight_azimuth = 1.0 / ray.sigma_azimuth.powi(2);
        let weight_elevation = 1.0 / ray.sigma_elevation.powi(2);

        matrix_add_outer(
            &mut normal,
            jacobian_azimuth,
            jacobian_azimuth,
            weight_azimuth,
        );
        matrix_add_outer(
            &mut normal,
            jacobian_elevation,
            jacobian_elevation,
            weight_elevation,
        );
        gradient = gradient
            .add(jacobian_azimuth.scale(weight_azimuth * residual_azimuth))
            .add(jacobian_elevation.scale(weight_elevation * residual_elevation));

        chi_square += weight_azimuth * residual_azimuth.powi(2)
            + weight_elevation * residual_elevation.powi(2);
        sum_residual_squared += residual_azimuth.powi(2) + residual_elevation.powi(2);
    }

    if !chi_square.is_finite() || !gradient.is_finite() {
        return Err(TriangulationError::NonFiniteIntermediate);
    }
    Ok((normal, gradient, chi_square, sum_residual_squared))
}

fn identity_minus_outer(direction: Vector3) -> Matrix3 {
    let values = [direction.x, direction.y, direction.z];
    let mut output = [[0.0; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            let identity_value = if row == column { 1.0 } else { 0.0 };
            output[row][column] = identity_value - values[row] * values[column];
        }
    }
    output
}

fn matrix_add_assign(target: &mut Matrix3, addend: Matrix3) {
    for row in 0..3 {
        for column in 0..3 {
            target[row][column] += addend[row][column];
        }
    }
}

fn matrix_add_outer(target: &mut Matrix3, left: Vector3, right: Vector3, weight: f64) {
    let left_values = [left.x, left.y, left.z];
    let right_values = [right.x, right.y, right.z];
    for row in 0..3 {
        for column in 0..3 {
            target[row][column] += weight * left_values[row] * right_values[column];
        }
    }
}

fn matrix_vector(matrix: Matrix3, vector: Vector3) -> Vector3 {
    let v = [vector.x, vector.y, vector.z];
    let mut out = [0.0; 3];
    for row in 0..3 {
        for column in 0..3 {
            out[row] += matrix[row][column] * v[column];
        }
    }
    Vector3::new(out[0], out[1], out[2])
}

fn invert_well_conditioned(
    matrix: Matrix3,
    max_condition_number: f64,
) -> Result<Matrix3, TriangulationError> {
    let condition_number = symmetric_condition_number(matrix)?;
    if !condition_number.is_finite() || condition_number > max_condition_number {
        return Err(TriangulationError::ParallelOrIllConditioned { condition_number });
    }
    invert3(matrix)
}

fn symmetric_condition_number(matrix: Matrix3) -> Result<f64, TriangulationError> {
    let mut diagonalized = matrix;
    for _ in 0..64 {
        let mut p = 0;
        let mut q = 1;
        let mut largest = diagonalized[p][q].abs();
        for row in 0..3 {
            for column in (row + 1)..3 {
                if diagonalized[row][column].abs() > largest {
                    largest = diagonalized[row][column].abs();
                    p = row;
                    q = column;
                }
            }
        }
        let scale = diagonalized
            .iter()
            .flat_map(|row| row.iter())
            .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
        if largest <= 1.0e-14 * scale.max(1.0) {
            break;
        }

        let off_diagonal = diagonalized[p][q];
        let tau = (diagonalized[q][q] - diagonalized[p][p]) / (2.0 * off_diagonal);
        let denominator = tau.abs() + tau.hypot(1.0);
        let t = if tau >= 0.0 {
            1.0 / denominator
        } else {
            -1.0 / denominator
        };
        let cosine = 1.0 / (1.0 + t * t).sqrt();
        let sine = t * cosine;
        let pp = diagonalized[p][p];
        let qq = diagonalized[q][q];

        for index in 0..3 {
            if index == p || index == q {
                continue;
            }
            let ip = diagonalized[index][p];
            let iq = diagonalized[index][q];
            diagonalized[index][p] = cosine * ip - sine * iq;
            diagonalized[p][index] = diagonalized[index][p];
            diagonalized[index][q] = sine * ip + cosine * iq;
            diagonalized[q][index] = diagonalized[index][q];
        }
        diagonalized[p][p] = pp - t * off_diagonal;
        diagonalized[q][q] = qq + t * off_diagonal;
        diagonalized[p][q] = 0.0;
        diagonalized[q][p] = 0.0;
    }

    let eigenvalues = [diagonalized[0][0], diagonalized[1][1], diagonalized[2][2]];
    let maximum = eigenvalues.iter().copied().fold(0.0_f64, f64::max);
    let minimum = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    if !maximum.is_finite() || !minimum.is_finite() || minimum <= 0.0 {
        return Err(TriangulationError::ParallelOrIllConditioned {
            condition_number: f64::INFINITY,
        });
    }
    Ok(maximum / minimum)
}

fn invert3(matrix: Matrix3) -> Result<Matrix3, TriangulationError> {
    let mut augmented = [[0.0_f64; 6]; 3];
    let scale = matrix
        .iter()
        .flat_map(|row| row.iter())
        .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
    if !scale.is_finite() || scale <= 0.0 {
        return Err(TriangulationError::NonFiniteIntermediate);
    }

    for row in 0..3 {
        for column in 0..3 {
            augmented[row][column] = matrix[row][column];
        }
        augmented[row][row + 3] = 1.0;
    }

    for pivot_column in 0..3 {
        let mut pivot_row = pivot_column;
        let mut pivot_magnitude = augmented[pivot_row][pivot_column].abs();
        for row in (pivot_column + 1)..3 {
            let candidate = augmented[row][pivot_column].abs();
            if candidate > pivot_magnitude {
                pivot_magnitude = candidate;
                pivot_row = row;
            }
        }
        if pivot_magnitude <= PIVOT_RELATIVE_TOLERANCE * scale {
            return Err(TriangulationError::ParallelOrIllConditioned {
                condition_number: f64::INFINITY,
            });
        }
        if pivot_row != pivot_column {
            augmented.swap(pivot_row, pivot_column);
        }
        let pivot = augmented[pivot_column][pivot_column];
        for column in 0..6 {
            augmented[pivot_column][column] /= pivot;
        }
        for row in 0..3 {
            if row == pivot_column {
                continue;
            }
            let factor = augmented[row][pivot_column];
            for column in 0..6 {
                augmented[row][column] -= factor * augmented[pivot_column][column];
            }
        }
    }

    let mut inverse = [[0.0; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            let value = augmented[row][column + 3];
            if !value.is_finite() {
                return Err(TriangulationError::NonFiniteIntermediate);
            }
            inverse[row][column] = value;
        }
    }
    Ok(inverse)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bearing_from_target(
        sensor_id: &str,
        origin: Vector3,
        target: Vector3,
        sigma: f64,
    ) -> RayObservation {
        let relative = target.subtract(origin);
        let horizontal = relative.x.hypot(relative.y);
        RayObservation {
            sensor_id: sensor_id.to_string(),
            origin_ecef_m: origin,
            sensor_to_ecef: Quaternion::IDENTITY,
            azimuth_rad: relative.y.atan2(relative.x),
            elevation_rad: relative.z.atan2(horizontal),
            sigma_azimuth_rad: sigma,
            sigma_elevation_rad: sigma,
        }
    }

    #[test]
    fn three_sensor_bearing_triangulation_recovers_known_ecef_target() {
        let target = Vector3::new(50.0, 40.0, 200.0);
        let observations = vec![
            bearing_from_target("eo-north", Vector3::new(0.0, 0.0, 0.0), target, 1.0e-4),
            bearing_from_target("eo-east", Vector3::new(100.0, 0.0, 0.0), target, 1.2e-4),
            bearing_from_target("eo-south", Vector3::new(0.0, 100.0, 5.0), target, 0.8e-4),
            bearing_from_target("eo-west", Vector3::new(100.0, 100.0, 10.0), target, 1.1e-4),
        ];
        let result = triangulate_bearings(&observations).unwrap();
        assert!(result.position_ecef_m.subtract(target).norm() < 1.0e-4);
        assert_eq!(result.observations, 4);
        assert_eq!(result.degrees_of_freedom, 5);
        assert!(result.rms_angular_residual_rad < 1.0e-8);
        assert!(result.condition_number.is_finite());
        assert!(result.covariance_ecef_m2[0][0] > 0.0);
        assert!(result.covariance_ecef_m2[1][1] > 0.0);
        assert!(result.covariance_ecef_m2[2][2] > 0.0);
    }

    #[test]
    fn sensor_to_ecef_quaternion_rotates_bearing_frame() {
        let ninety_degrees_about_z = Quaternion {
            w: (std::f64::consts::FRAC_PI_4).cos(),
            x: 0.0,
            y: 0.0,
            z: (std::f64::consts::FRAC_PI_4).sin(),
        };
        let rotated = ninety_degrees_about_z
            .rotate_vector(Vector3::new(1.0, 0.0, 0.0))
            .unwrap();
        assert!((rotated.x).abs() < 1.0e-12);
        assert!((rotated.y - 1.0).abs() < 1.0e-12);
        assert!(rotated.z.abs() < 1.0e-12);
    }

    #[test]
    fn noisy_bearings_produce_finite_uncertainty_and_low_normalized_residual() {
        let target = Vector3::new(50.0, 40.0, 200.0);
        let mut observations = vec![
            bearing_from_target("north", Vector3::new(0.0, 0.0, 0.0), target, 2.0e-4),
            bearing_from_target("east", Vector3::new(100.0, 0.0, 0.0), target, 2.0e-4),
            bearing_from_target("south", Vector3::new(0.0, 100.0, 5.0), target, 2.0e-4),
            bearing_from_target("west", Vector3::new(100.0, 100.0, 10.0), target, 2.0e-4),
        ];
        observations[0].azimuth_rad += 1.0e-5;
        observations[1].elevation_rad -= 1.5e-5;
        observations[2].azimuth_rad -= 0.8e-5;
        observations[3].elevation_rad += 1.2e-5;
        let result = triangulate_bearings(&observations).unwrap();
        assert!(result.position_ecef_m.subtract(target).norm() < 0.2);
        assert!(
            result
                .covariance_ecef_m2
                .iter()
                .flatten()
                .all(|value| value.is_finite())
        );
        assert!(result.residual_chi_square > 0.0);
        assert!(result.rms_angular_residual_rad < 1.0e-8);
        assert!(result.condition_number < 1.0e10);
    }

    #[test]
    fn fisher_information_and_sensor_placement_ranking_are_geometry_sensitive() {
        let target = Vector3::new(50.0, 40.0, 200.0);
        let existing = vec![
            bearing_from_target("north", Vector3::new(0.0, 0.0, 0.0), target, 1.0e-4),
            bearing_from_target("east", Vector3::new(100.0, 0.0, 0.0), target, 1.0e-4),
        ];
        let information = bearing_fisher_information(&existing, target).unwrap();
        assert!(information[0][0] > 0.0);
        assert!(information[1][1] > 0.0);
        assert!(information[2][2] > 0.0);

        let candidates = vec![
            SensorPlacementCandidate {
                sensor_id: "north-far".to_string(),
                origin_ecef_m: Vector3::new(-100.0, 0.0, 0.0),
                sensor_to_ecef: Quaternion::IDENTITY,
                sigma_azimuth_rad: 1.0e-4,
                sigma_elevation_rad: 1.0e-4,
            },
            SensorPlacementCandidate {
                sensor_id: "south".to_string(),
                origin_ecef_m: Vector3::new(0.0, 100.0, 0.0),
                sensor_to_ecef: Quaternion::IDENTITY,
                sigma_azimuth_rad: 1.0e-4,
                sigma_elevation_rad: 1.0e-4,
            },
            SensorPlacementCandidate {
                sensor_id: "high".to_string(),
                origin_ecef_m: Vector3::new(0.0, 0.0, 100.0),
                sensor_to_ecef: Quaternion::IDENTITY,
                sigma_azimuth_rad: 1.0e-4,
                sigma_elevation_rad: 1.0e-4,
            },
        ];
        let ranking = rank_sensor_placements(&existing, target, &candidates).unwrap();
        assert_eq!(ranking.len(), 3);
        assert_eq!(ranking[0].sensor_id, "high");
        assert!((ranking[0].log_determinant_information - 28.778_305_173_1).abs() < 1e-8);
        assert!((ranking[0].a_optimal_variance_sum_m2 - 5.541_708_743_8e-4).abs() < 1e-12);
        assert!((ranking[0].covariance_lower_bound_m2[2][2] - 4.938_144_56e-4).abs() < 1e-12);
        assert!(ranking.windows(2).all(|pair| {
            pair[0].log_determinant_information >= pair[1].log_determinant_information
        }));
        assert!(ranking.iter().all(|score| {
            score.condition_number.is_finite()
                && score.a_optimal_variance_sum_m2.is_finite()
                && score
                    .covariance_lower_bound_m2
                    .iter()
                    .flatten()
                    .all(|v| v.is_finite())
        }));
    }

    #[test]
    fn placement_ranking_rejects_duplicate_candidate_and_observed_ids() {
        let target = Vector3::new(50.0, 40.0, 200.0);
        let existing = vec![
            bearing_from_target("north", Vector3::new(0.0, 0.0, 0.0), target, 1.0e-4),
            bearing_from_target("east", Vector3::new(100.0, 0.0, 0.0), target, 1.0e-4),
        ];
        let candidate = SensorPlacementCandidate {
            sensor_id: "north".to_string(),
            origin_ecef_m: Vector3::new(0.0, 100.0, 0.0),
            sensor_to_ecef: Quaternion::IDENTITY,
            sigma_azimuth_rad: 1.0e-4,
            sigma_elevation_rad: 1.0e-4,
        };
        assert_eq!(
            rank_sensor_placements(&existing, target, &[candidate]),
            Err(TriangulationError::DuplicateCandidateId(
                "north".to_string()
            ))
        );
    }

    #[test]
    fn parallel_bearings_are_rejected_as_unobservable() {
        let observations = vec![
            RayObservation {
                sensor_id: "left".to_string(),
                origin_ecef_m: Vector3::new(0.0, 0.0, 0.0),
                sensor_to_ecef: Quaternion::IDENTITY,
                azimuth_rad: 0.0,
                elevation_rad: 0.0,
                sigma_azimuth_rad: 1.0e-3,
                sigma_elevation_rad: 1.0e-3,
            },
            RayObservation {
                sensor_id: "right".to_string(),
                origin_ecef_m: Vector3::new(100.0, 0.0, 0.0),
                sensor_to_ecef: Quaternion::IDENTITY,
                azimuth_rad: 0.0,
                elevation_rad: 0.0,
                sigma_azimuth_rad: 1.0e-3,
                sigma_elevation_rad: 1.0e-3,
            },
        ];
        assert!(matches!(
            triangulate_bearings(&observations),
            Err(TriangulationError::ParallelOrIllConditioned { .. })
        ));
    }

    #[test]
    fn insufficient_duplicate_and_invalid_records_fail_closed() {
        assert_eq!(
            triangulate_bearings(&[]),
            Err(TriangulationError::InsufficientObservations)
        );

        let origin = Vector3::ZERO;
        let target = Vector3::new(1.0, 2.0, 3.0);
        let mut duplicate = bearing_from_target("same", origin, target, 1.0e-3);
        let duplicate_two = duplicate.clone();
        duplicate.sensor_id = "same".to_string();
        assert_eq!(
            triangulate_bearings(&[duplicate, duplicate_two]),
            Err(TriangulationError::DuplicateSensorId("same".to_string()))
        );

        let mut invalid_quaternion = bearing_from_target("invalid", origin, target, 1.0e-3);
        invalid_quaternion.sensor_to_ecef = Quaternion {
            w: 0.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        let valid = bearing_from_target("valid", Vector3::new(3.0, 0.0, 0.0), target, 1.0e-3);
        assert_eq!(
            triangulate_bearings(&[invalid_quaternion, valid]),
            Err(TriangulationError::InvalidQuaternion)
        );
    }
}
