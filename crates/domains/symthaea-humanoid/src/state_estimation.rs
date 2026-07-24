// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic fused humanoid state estimation.
//!
//! The estimator accepts a sensor-neutral measurement contract so simulation,
//! replay, and hardware can use the same fusion path. It combines orientation,
//! angular/linear velocity, joint encoder state, and contact confidence while
//! preserving the morphology-specific state dimensions.

use serde::{Deserialize, Serialize};

use crate::contact::{BipedSupport, ContactFrame};
use crate::morphology::HumanoidMorphology;
use crate::types::HumanoidState;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateEstimatorConfig {
    /// Orientation correction applied per update. Lower values trust inertial
    /// integration, higher values trust the measured orientation.
    pub orientation_correction: f64,
    /// Low-pass correction for root linear velocity.
    pub linear_velocity_correction: f64,
    /// Low-pass correction for root angular velocity.
    pub angular_velocity_correction: f64,
    /// Low-pass correction for joint positions.
    pub joint_position_correction: f64,
    /// Low-pass correction for joint velocities.
    pub joint_velocity_correction: f64,
    /// Velocity damping applied while reliable double support is present.
    pub double_support_velocity_damping: f64,
    /// Maximum accepted measurement age.
    pub maximum_measurement_age_s: f64,
    /// Maximum accepted forward timestamp jump.
    pub maximum_dt_s: f64,
    /// Maximum orientation innovation accepted after prediction.
    pub maximum_orientation_innovation_rad: f64,
    /// Maximum root linear-velocity innovation accepted after prediction.
    pub maximum_linear_velocity_innovation_mps: f64,
    /// Maximum per-joint position innovation accepted after prediction.
    pub maximum_joint_position_innovation_rad: f64,
}

impl Default for StateEstimatorConfig {
    fn default() -> Self {
        Self {
            orientation_correction: 0.18,
            linear_velocity_correction: 0.30,
            angular_velocity_correction: 0.35,
            joint_position_correction: 0.80,
            joint_velocity_correction: 0.55,
            double_support_velocity_damping: 0.12,
            maximum_measurement_age_s: 0.08,
            maximum_dt_s: 0.10,
            maximum_orientation_innovation_rad: 1.20,
            maximum_linear_velocity_innovation_mps: 6.0,
            maximum_joint_position_innovation_rad: 1.50,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProprioceptiveMeasurement {
    pub morphology: HumanoidMorphology,
    pub sequence: u64,
    pub sampled_at_s: f64,
    pub received_at_s: f64,
    pub state: HumanoidState,
    pub contact: ContactFrame,
}

impl ProprioceptiveMeasurement {
    pub fn from_simulator(
        morphology: HumanoidMorphology,
        sequence: u64,
        state: HumanoidState,
        contact: ContactFrame,
    ) -> Self {
        let sampled_at_s = state.timestamp;
        Self {
            morphology,
            sequence,
            sampled_at_s,
            received_at_s: sampled_at_s,
            state,
            contact,
        }
    }

    pub fn age_s(&self) -> f64 {
        if !self.sampled_at_s.is_finite() || !self.received_at_s.is_finite() {
            return f64::INFINITY;
        }
        (self.received_at_s - self.sampled_at_s).max(0.0)
    }

    pub fn validate(&self) -> Result<(), StateEstimatorError> {
        if self.state.num_actuators() != self.morphology.num_actuators() {
            return Err(StateEstimatorError::MorphologyMismatch);
        }
        if !self.sampled_at_s.is_finite() || !self.received_at_s.is_finite() {
            return Err(StateEstimatorError::NonFiniteTimestamp);
        }
        if self.received_at_s < self.sampled_at_s {
            return Err(StateEstimatorError::TimestampRegression);
        }
        self.state
            .validate_for(self.morphology)
            .map_err(|_| StateEstimatorError::InvalidState)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StateEstimatorError {
    MorphologyMismatch,
    SequenceRegression,
    NonFiniteTimestamp,
    TimestampRegression,
    StaleMeasurement,
    InvalidState,
    InnovationRejected,
}

impl std::fmt::Display for StateEstimatorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MorphologyMismatch => {
                write!(f, "measurement morphology does not match estimator")
            }
            Self::SequenceRegression => write!(f, "measurement sequence did not advance"),
            Self::NonFiniteTimestamp => write!(f, "measurement timestamp is not finite"),
            Self::TimestampRegression => {
                write!(f, "measurement timestamp regressed or is future-dated")
            }
            Self::StaleMeasurement => write!(f, "measurement exceeded the configured age limit"),
            Self::InvalidState => write!(f, "measurement state failed validation"),
            Self::InnovationRejected => {
                write!(f, "measurement innovation exceeded estimator gates")
            }
        }
    }
}

impl std::error::Error for StateEstimatorError {}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct StateEstimatorReport {
    pub sequence: u64,
    pub measurement_age_s: f64,
    pub dt_s: f64,
    pub contact_trust: f32,
    pub support: BipedSupport,
    pub orientation_innovation_rad: f64,
    pub linear_velocity_innovation_mps: f64,
    pub maximum_joint_position_innovation_rad: f64,
    pub accepted: bool,
}

pub struct FusedHumanoidStateEstimator {
    morphology: HumanoidMorphology,
    config: StateEstimatorConfig,
    estimate: HumanoidState,
    last_sequence: Option<u64>,
    initialized: bool,
}

impl FusedHumanoidStateEstimator {
    pub fn new(morphology: HumanoidMorphology) -> Self {
        Self::with_config(morphology, StateEstimatorConfig::default())
    }

    pub fn with_config(morphology: HumanoidMorphology, config: StateEstimatorConfig) -> Self {
        Self {
            morphology,
            config,
            estimate: HumanoidState::standing_for(morphology),
            last_sequence: None,
            initialized: false,
        }
    }

    pub fn reset(&mut self, state: &HumanoidState) -> Result<(), StateEstimatorError> {
        state
            .validate_for(self.morphology)
            .map_err(|_| StateEstimatorError::InvalidState)?;
        self.estimate = state.clone();
        self.last_sequence = None;
        self.initialized = true;
        Ok(())
    }

    pub fn estimate(&self) -> &HumanoidState {
        &self.estimate
    }

    pub fn update(
        &mut self,
        measurement: &ProprioceptiveMeasurement,
    ) -> Result<(&HumanoidState, StateEstimatorReport), StateEstimatorError> {
        measurement.validate()?;
        if measurement.morphology != self.morphology {
            return Err(StateEstimatorError::MorphologyMismatch);
        }
        if let Some(last) = self.last_sequence {
            if measurement.sequence <= last {
                return Err(StateEstimatorError::SequenceRegression);
            }
        }
        if self.initialized && measurement.sampled_at_s < self.estimate.timestamp {
            return Err(StateEstimatorError::TimestampRegression);
        }
        let age_s = measurement.age_s();
        if age_s > self.config.maximum_measurement_age_s.max(0.0) {
            return Err(StateEstimatorError::StaleMeasurement);
        }

        if !self.initialized {
            self.estimate = measurement.state.clone();
            self.initialized = true;
        }

        let raw_dt = measurement.sampled_at_s - self.estimate.timestamp;
        let dt_s = raw_dt.max(0.0).min(self.config.maximum_dt_s.max(1.0e-4));
        let prior = self.estimate.clone();
        self.predict(dt_s);

        let orientation_innovation_rad = quaternion_angle(
            self.estimate.root_quaternion,
            measurement.state.root_quaternion,
        );
        let linear_velocity_innovation_mps = vector_distance(
            self.estimate.root_linear_velocity,
            measurement.state.root_linear_velocity,
        );
        let maximum_joint_position_innovation_rad = self
            .estimate
            .joint_angles
            .iter()
            .zip(measurement.state.joint_angles.iter())
            .map(|(predicted, measured)| (predicted - measured).abs())
            .fold(0.0, f64::max);
        if orientation_innovation_rad > self.config.maximum_orientation_innovation_rad.max(0.0)
            || linear_velocity_innovation_mps
                > self.config.maximum_linear_velocity_innovation_mps.max(0.0)
            || maximum_joint_position_innovation_rad
                > self.config.maximum_joint_position_innovation_rad.max(0.0)
        {
            self.estimate = prior;
            return Err(StateEstimatorError::InnovationRejected);
        }
        let contact_trust = measurement.contact.control_trust(
            measurement.received_at_s,
            self.config.maximum_measurement_age_s,
        );

        self.correct(&measurement.state, &measurement.contact, contact_trust);
        self.estimate.timestamp = measurement.sampled_at_s;
        self.last_sequence = Some(measurement.sequence);

        Ok((
            &self.estimate,
            StateEstimatorReport {
                sequence: measurement.sequence,
                measurement_age_s: age_s,
                dt_s,
                contact_trust,
                support: measurement.contact.support(),
                orientation_innovation_rad,
                linear_velocity_innovation_mps,
                maximum_joint_position_innovation_rad,
                accepted: true,
            },
        ))
    }

    fn predict(&mut self, dt_s: f64) {
        if dt_s <= 0.0 {
            return;
        }
        self.estimate.root_position[0] += self.estimate.root_linear_velocity[0] * dt_s;
        self.estimate.root_position[1] += self.estimate.root_linear_velocity[1] * dt_s;
        self.estimate.root_position[2] += self.estimate.root_linear_velocity[2] * dt_s;
        self.estimate.root_height = self.estimate.root_position[2];
        self.estimate.root_quaternion = integrate_quaternion(
            self.estimate.root_quaternion,
            self.estimate.root_angular_velocity,
            dt_s,
        );
        for (q, qd) in self
            .estimate
            .joint_angles
            .iter_mut()
            .zip(self.estimate.joint_velocities.iter())
        {
            *q += *qd * dt_s;
        }
    }

    fn correct(&mut self, measured: &HumanoidState, contacts: &ContactFrame, contact_trust: f32) {
        let orientation_gain = self.config.orientation_correction.clamp(0.0, 1.0);
        self.estimate.root_quaternion = nlerp_quaternion(
            self.estimate.root_quaternion,
            measured.root_quaternion,
            orientation_gain,
        );
        let linear_gain = self.config.linear_velocity_correction.clamp(0.0, 1.0);
        let angular_gain = self.config.angular_velocity_correction.clamp(0.0, 1.0);
        for i in 0..3 {
            self.estimate.root_linear_velocity[i] = blend(
                self.estimate.root_linear_velocity[i],
                measured.root_linear_velocity[i],
                linear_gain,
            );
            self.estimate.root_angular_velocity[i] = blend(
                self.estimate.root_angular_velocity[i],
                measured.root_angular_velocity[i],
                angular_gain,
            );
            self.estimate.com_velocity[i] = blend(
                self.estimate.com_velocity[i],
                measured.com_velocity[i],
                linear_gain,
            );
            self.estimate.torso_vertical[i] = blend(
                self.estimate.torso_vertical[i],
                measured.torso_vertical[i],
                orientation_gain,
            );
        }

        let q_gain = self.config.joint_position_correction.clamp(0.0, 1.0);
        let qd_gain = self.config.joint_velocity_correction.clamp(0.0, 1.0);
        for i in 0..self.estimate.joint_angles.len() {
            self.estimate.joint_angles[i] = blend(
                self.estimate.joint_angles[i],
                measured.joint_angles[i],
                q_gain,
            );
            self.estimate.joint_velocities[i] = blend(
                self.estimate.joint_velocities[i],
                measured.joint_velocities[i],
                qd_gain,
            );
        }

        self.estimate.root_position = measured.root_position;
        self.estimate.root_height = measured.root_height;
        self.estimate.head_height = measured.head_height;
        self.estimate.extremities.clone_from(&measured.extremities);

        if matches!(contacts.support(), BipedSupport::Double) {
            let damping = (self.config.double_support_velocity_damping * contact_trust as f64)
                .clamp(0.0, 0.95);
            self.estimate.root_linear_velocity[0] *= 1.0 - damping;
            self.estimate.root_linear_velocity[1] *= 1.0 - damping;
        }
    }
}

fn blend(a: f64, b: f64, gain: f64) -> f64 {
    a + (b - a) * gain
}

fn vector_distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn quaternion_angle(a: [f64; 4], b: [f64; 4]) -> f64 {
    let dot = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum::<f64>()
        .abs()
        .clamp(0.0, 1.0);
    2.0 * dot.acos()
}

fn nlerp_quaternion(a: [f64; 4], mut b: [f64; 4], gain: f64) -> [f64; 4] {
    let dot = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>();
    if dot < 0.0 {
        for value in &mut b {
            *value = -*value;
        }
    }
    normalize_quaternion([
        blend(a[0], b[0], gain),
        blend(a[1], b[1], gain),
        blend(a[2], b[2], gain),
        blend(a[3], b[3], gain),
    ])
}

fn integrate_quaternion(q: [f64; 4], omega: [f64; 3], dt: f64) -> [f64; 4] {
    let [w, x, y, z] = q;
    let [ox, oy, oz] = omega;
    normalize_quaternion([
        w + 0.5 * (-x * ox - y * oy - z * oz) * dt,
        x + 0.5 * (w * ox + y * oz - z * oy) * dt,
        y + 0.5 * (w * oy + z * ox - x * oz) * dt,
        z + 0.5 * (w * oz + x * oy - y * ox) * dt,
    ])
}

fn normalize_quaternion(q: [f64; 4]) -> [f64; 4] {
    let norm = q.iter().map(|value| value * value).sum::<f64>().sqrt();
    if norm <= 1.0e-12 {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_replayed_measurements() {
        let state = HumanoidState::standing();
        let contact = ContactFrame::estimated_from_state(&state, 0.05);
        let measurement =
            ProprioceptiveMeasurement::from_simulator(HumanoidMorphology::Dmc21, 7, state, contact);
        let mut estimator = FusedHumanoidStateEstimator::new(HumanoidMorphology::Dmc21);
        estimator.update(&measurement).unwrap();
        assert_eq!(
            estimator.update(&measurement).unwrap_err(),
            StateEstimatorError::SequenceRegression
        );
    }

    #[test]
    fn extreme_innovation_is_rejected_transactionally() {
        let state = HumanoidState::standing();
        let contact = ContactFrame::estimated_from_state(&state, 0.05);
        let first = ProprioceptiveMeasurement::from_simulator(
            HumanoidMorphology::Dmc21,
            1,
            state.clone(),
            contact,
        );
        let mut estimator = FusedHumanoidStateEstimator::new(HumanoidMorphology::Dmc21);
        estimator.update(&first).unwrap();
        let before = estimator.estimate().clone();
        let mut corrupted_state = state;
        corrupted_state.timestamp = 0.01;
        corrupted_state.root_quaternion = [0.0, 1.0, 0.0, 0.0];
        let corrupted = ProprioceptiveMeasurement::from_simulator(
            HumanoidMorphology::Dmc21,
            2,
            corrupted_state,
            ContactFrame::estimated_from_state(&before, 0.05),
        );
        assert_eq!(
            estimator.update(&corrupted).unwrap_err(),
            StateEstimatorError::InnovationRejected
        );
        assert_eq!(estimator.estimate().root_quaternion, before.root_quaternion);
    }

    #[test]
    fn double_support_damps_horizontal_velocity() {
        let mut state = HumanoidState::standing();
        state.root_linear_velocity = [1.0, 0.5, 0.0];
        state.extremities[8] = 0.0;
        state.extremities[11] = 0.0;
        let contact = ContactFrame::estimated_from_state(&state, 0.05);
        let measurement =
            ProprioceptiveMeasurement::from_simulator(HumanoidMorphology::Dmc21, 1, state, contact);
        let mut estimator = FusedHumanoidStateEstimator::new(HumanoidMorphology::Dmc21);
        let (estimate, report) = estimator.update(&measurement).unwrap();
        assert_eq!(report.support, BipedSupport::Double);
        assert!(estimate.root_linear_velocity[0] < 1.0);
    }

    #[test]
    fn quaternion_fusion_is_sign_invariant() {
        let mut estimator = FusedHumanoidStateEstimator::new(HumanoidMorphology::Dmc21);
        let state = HumanoidState::standing();
        estimator.reset(&state).unwrap();
        let mut measured = state.clone();
        measured.timestamp = 0.025;
        measured.root_quaternion = [-1.0, 0.0, 0.0, 0.0];
        let contact = ContactFrame::estimated_from_state(&measured, 0.05);
        let measurement = ProprioceptiveMeasurement::from_simulator(
            HumanoidMorphology::Dmc21,
            1,
            measured,
            contact,
        );
        let (estimate, report) = estimator.update(&measurement).unwrap();
        assert!(report.orientation_innovation_rad < 1.0e-9);
        assert!(estimate.root_quaternion[0].abs() > 0.999);
    }
    #[test]
    fn rejects_future_dated_measurement() {
        let state = HumanoidState::standing();
        let contact = ContactFrame::estimated_from_state(&state, 0.05);
        let mut measurement =
            ProprioceptiveMeasurement::from_simulator(HumanoidMorphology::Dmc21, 1, state, contact);
        measurement.received_at_s = measurement.sampled_at_s - 0.01;
        let mut estimator = FusedHumanoidStateEstimator::new(HumanoidMorphology::Dmc21);
        assert_eq!(
            estimator.update(&measurement).unwrap_err(),
            StateEstimatorError::TimestampRegression
        );
    }
}
