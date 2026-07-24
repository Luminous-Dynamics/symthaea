// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hardware proprioception adapter for the shared humanoid state estimator.
//!
//! The adapter converts calibrated SI-unit IMU, encoder, and foot-pressure data
//! into the sensor-neutral `ProprioceptiveMeasurement` contract. Kinematic
//! reconstruction is deliberately conservative: unavailable world quantities
//! are propagated from the previous estimate instead of being invented from
//! PWM commands.

use serde::{Deserialize, Serialize};

use crate::contact::{ContactFrame, ContactSource, FootContact};
use crate::morphology::HumanoidMorphology;
use crate::state_estimation::{ProprioceptiveMeasurement, StateEstimatorError};
use crate::types::HumanoidState;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProprioceptiveFrame {
    pub morphology: HumanoidMorphology,
    pub sequence: u64,
    pub sampled_at_s: f64,
    pub received_at_s: f64,
    pub calibration_fingerprint: u64,
    pub orientation_quaternion: [f64; 4],
    pub angular_velocity_rad_s: [f64; 3],
    pub linear_acceleration_mps2: [f64; 3],
    pub joint_positions_rad: Vec<f64>,
    pub joint_velocities_rad_s: Vec<f64>,
    pub right_normal_force_n: f64,
    pub left_normal_force_n: f64,
    pub right_center_of_pressure_m: [f64; 2],
    pub left_center_of_pressure_m: [f64; 2],
}

impl HardwareProprioceptiveFrame {
    pub fn validate(&self) -> Result<(), StateEstimatorError> {
        let joints = self.morphology.num_actuators();
        if self.joint_positions_rad.len() != joints || self.joint_velocities_rad_s.len() != joints {
            return Err(StateEstimatorError::MorphologyMismatch);
        }
        if !self.sampled_at_s.is_finite()
            || !self.received_at_s.is_finite()
            || self.received_at_s < self.sampled_at_s
        {
            return Err(StateEstimatorError::NonFiniteTimestamp);
        }
        if self.calibration_fingerprint == 0
            || self
                .orientation_quaternion
                .iter()
                .chain(self.angular_velocity_rad_s.iter())
                .chain(self.linear_acceleration_mps2.iter())
                .chain(self.joint_positions_rad.iter())
                .chain(self.joint_velocities_rad_s.iter())
                .chain(self.right_center_of_pressure_m.iter())
                .chain(self.left_center_of_pressure_m.iter())
                .any(|value| !value.is_finite())
            || !self.right_normal_force_n.is_finite()
            || !self.left_normal_force_n.is_finite()
            || self.right_normal_force_n < 0.0
            || self.left_normal_force_n < 0.0
        {
            return Err(StateEstimatorError::InvalidState);
        }
        let norm = self
            .orientation_quaternion
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if norm <= 1.0e-8 {
            return Err(StateEstimatorError::InvalidState);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareStateAdapterConfig {
    pub gravity_mps2: f64,
    pub maximum_dt_s: f64,
    pub support_force_threshold_n: f64,
    pub double_support_velocity_damping: f64,
    pub horizontal_acceleration_limit_mps2: f64,
}

impl Default for HardwareStateAdapterConfig {
    fn default() -> Self {
        Self {
            gravity_mps2: 9.806_65,
            maximum_dt_s: 0.05,
            support_force_threshold_n: 8.0,
            double_support_velocity_damping: 0.35,
            horizontal_acceleration_limit_mps2: 12.0,
        }
    }
}

pub struct HumanoidHardwareStateAdapter {
    morphology: HumanoidMorphology,
    config: HardwareStateAdapterConfig,
    previous: HumanoidState,
    last_sequence: Option<u64>,
    last_calibration_fingerprint: Option<u64>,
}

impl HumanoidHardwareStateAdapter {
    pub fn new(morphology: HumanoidMorphology) -> Self {
        Self::with_config(morphology, HardwareStateAdapterConfig::default())
    }

    pub fn with_config(morphology: HumanoidMorphology, config: HardwareStateAdapterConfig) -> Self {
        Self {
            morphology,
            config,
            previous: HumanoidState::standing_for(morphology),
            last_sequence: None,
            last_calibration_fingerprint: None,
        }
    }

    pub fn reset(&mut self, state: HumanoidState) -> Result<(), StateEstimatorError> {
        state
            .validate_for(self.morphology)
            .map_err(|_| StateEstimatorError::InvalidState)?;
        self.previous = state;
        self.last_sequence = None;
        self.last_calibration_fingerprint = None;
        Ok(())
    }

    pub fn adapt(
        &mut self,
        frame: &HardwareProprioceptiveFrame,
    ) -> Result<ProprioceptiveMeasurement, StateEstimatorError> {
        frame.validate()?;
        if frame.morphology != self.morphology {
            return Err(StateEstimatorError::MorphologyMismatch);
        }
        if self
            .last_sequence
            .map(|sequence| frame.sequence <= sequence)
            .unwrap_or(false)
        {
            return Err(StateEstimatorError::SequenceRegression);
        }
        if let Some(fingerprint) = self.last_calibration_fingerprint {
            if fingerprint != frame.calibration_fingerprint {
                return Err(StateEstimatorError::InvalidState);
            }
        }
        let dt = (frame.sampled_at_s - self.previous.timestamp)
            .clamp(0.0, self.config.maximum_dt_s.max(1.0e-4));
        let orientation = normalize_quaternion(frame.orientation_quaternion);
        let torso_vertical = rotate_body_z(orientation);
        let right_loaded = frame.right_normal_force_n >= self.config.support_force_threshold_n;
        let left_loaded = frame.left_normal_force_n >= self.config.support_force_threshold_n;
        let support_count = right_loaded as usize + left_loaded as usize;

        // The IMU acceleration is body-frame specific force. Rotate it into
        // world coordinates before removing gravity; integrating body-frame
        // components directly as world velocity produces false motion whenever
        // the torso is tilted.
        let specific_force_world = rotate_vector(orientation, frame.linear_acceleration_mps2);
        let world_acceleration = [
            specific_force_world[0],
            specific_force_world[1],
            specific_force_world[2] - self.config.gravity_mps2,
        ];
        let mut root_linear_velocity = self.previous.root_linear_velocity;
        for axis in 0..3 {
            let acceleration = world_acceleration[axis].clamp(
                -self.config.horizontal_acceleration_limit_mps2,
                self.config.horizontal_acceleration_limit_mps2,
            );
            root_linear_velocity[axis] += acceleration * dt;
        }
        if support_count == 2 {
            let damping = self.config.double_support_velocity_damping.clamp(0.0, 0.95);
            root_linear_velocity[0] *= 1.0 - damping;
            root_linear_velocity[1] *= 1.0 - damping;
        }
        let mut root_position = self.previous.root_position;
        for axis in 0..3 {
            root_position[axis] += root_linear_velocity[axis] * dt;
        }
        let reconstructed =
            reconstruct_extremities(self.morphology, root_position, &frame.joint_positions_rad);
        if support_count > 0 {
            let ground_height = if right_loaded && left_loaded {
                0.5 * (reconstructed.right_foot[2] + reconstructed.left_foot[2])
            } else if right_loaded {
                reconstructed.right_foot[2]
            } else {
                reconstructed.left_foot[2]
            };
            root_position[2] = (root_position[2] - ground_height).clamp(0.35, 1.8);
        }
        let reconstructed =
            reconstruct_extremities(self.morphology, root_position, &frame.joint_positions_rad);
        let mut extremities = vec![0.0; self.morphology.num_extremity_channels()];
        extremities[0..3].copy_from_slice(&reconstructed.right_hand);
        extremities[3..6].copy_from_slice(&reconstructed.left_hand);
        extremities[6..9].copy_from_slice(&reconstructed.right_foot);
        extremities[9..12].copy_from_slice(&reconstructed.left_foot);
        if extremities.len() >= 18 {
            extremities[12..15].copy_from_slice(&reconstructed.right_hand);
            extremities[15..18].copy_from_slice(&reconstructed.left_hand);
        }
        let state = HumanoidState {
            root_height: root_position[2],
            root_position,
            root_quaternion: orientation,
            joint_angles: frame.joint_positions_rad.clone(),
            root_linear_velocity,
            root_angular_velocity: frame.angular_velocity_rad_s,
            joint_velocities: frame.joint_velocities_rad_s.clone(),
            head_height: reconstructed.head[2],
            torso_vertical,
            extremities,
            com_velocity: root_linear_velocity,
            timestamp: frame.sampled_at_s,
        };
        state
            .validate_for(self.morphology)
            .map_err(|_| StateEstimatorError::InvalidState)?;
        let contact = ContactFrame {
            right: pressure_contact(
                right_loaded,
                reconstructed.right_foot,
                frame.right_normal_force_n,
                frame.right_center_of_pressure_m,
            ),
            left: pressure_contact(
                left_loaded,
                reconstructed.left_foot,
                frame.left_normal_force_n,
                frame.left_center_of_pressure_m,
            ),
            source: ContactSource::ForceSensor,
            timestamp: frame.sampled_at_s,
        };
        self.previous = state.clone();
        self.last_sequence = Some(frame.sequence);
        self.last_calibration_fingerprint = Some(frame.calibration_fingerprint);
        Ok(ProprioceptiveMeasurement {
            morphology: self.morphology,
            sequence: frame.sequence,
            sampled_at_s: frame.sampled_at_s,
            received_at_s: frame.received_at_s,
            state,
            contact,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct ReconstructedExtremities {
    right_hand: [f64; 3],
    left_hand: [f64; 3],
    right_foot: [f64; 3],
    left_foot: [f64; 3],
    head: [f64; 3],
}

fn reconstruct_extremities(
    morphology: HumanoidMorphology,
    root: [f64; 3],
    joints: &[f64],
) -> ReconstructedExtremities {
    let names = morphology.joint_names();
    let angle = |name: &str| {
        names
            .iter()
            .position(|candidate| candidate == name)
            .and_then(|index| joints.get(index).copied())
            .unwrap_or(0.0)
    };
    let foot = |prefix: &str, lateral: f64| {
        let hip_pitch = angle(&format!("{prefix}hip_y"));
        let knee = angle(&format!("{prefix}knee"));
        let ankle = angle(&format!("{prefix}ankle_y"));
        let thigh = 0.34;
        let shin = 0.32;
        let x = root[0]
            + thigh * hip_pitch.sin()
            + shin * (hip_pitch + knee).sin()
            + 0.08 * (hip_pitch + knee + ankle).cos();
        let z = root[2] - thigh * hip_pitch.cos() - shin * (hip_pitch + knee).cos();
        [x, root[1] + lateral, z]
    };
    let hand = |prefix: &str, lateral: f64| {
        let shoulder = angle(&format!("{prefix}shoulder1"));
        let elbow = angle(&format!("{prefix}elbow"));
        let x = root[0] + 0.28 * shoulder.sin() + 0.25 * (shoulder + elbow).sin();
        let z = root[2] + 0.22 - 0.28 * shoulder.cos() - 0.25 * (shoulder + elbow).cos();
        [x, root[1] + lateral, z]
    };
    ReconstructedExtremities {
        right_hand: hand("right_", -0.24),
        left_hand: hand("left_", 0.24),
        right_foot: foot("right_", -0.10),
        left_foot: foot("left_", 0.10),
        head: [root[0], root[1], root[2] + 0.46],
    }
}

fn pressure_contact(
    in_contact: bool,
    point_world_m: [f64; 3],
    normal_force_n: f64,
    center_of_pressure_m: [f64; 2],
) -> FootContact {
    FootContact {
        in_contact,
        point_world_m,
        force_world_n: [0.0, 0.0, normal_force_n],
        torque_world_nm: [0.0; 3],
        center_of_pressure_world_m: [
            point_world_m[0] + center_of_pressure_m[0],
            point_world_m[1] + center_of_pressure_m[1],
        ],
        confidence: if in_contact { 1.0 } else { 0.7 },
    }
}

fn normalize_quaternion(q: [f64; 4]) -> [f64; 4] {
    let norm = q.iter().map(|value| value * value).sum::<f64>().sqrt();
    if norm <= 1.0e-12 {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm]
    }
}

fn rotate_vector(q: [f64; 4], value: [f64; 3]) -> [f64; 3] {
    let [w, x, y, z] = normalize_quaternion(q);
    let twice_cross = [
        2.0 * (y * value[2] - z * value[1]),
        2.0 * (z * value[0] - x * value[2]),
        2.0 * (x * value[1] - y * value[0]),
    ];
    let second_cross = [
        y * twice_cross[2] - z * twice_cross[1],
        z * twice_cross[0] - x * twice_cross[2],
        x * twice_cross[1] - y * twice_cross[0],
    ];
    [
        value[0] + w * twice_cross[0] + second_cross[0],
        value[1] + w * twice_cross[1] + second_cross[1],
        value[2] + w * twice_cross[2] + second_cross[2],
    ]
}

fn rotate_body_z(q: [f64; 4]) -> [f64; 3] {
    let [w, x, y, z] = q;
    [
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(sequence: u64) -> HardwareProprioceptiveFrame {
        HardwareProprioceptiveFrame {
            morphology: HumanoidMorphology::Dmc21,
            sequence,
            sampled_at_s: sequence as f64 * 0.01,
            received_at_s: sequence as f64 * 0.01,
            calibration_fingerprint: 7,
            orientation_quaternion: [1.0, 0.0, 0.0, 0.0],
            angular_velocity_rad_s: [0.0; 3],
            linear_acceleration_mps2: [0.0, 0.0, 9.806_65],
            joint_positions_rad: vec![0.0; 21],
            joint_velocities_rad_s: vec![0.0; 21],
            right_normal_force_n: 340.0,
            left_normal_force_n: 340.0,
            right_center_of_pressure_m: [0.0; 2],
            left_center_of_pressure_m: [0.0; 2],
        }
    }

    #[test]
    fn calibrated_hardware_frame_becomes_estimator_measurement() {
        let mut adapter = HumanoidHardwareStateAdapter::new(HumanoidMorphology::Dmc21);
        let measurement = adapter.adapt(&frame(1)).unwrap();
        assert_eq!(measurement.sequence, 1);
        assert_eq!(measurement.contact.source, ContactSource::ForceSensor);
        assert!(measurement.contact.right.in_contact);
        assert!(
            measurement
                .state
                .validate_for(HumanoidMorphology::Dmc21)
                .is_ok()
        );
    }

    #[test]
    fn tilted_specific_force_is_rotated_before_gravity_removal() {
        let mut adapter = HumanoidHardwareStateAdapter::new(HumanoidMorphology::Dmc21);
        let mut first = frame(1);
        let half = std::f64::consts::FRAC_PI_4;
        first.orientation_quaternion = [half.cos(), 0.0, half.sin(), 0.0];
        // For a +90 degree body-to-world pitch, world up is -X in body coordinates.
        first.linear_acceleration_mps2 = [-9.806_65, 0.0, 0.0];
        let measurement = adapter.adapt(&first).unwrap();
        assert!(measurement.state.root_linear_velocity[0].abs() < 1.0e-9);
        assert!(measurement.state.root_linear_velocity[1].abs() < 1.0e-9);
        assert!(measurement.state.root_linear_velocity[2].abs() < 1.0e-9);
    }

    #[test]
    fn calibration_identity_cannot_change_mid_stream() {
        let mut adapter = HumanoidHardwareStateAdapter::new(HumanoidMorphology::Dmc21);
        adapter.adapt(&frame(1)).unwrap();
        let mut changed = frame(2);
        changed.calibration_fingerprint = 8;
        assert_eq!(
            adapter.adapt(&changed).unwrap_err(),
            StateEstimatorError::InvalidState
        );
    }

    #[test]
    fn replayed_sequence_is_rejected() {
        let mut adapter = HumanoidHardwareStateAdapter::new(HumanoidMorphology::Dmc21);
        adapter.adapt(&frame(1)).unwrap();
        assert_eq!(
            adapter.adapt(&frame(1)).unwrap_err(),
            StateEstimatorError::SequenceRegression
        );
    }
}
