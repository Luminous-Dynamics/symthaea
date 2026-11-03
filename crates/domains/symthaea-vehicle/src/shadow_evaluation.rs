// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Same-observation shadow comparison for the road-vehicle research domain.
//!
//! The advanced controller proposal and existing PD road-teacher proposal are
//! evaluated on the exact same frozen state. This module records proposal
//! disagreement; it does not run a simulator, choose an actuator command,
//! establish safety, or compare closed-loop trajectories.

use blake3::Hasher;
use serde::{Deserialize, Serialize};

use crate::types::{VehicleCommand, VehicleState, pd_cruise_baseline_with_road};

pub const VEHICLE_SHADOW_SCHEMA: &str = "symthaea.vehicle.shadow-comparison.v0.1";
pub const PD_ROAD_BASELINE_POLICY_ID: &str = "symthaea.vehicle.pd-road-baseline.v1";

const PD_SPEED_GAIN: f64 = 0.3;
const PD_BRAKE_DEADBAND_MPS: f64 = 1.0;
const PD_BRAKE_GAIN: f64 = 0.2;
const PD_LANE_GAIN: f64 = 0.8;
const PD_HEADING_GAIN: f64 = 2.0;
const PD_CURVATURE_FEED_FORWARD_GAIN: f64 = 5.0;

/// Stable profile identity for the repository's existing PD teacher baseline.
///
/// This binds the exact algorithm name and fixed gain bytes. It is not a
/// source/binary attestation and does not imply architectural independence.
pub fn pd_road_baseline_digest() -> String {
    let mut hasher = Hasher::new();
    hasher.update(PD_ROAD_BASELINE_POLICY_ID.as_bytes());
    for value in [
        PD_SPEED_GAIN,
        PD_BRAKE_DEADBAND_MPS,
        PD_BRAKE_GAIN,
        PD_LANE_GAIN,
        PD_HEADING_GAIN,
        PD_CURVATURE_FEED_FORWARD_GAIN,
    ] {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VehicleShadowError {
    EmptyCandidateId,
    InvalidAdvancedControllerDigest,
    ControllerIdentityAliased,
    InvalidStepSequence,
    TimestampNonMonotonic,
    StateInvalid,
    InputInvalid,
    AdvancedCommandInvalid,
    BaselineCommandInvalid,
}

/// Frozen identity for one scenario/controller comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VehicleShadowProfile {
    pub schema_version: String,
    pub candidate_id: String,
    pub advanced_controller_digest: String,
    pub baseline_policy_id: String,
    pub baseline_controller_digest: String,
}

/// One immutable input frame. The advanced command must have been produced
/// from this exact state by the frozen advanced-controller subject.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VehicleShadowInput {
    pub step_index: u64,
    pub state: VehicleState,
    pub road_heading_rad: f64,
    pub target_speed_mps: f64,
    pub advanced_command: VehicleCommand,
}

/// Per-frame paired proposal record. No command is applied to a simulator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VehicleShadowRecord {
    pub schema_version: String,
    pub candidate_id: String,
    pub step_index: u64,
    pub timestamp_s: f64,
    pub state_digest: String,
    pub advanced_controller_digest: String,
    pub baseline_policy_id: String,
    pub baseline_controller_digest: String,
    pub advanced_command: [f32; 3],
    pub baseline_command: [f32; 3],
    pub absolute_command_delta: [f32; 3],
    pub command_l1_delta: f32,
    pub advanced_throttle_brake_overlap: bool,
    pub baseline_throttle_brake_overlap: bool,
}

/// Distributional summary of proposal disagreement for one candidate.
///
/// These are controller-output measurements, not safety outcomes. The
/// evaluator deliberately stores all records so the full trace is retained.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VehicleShadowReport {
    pub schema_version: String,
    pub candidate_id: String,
    pub advanced_controller_digest: String,
    pub baseline_policy_id: String,
    pub baseline_controller_digest: String,
    pub steps: usize,
    pub mean_absolute_command_delta: [f64; 3],
    pub maximum_absolute_command_delta: [f32; 3],
    pub command_l1_delta_p50: f32,
    pub command_l1_delta_p95: f32,
    pub command_l1_delta_p99: f32,
    pub advanced_throttle_brake_overlap_steps: usize,
    pub baseline_throttle_brake_overlap_steps: usize,
    pub records: Vec<VehicleShadowRecord>,
}

pub struct VehicleShadowEvaluator {
    profile: VehicleShadowProfile,
    records: Vec<VehicleShadowRecord>,
    absolute_delta_sums: [f64; 3],
    absolute_delta_maxima: [f32; 3],
    command_l1_deltas: Vec<f32>,
    advanced_overlap_steps: usize,
    baseline_overlap_steps: usize,
    last_timestamp_s: Option<f64>,
}

impl VehicleShadowEvaluator {
    pub fn new(
        candidate_id: impl Into<String>,
        advanced_controller_digest: impl Into<String>,
    ) -> Result<Self, VehicleShadowError> {
        let candidate_id = candidate_id.into();
        let advanced_controller_digest = advanced_controller_digest.into();
        let baseline_controller_digest = pd_road_baseline_digest();

        if candidate_id.trim().is_empty() {
            return Err(VehicleShadowError::EmptyCandidateId);
        }
        if !digest_is_valid(&advanced_controller_digest) {
            return Err(VehicleShadowError::InvalidAdvancedControllerDigest);
        }
        if advanced_controller_digest.eq_ignore_ascii_case(&baseline_controller_digest) {
            return Err(VehicleShadowError::ControllerIdentityAliased);
        }

        Ok(Self {
            profile: VehicleShadowProfile {
                schema_version: VEHICLE_SHADOW_SCHEMA.into(),
                candidate_id,
                advanced_controller_digest,
                baseline_policy_id: PD_ROAD_BASELINE_POLICY_ID.into(),
                baseline_controller_digest,
            },
            records: Vec::new(),
            absolute_delta_sums: [0.0; 3],
            absolute_delta_maxima: [0.0; 3],
            command_l1_deltas: Vec::new(),
            advanced_overlap_steps: 0,
            baseline_overlap_steps: 0,
            last_timestamp_s: None,
        })
    }

    pub fn profile(&self) -> &VehicleShadowProfile {
        &self.profile
    }

    /// Compare both controller proposals on one identical observation.
    ///
    /// The input state must already contain the road-relative fields expected
    /// by the PD baseline. This function does not mutate the state.
    pub fn observe(
        &mut self,
        input: &VehicleShadowInput,
    ) -> Result<&VehicleShadowRecord, VehicleShadowError> {
        if input.step_index != self.records.len() as u64 {
            return Err(VehicleShadowError::InvalidStepSequence);
        }
        if !state_is_finite(&input.state) {
            return Err(VehicleShadowError::StateInvalid);
        }
        if !input.road_heading_rad.is_finite()
            || !input.target_speed_mps.is_finite()
            || input.target_speed_mps < 0.0
        {
            return Err(VehicleShadowError::InputInvalid);
        }
        if self
            .last_timestamp_s
            .is_some_and(|previous| input.state.timestamp <= previous)
        {
            return Err(VehicleShadowError::TimestampNonMonotonic);
        }
        if !command_is_valid(&input.advanced_command) {
            return Err(VehicleShadowError::AdvancedCommandInvalid);
        }

        let baseline_command = pd_cruise_baseline_with_road(
            &input.state,
            input.target_speed_mps,
            input.road_heading_rad,
        );
        if !command_is_valid(&baseline_command) {
            return Err(VehicleShadowError::BaselineCommandInvalid);
        }

        let advanced_values = command_values(&input.advanced_command);
        let baseline_values = command_values(&baseline_command);
        let absolute_command_delta = [
            (advanced_values[0] - baseline_values[0]).abs(),
            (advanced_values[1] - baseline_values[1]).abs(),
            (advanced_values[2] - baseline_values[2]).abs(),
        ];
        let command_l1_delta = absolute_command_delta.iter().sum();

        let state_bytes =
            serde_json::to_vec(&input.state).map_err(|_| VehicleShadowError::StateInvalid)?;
        let state_digest = blake3::hash(&state_bytes).to_hex().to_string();

        let record = VehicleShadowRecord {
            schema_version: VEHICLE_SHADOW_SCHEMA.into(),
            candidate_id: self.profile.candidate_id.clone(),
            step_index: input.step_index,
            timestamp_s: input.state.timestamp,
            state_digest,
            advanced_controller_digest: self.profile.advanced_controller_digest.clone(),
            baseline_policy_id: self.profile.baseline_policy_id.clone(),
            baseline_controller_digest: self.profile.baseline_controller_digest.clone(),
            advanced_command: advanced_values,
            baseline_command: baseline_values,
            absolute_command_delta,
            command_l1_delta,
            advanced_throttle_brake_overlap: throttle_brake_overlap(&input.advanced_command),
            baseline_throttle_brake_overlap: throttle_brake_overlap(&baseline_command),
        };

        for axis in 0..3 {
            self.absolute_delta_sums[axis] += absolute_command_delta[axis] as f64;
            self.absolute_delta_maxima[axis] =
                self.absolute_delta_maxima[axis].max(absolute_command_delta[axis]);
        }
        self.command_l1_deltas.push(command_l1_delta);
        if record.advanced_throttle_brake_overlap {
            self.advanced_overlap_steps += 1;
        }
        if record.baseline_throttle_brake_overlap {
            self.baseline_overlap_steps += 1;
        }
        self.last_timestamp_s = Some(input.state.timestamp);
        self.records.push(record);

        Ok(self.records.last().expect("record was just appended"))
    }

    pub fn report(&self) -> VehicleShadowReport {
        let steps = self.records.len();
        let denominator = steps.max(1) as f64;
        let mut sorted_l1 = self.command_l1_deltas.clone();
        sorted_l1.sort_by(f32::total_cmp);

        VehicleShadowReport {
            schema_version: VEHICLE_SHADOW_SCHEMA.into(),
            candidate_id: self.profile.candidate_id.clone(),
            advanced_controller_digest: self.profile.advanced_controller_digest.clone(),
            baseline_policy_id: self.profile.baseline_policy_id.clone(),
            baseline_controller_digest: self.profile.baseline_controller_digest.clone(),
            steps,
            mean_absolute_command_delta: [
                self.absolute_delta_sums[0] / denominator,
                self.absolute_delta_sums[1] / denominator,
                self.absolute_delta_sums[2] / denominator,
            ],
            maximum_absolute_command_delta: self.absolute_delta_maxima,
            command_l1_delta_p50: percentile(&sorted_l1, 0.50),
            command_l1_delta_p95: percentile(&sorted_l1, 0.95),
            command_l1_delta_p99: percentile(&sorted_l1, 0.99),
            advanced_throttle_brake_overlap_steps: self.advanced_overlap_steps,
            baseline_throttle_brake_overlap_steps: self.baseline_overlap_steps,
            records: self.records.clone(),
        }
    }
}

fn percentile(sorted: &[f32], probability: f64) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((sorted.len() - 1) as f64 * probability).ceil() as usize;
    sorted[index.min(sorted.len() - 1)]
}

fn digest_is_valid(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn state_is_finite(state: &VehicleState) -> bool {
    [
        state.speed,
        state.lateral_velocity,
        state.yaw_rate,
        state.heading,
        state.position_x,
        state.position_y,
        state.steering_angle,
        state.throttle_position,
        state.brake_pressure,
        state.longitudinal_accel,
        state.lateral_accel,
        state.tire_slip_front,
        state.tire_slip_rear,
        state.pitch,
        state.lateral_offset,
        state.curvature_ahead,
        state.nearest_peer_distance,
        state.peer_relative_speed,
        state.mesh_brake_density,
        state.mesh_confidence,
        state.timestamp,
    ]
    .iter()
    .all(|value| value.is_finite())
}

fn command_is_valid(command: &VehicleCommand) -> bool {
    command.steering.is_finite()
        && (-1.0..=1.0).contains(&command.steering)
        && command.throttle.is_finite()
        && (0.0..=1.0).contains(&command.throttle)
        && command.brake.is_finite()
        && (0.0..=1.0).contains(&command.brake)
}

fn command_values(command: &VehicleCommand) -> [f32; 3] {
    [command.steering, command.throttle, command.brake]
}

fn throttle_brake_overlap(command: &VehicleCommand) -> bool {
    command.throttle > 0.05 && command.brake > 0.05
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evaluator() -> VehicleShadowEvaluator {
        VehicleShadowEvaluator::new("synthetic-lane-keep-001", "a".repeat(64))
            .expect("valid advanced identity")
    }

    fn input(step_index: u64, timestamp_s: f64) -> VehicleShadowInput {
        let mut state = VehicleState::cruising(10.0);
        state.timestamp = timestamp_s;
        VehicleShadowInput {
            step_index,
            state,
            road_heading_rad: 0.0,
            target_speed_mps: 13.4,
            advanced_command: VehicleCommand {
                steering: 0.0,
                throttle: 0.5,
                brake: 0.0,
            },
        }
    }

    #[test]
    fn compares_advanced_and_pd_proposals_on_the_same_state() {
        let mut evaluator = evaluator();
        let record = evaluator.observe(&input(0, 0.0)).unwrap();
        assert_eq!(record.candidate_id, "synthetic-lane-keep-001");
        assert_eq!(record.baseline_policy_id, PD_ROAD_BASELINE_POLICY_ID);
        assert_eq!(record.baseline_command[0], 0.0);
        assert!(record.baseline_command[1] > 0.0);
        assert!(record.command_l1_delta.is_finite());
        assert_eq!(record.step_index, 0);
    }

    #[test]
    fn report_preserves_trace_and_distributional_disagreement() {
        let mut evaluator = evaluator();
        evaluator.observe(&input(0, 0.0)).unwrap();
        let mut second = input(1, 0.005);
        second.advanced_command.steering = 0.4;
        second.advanced_command.throttle = 0.0;
        second.advanced_command.brake = 0.2;
        evaluator.observe(&second).unwrap();

        let report = evaluator.report();
        assert_eq!(report.steps, 2);
        assert_eq!(report.records.len(), 2);
        assert!(report.maximum_absolute_command_delta[0] > 0.0);
        assert!(report.command_l1_delta_p99 >= report.command_l1_delta_p50);
    }

    #[test]
    fn rejects_empty_candidate_or_invalid_controller_identity() {
        assert_eq!(
            VehicleShadowEvaluator::new(" ", "a".repeat(64)).err(),
            Some(VehicleShadowError::EmptyCandidateId)
        );
        assert_eq!(
            VehicleShadowEvaluator::new("candidate", "invalid").err(),
            Some(VehicleShadowError::InvalidAdvancedControllerDigest)
        );
    }

    #[test]
    fn rejects_controller_identity_aliasing() {
        let digest = pd_road_baseline_digest();
        assert_eq!(
            VehicleShadowEvaluator::new("candidate", digest).err(),
            Some(VehicleShadowError::ControllerIdentityAliased)
        );
    }

    #[test]
    fn rejects_non_monotonic_steps_and_timestamps() {
        let mut evaluator = evaluator();
        evaluator.observe(&input(0, 0.0)).unwrap();
        assert_eq!(
            evaluator.observe(&input(2, 0.01)).err(),
            Some(VehicleShadowError::InvalidStepSequence)
        );
        assert_eq!(
            evaluator.observe(&input(1, 0.0)).err(),
            Some(VehicleShadowError::TimestampNonMonotonic)
        );
    }

    #[test]
    fn rejects_invalid_state_inputs_and_commands() {
        let mut evaluator = evaluator();
        let mut invalid_state = input(0, 0.0);
        invalid_state.state.brake_pressure = f64::NAN;
        assert_eq!(
            evaluator.observe(&invalid_state).err(),
            Some(VehicleShadowError::StateInvalid)
        );

        let mut invalid_heading = input(0, 0.0);
        invalid_heading.road_heading_rad = f64::INFINITY;
        assert_eq!(
            evaluator.observe(&invalid_heading).err(),
            Some(VehicleShadowError::InputInvalid)
        );

        let mut invalid_command = input(0, 0.0);
        invalid_command.advanced_command.throttle = f32::NAN;
        assert_eq!(
            evaluator.observe(&invalid_command).err(),
            Some(VehicleShadowError::AdvancedCommandInvalid)
        );
    }

    #[test]
    fn baseline_profile_identity_is_deterministic() {
        assert_eq!(pd_road_baseline_digest(), pd_road_baseline_digest());
        assert_eq!(pd_road_baseline_digest().len(), 64);
        assert!(digest_is_valid(&pd_road_baseline_digest()));
    }
}
