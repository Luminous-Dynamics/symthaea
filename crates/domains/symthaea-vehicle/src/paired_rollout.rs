// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Paired closed-loop lane-keeping rollouts on cloned bicycle-model plants.
//!
//! The advanced HDC-LTC controller and the existing PD road baseline receive
//! the same exogenous disturbance schedule and begin from the same cloned
//! simulator state. Each plant then evolves on its own state. This is a
//! simulator-only comparison primitive; it contains no pedestrian actors,
//! intersection traffic, FEP agent, RTA authority or hardware interface.

use serde::{Deserialize, Serialize};

use crate::classical_baselines::VehicleClassicalBaseline;
use crate::controller::VehicleController;
use crate::encoder::VehicleHdcEncoder;
use crate::perturbations::{PerturbationSchedule, VehiclePerturbation};
use crate::road::Road;
use crate::simulator::{BicycleModelSimulator, VehiclePhysicsSimulator};
use crate::types::{
    VehicleCommand, VehicleState,
};

pub const VEHICLE_PAIRED_ROLLOUT_SCHEMA: &str = "symthaea.vehicle.paired-rollout.v0.1";
pub const MAX_PAIRED_ROLLOUT_STEPS: usize = 100_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VehiclePairedRolloutError {
    EmptyCandidateId,
    EmptyRoadProfileId,
    InvalidRoadProfile,
    InvalidStateDuringRollout,
    InvalidAdvancedControllerDigest,
    InvalidBaselinePolicy,
    ControllerIdentityAliased,
    InvalidTargetSpeed,
    InvalidTimestep,
    InvalidStepBudget,
    InvalidInitialState,
    InvalidRolloutProfile,
    AdvancedCommandInvalid,
    BaselineCommandInvalid,
}

/// Frozen inputs for a matched closed-loop lane-keeping experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VehiclePairedRolloutConfig {
    pub candidate_id: String,
    /// Caller-supplied identity for the exact Road profile used by the run.
    pub road_profile_id: String,
    /// Digest of resolved road segments and projection semantics.
    pub road_geometry_digest: String,
    /// Caller-supplied controller/config digest; not authenticated here.
    pub advanced_controller_digest: String,
    pub baseline_policy_id: String,
    pub baseline_controller_digest: String,
    pub simulator_profile_digest: String,
    pub initial_state_digest: String,
    pub perturbation_schedule_digest: String,
    pub target_speed_mps: f64,
    pub timestep_s: f64,
    pub steps: usize,
    /// BLAKE3 over this struct's serde_json representation before this field is populated.
    pub experiment_profile_digest: String,
}

impl VehiclePairedRolloutConfig {
    /// Verify that all frozen experiment inputs still match their profile digest.
    pub fn identity_is_valid(&self) -> bool {
        let mut profile = self.clone();
        let expected = profile.experiment_profile_digest.clone();
        profile.experiment_profile_digest.clear();
        serde_json::to_vec(&profile)
            .ok()
            .map(|bytes| blake3::hash(&bytes).to_hex().to_string() == expected)
            .unwrap_or(false)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VehiclePairedRolloutFrame {
    pub step_index: usize,
    pub elapsed_s: f64,
    /// Physical plant states before each controller acts.
    pub advanced_plant_state_before: VehicleState,
    pub baseline_plant_state_before: VehicleState,
    /// Road-relative and sensor-perturbed observations delivered to each controller.
    pub advanced_observation_state: VehicleState,
    pub baseline_observation_state: VehicleState,
    pub advanced_command: VehicleCommand,
    pub baseline_command: VehicleCommand,
    pub advanced_plant_state_after: VehicleState,
    pub baseline_plant_state_after: VehicleState,
    pub advanced_progress_m: f64,
    pub baseline_progress_m: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VehicleTrajectoryMetrics {
    pub steps: usize,
    pub mean_absolute_lateral_offset_m: f64,
    pub rms_lateral_offset_m: f64,
    pub maximum_absolute_lateral_offset_m: f64,
    pub mean_absolute_speed_error_mps: f64,
    pub mean_control_effort: f64,
    pub maximum_absolute_yaw_rate_rps: f64,
    pub projected_road_progress_m: f64,
    pub final_speed_mps: f64,
    pub final_lateral_offset_m: f64,
    pub final_heading_error_rad: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VehiclePairedRolloutResult {
    pub schema_version: String,
    pub profile: VehiclePairedRolloutConfig,
    pub advanced_metrics: VehicleTrajectoryMetrics,
    pub baseline_metrics: VehicleTrajectoryMetrics,
    /// Signed advanced-minus-baseline deltas. Negative is preferable for the
    /// error/effort metrics; positive is preferable for projected progress.
    pub advanced_minus_baseline: VehicleTrajectoryMetrics,
    pub frames: Vec<VehiclePairedRolloutFrame>,
    /// BLAKE3 over the complete result with this field empty; not a signature.
    pub run_output_digest: String,
    /// Explicitly records the simulator's actor-model limitation.
    pub actor_capabilities: Vec<String>,
}

#[derive(Default)]
struct MetricsAccumulator {
    steps: usize,
    abs_lateral_sum: f64,
    squared_lateral_sum: f64,
    max_abs_lateral: f64,
    abs_speed_error_sum: f64,
    control_effort_sum: f64,
    max_abs_yaw_rate: f64,
    initial_progress_m: Option<f64>,
}

impl MetricsAccumulator {
    fn observe(
        &mut self,
        plant_state: &VehicleState,
        road: &Road,
        target_speed_mps: f64,
        command: &VehicleCommand,
    ) {
        let projection = road.project(plant_state.position_x, plant_state.position_y);
        let lateral = projection.lateral_offset.abs();
        self.steps += 1;
        self.abs_lateral_sum += lateral;
        self.squared_lateral_sum += lateral * lateral;
        self.max_abs_lateral = self.max_abs_lateral.max(lateral);
        self.abs_speed_error_sum += (plant_state.speed - target_speed_mps).abs();
        self.control_effort_sum += command.control_effort() as f64;
        self.max_abs_yaw_rate = self.max_abs_yaw_rate.max(plant_state.yaw_rate.abs());
        self.initial_progress_m.get_or_insert(projection.s);
    }

    fn finish(
        &self,
        final_state: &VehicleState,
        road: &Road,
    ) -> VehicleTrajectoryMetrics {
        let n = self.steps.max(1) as f64;
        let final_projection = road.project(final_state.position_x, final_state.position_y);
        let initial_progress = self.initial_progress_m.unwrap_or(final_projection.s);
        VehicleTrajectoryMetrics {
            steps: self.steps,
            mean_absolute_lateral_offset_m: self.abs_lateral_sum / n,
            rms_lateral_offset_m: (self.squared_lateral_sum / n).sqrt(),
            maximum_absolute_lateral_offset_m: self.max_abs_lateral,
            mean_absolute_speed_error_mps: self.abs_speed_error_sum / n,
            mean_control_effort: self.control_effort_sum / n,
            maximum_absolute_yaw_rate_rps: self.max_abs_yaw_rate,
            projected_road_progress_m: final_projection.s - initial_progress,
            final_speed_mps: final_state.speed,
            final_lateral_offset_m: final_projection.lateral_offset,
            final_heading_error_rad: wrap_angle(final_state.heading - final_projection.road_heading),
        }
    }
}

/// Run a matched HDC-LTC-vs-PD closed-loop lane-keeping experiment.
///
/// - `initial_simulator` is cloned once for each controller, preserving its
///   exact state, chassis parameters and friction configuration.
/// - The same perturbation schedule is applied independently to each plant.
/// - The advanced controller and encoder are reset before the run and are not
///   trained or modified during the episode.
/// - The PD baseline receives its own plant's current observation.
/// - The two plants evolve independently after the first action.
///
/// This runner does not model traffic actors or collision outcomes. It is not
/// an intersection test and does not establish physical dynamics fidelity.
pub fn run_paired_lane_keep_rollout(
    candidate_id: impl Into<String>,
    road_profile_id: impl Into<String>,
    advanced_controller_digest: impl Into<String>,
    initial_simulator: &BicycleModelSimulator,
    road: &Road,
    perturbations: &PerturbationSchedule,
    target_speed_mps: f64,
    timestep_s: f64,
    steps: usize,
    advanced_controller: &mut VehicleController,
    encoder: &mut VehicleHdcEncoder,
) -> Result<VehiclePairedRolloutResult, VehiclePairedRolloutError> {
    run_paired_lane_keep_rollout_with_baseline(
        candidate_id,
        road_profile_id,
        advanced_controller_digest,
        initial_simulator,
        road,
        perturbations,
        target_speed_mps,
        timestep_s,
        steps,
        &VehicleClassicalBaseline::PdRoadTeacher,
        advanced_controller,
        encoder,
    )
}

/// Run the same paired experiment against a selectable, versioned classical baseline.
pub fn run_paired_lane_keep_rollout_with_baseline(
    candidate_id: impl Into<String>,
    road_profile_id: impl Into<String>,
    advanced_controller_digest: impl Into<String>,
    initial_simulator: &BicycleModelSimulator,
    road: &Road,
    perturbations: &PerturbationSchedule,
    target_speed_mps: f64,
    timestep_s: f64,
    steps: usize,
    baseline_policy: &VehicleClassicalBaseline,
    advanced_controller: &mut VehicleController,
    encoder: &mut VehicleHdcEncoder,
) -> Result<VehiclePairedRolloutResult, VehiclePairedRolloutError> {
    let candidate_id = candidate_id.into();
    let road_profile_id = road_profile_id.into();
    let advanced_controller_digest = advanced_controller_digest.into();
    let baseline_controller_digest = baseline_policy
        .controller_digest()
        .ok_or(VehiclePairedRolloutError::InvalidBaselinePolicy)?;

    if candidate_id.trim().is_empty() {
        return Err(VehiclePairedRolloutError::EmptyCandidateId);
    }
    if road_profile_id.trim().is_empty() {
        return Err(VehiclePairedRolloutError::EmptyRoadProfileId);
    }
    let road_geometry_digest = road
        .profile_digest()
        .ok_or(VehiclePairedRolloutError::InvalidRoadProfile)?;
    if !initial_simulator.dynamics_profile_is_valid() {
        return Err(VehiclePairedRolloutError::InvalidRolloutProfile);
    }
    if !digest_is_valid(&advanced_controller_digest) {
        return Err(VehiclePairedRolloutError::InvalidAdvancedControllerDigest);
    }
    if advanced_controller_digest.eq_ignore_ascii_case(&baseline_controller_digest) {
        return Err(VehiclePairedRolloutError::ControllerIdentityAliased);
    }
    if !target_speed_mps.is_finite() || !(0.0..=80.0).contains(&target_speed_mps) {
        return Err(VehiclePairedRolloutError::InvalidTargetSpeed);
    }
    if !timestep_s.is_finite() || timestep_s <= 0.0 || timestep_s > 0.1 {
        return Err(VehiclePairedRolloutError::InvalidTimestep);
    }
    if steps == 0 || steps > MAX_PAIRED_ROLLOUT_STEPS {
        return Err(VehiclePairedRolloutError::InvalidStepBudget);
    }
    if !state_is_acceptable(initial_simulator.state()) {
        return Err(VehiclePairedRolloutError::InvalidInitialState);
    }

    if !perturbation_schedule_is_valid(perturbations) {
        return Err(VehiclePairedRolloutError::InvalidRolloutProfile);
    }
    let initial_state_bytes = serde_json::to_vec(initial_simulator.state())
        .map_err(|_| VehiclePairedRolloutError::InvalidInitialState)?;
    let initial_state_digest = blake3::hash(&initial_state_bytes).to_hex().to_string();
    let perturbation_bytes = serde_json::to_vec(perturbations)
        .map_err(|_| VehiclePairedRolloutError::InvalidRolloutProfile)?;
    let perturbation_schedule_digest = blake3::hash(&perturbation_bytes).to_hex().to_string();

    let mut profile = VehiclePairedRolloutConfig {
        candidate_id,
        road_profile_id,
        road_geometry_digest,
        advanced_controller_digest,
        baseline_policy_id: baseline_policy.policy_id().into(),
        baseline_controller_digest,
        simulator_profile_digest: initial_simulator.dynamics_profile_digest(),
        initial_state_digest,
        perturbation_schedule_digest,
        target_speed_mps,
        timestep_s,
        steps,
        experiment_profile_digest: String::new(),
    };
    let profile_bytes = serde_json::to_vec(&profile)
        .map_err(|_| VehiclePairedRolloutError::InvalidRolloutProfile)?;
    profile.experiment_profile_digest = blake3::hash(&profile_bytes).to_hex().to_string();
    if !profile.identity_is_valid() {
        return Err(VehiclePairedRolloutError::InvalidRolloutProfile);
    }

    let mut advanced_simulator = initial_simulator.clone();
    let mut baseline_simulator = initial_simulator.clone();
    advanced_controller.reset();
    encoder.reset();

    let mut advanced_accumulator = MetricsAccumulator::default();
    let mut baseline_accumulator = MetricsAccumulator::default();
    let mut frames = Vec::with_capacity(steps);

    for step_index in 0..steps {
        perturbations.apply(step_index, &mut advanced_simulator);
        perturbations.apply(step_index, &mut baseline_simulator);

        let advanced_plant_before = advanced_simulator.state().clone();
        let baseline_plant_before = baseline_simulator.state().clone();
        let mut advanced_observation = advanced_plant_before.clone();
        let mut baseline_observation = baseline_plant_before.clone();

        road.apply(&mut advanced_observation);
        road.apply(&mut baseline_observation);
        perturbations.apply_to_state(step_index, &mut advanced_observation);
        perturbations.apply_to_state(step_index, &mut baseline_observation);

        let advanced_projection = road.project(
            advanced_plant_before.position_x,
            advanced_plant_before.position_y,
        );
        let baseline_projection = road.project(
            baseline_plant_before.position_x,
            baseline_plant_before.position_y,
        );

        let sensor_hv = encoder.encode(&advanced_observation);
        let advanced_command = advanced_controller.forward(&sensor_hv, timestep_s as f32);
        if !command_is_valid(&advanced_command) {
            return Err(VehiclePairedRolloutError::AdvancedCommandInvalid);
        }
        let baseline_command = baseline_policy
            .command(
                &baseline_observation,
                road,
                target_speed_mps,
                initial_simulator.params(),
            )
            .ok_or(VehiclePairedRolloutError::BaselineCommandInvalid)?;
        if !command_is_valid(&baseline_command) {
            return Err(VehiclePairedRolloutError::BaselineCommandInvalid);
        }

        advanced_accumulator.observe(
            &advanced_plant_before,
            road,
            target_speed_mps,
            &advanced_command,
        );
        baseline_accumulator.observe(
            &baseline_plant_before,
            road,
            target_speed_mps,
            &baseline_command,
        );

        advanced_simulator.step(&advanced_command, timestep_s);
        baseline_simulator.step(&baseline_command, timestep_s);
        let expected_advanced_time = advanced_plant_before.timestamp + timestep_s;
        let expected_baseline_time = baseline_plant_before.timestamp + timestep_s;
        let advanced_time = advanced_simulator.state().timestamp;
        let baseline_time = baseline_simulator.state().timestamp;
        let advanced_time_tolerance = 1e-9 * expected_advanced_time.abs().max(1.0);
        let baseline_time_tolerance = 1e-9 * expected_baseline_time.abs().max(1.0);
        if !state_is_acceptable(advanced_simulator.state())
            || !state_is_acceptable(baseline_simulator.state())
            || (advanced_time - expected_advanced_time).abs() > advanced_time_tolerance
            || (baseline_time - expected_baseline_time).abs() > baseline_time_tolerance
        {
            return Err(VehiclePairedRolloutError::InvalidStateDuringRollout);
        }

        frames.push(VehiclePairedRolloutFrame {
            step_index,
            elapsed_s: step_index as f64 * timestep_s,
            advanced_plant_state_before: advanced_plant_before,
            baseline_plant_state_before: baseline_plant_before,
            advanced_observation_state: advanced_observation,
            baseline_observation_state: baseline_observation,
            advanced_command,
            baseline_command,
            advanced_plant_state_after: advanced_simulator.state().clone(),
            baseline_plant_state_after: baseline_simulator.state().clone(),
            advanced_progress_m: advanced_projection.s,
            baseline_progress_m: baseline_projection.s,
        });
    }

    let advanced_metrics = advanced_accumulator.finish(advanced_simulator.state(), road);
    let baseline_metrics = baseline_accumulator.finish(baseline_simulator.state(), road);
    let advanced_minus_baseline = subtract_metrics(&advanced_metrics, &baseline_metrics);

    let mut result = VehiclePairedRolloutResult {
        schema_version: VEHICLE_PAIRED_ROLLOUT_SCHEMA.into(),
        profile,
        advanced_metrics,
        baseline_metrics,
        advanced_minus_baseline,
        frames,
        run_output_digest: String::new(),
        actor_capabilities: vec![
            "no-pedestrian-actors".into(),
            "no-cyclist-actors".into(),
            "no-dynamic-traffic-actors".into(),
            "no-intersection-collision-observable".into(),
        ],
    };
    let output_bytes = serde_json::to_vec(&result)
        .map_err(|_| VehiclePairedRolloutError::InvalidRolloutProfile)?;
    result.run_output_digest = blake3::hash(&output_bytes).to_hex().to_string();
    Ok(result)
}

impl VehiclePairedRolloutResult {
    /// Verify the retained output trace and summary have not changed.
    pub fn output_identity_is_valid(&self) -> bool {
        let mut result = self.clone();
        let expected = result.run_output_digest.clone();
        result.run_output_digest.clear();
        if !result.profile.identity_is_valid() {
            return false;
        }
        serde_json::to_vec(&result)
            .ok()
            .map(|bytes| blake3::hash(&bytes).to_hex().to_string() == expected)
            .unwrap_or(false)
    }
}

fn subtract_metrics(
    advanced: &VehicleTrajectoryMetrics,
    baseline: &VehicleTrajectoryMetrics,
) -> VehicleTrajectoryMetrics {
    VehicleTrajectoryMetrics {
        steps: advanced.steps,
        mean_absolute_lateral_offset_m: advanced.mean_absolute_lateral_offset_m
            - baseline.mean_absolute_lateral_offset_m,
        rms_lateral_offset_m: advanced.rms_lateral_offset_m - baseline.rms_lateral_offset_m,
        maximum_absolute_lateral_offset_m: advanced.maximum_absolute_lateral_offset_m
            - baseline.maximum_absolute_lateral_offset_m,
        mean_absolute_speed_error_mps: advanced.mean_absolute_speed_error_mps
            - baseline.mean_absolute_speed_error_mps,
        mean_control_effort: advanced.mean_control_effort - baseline.mean_control_effort,
        maximum_absolute_yaw_rate_rps: advanced.maximum_absolute_yaw_rate_rps
            - baseline.maximum_absolute_yaw_rate_rps,
        projected_road_progress_m: advanced.projected_road_progress_m
            - baseline.projected_road_progress_m,
        final_speed_mps: advanced.final_speed_mps - baseline.final_speed_mps,
        final_lateral_offset_m: advanced.final_lateral_offset_m - baseline.final_lateral_offset_m,
        final_heading_error_rad: advanced.final_heading_error_rad
            - baseline.final_heading_error_rad,
    }
}

fn digest_is_valid(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn state_is_acceptable(state: &VehicleState) -> bool {
    state.is_bounded() && state_is_finite(state)
}

fn perturbation_schedule_is_valid(schedule: &PerturbationSchedule) -> bool {
    for (step, event) in &schedule.events {
        if schedule.clear_step.is_some_and(|clear| *step >= clear) {
            return false;
        }
        let valid = match event {
            VehiclePerturbation::BlackIce { friction_scale } => {
                friction_scale.is_finite() && (0.0..=2.0).contains(friction_scale)
            }
            VehiclePerturbation::SensorBlindness { affected_channels } => {
                *affected_channels <= 14
            }
            VehiclePerturbation::Crosswind { force_n } => force_n.is_finite(),
            VehiclePerturbation::TireBlowout {
                tire,
                grip_reduction,
            } => {
                matches!(
                    tire.as_str(),
                    "front_left" | "front_right" | "rear_left" | "rear_right"
                ) && grip_reduction.is_finite()
                    && (0.0..=1.0).contains(grip_reduction)
            }
            VehiclePerturbation::MeshSpoof { num_spoofed_peers } => {
                *num_spoofed_peers <= 1024
            }
            VehiclePerturbation::RoadImpulse { decel_impulse } => {
                decel_impulse.is_finite() && *decel_impulse >= 0.0
            }
        };
        if !valid {
            return false;
        }
    }
    true
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

fn wrap_angle(mut angle: f64) -> f64 {
    while angle > std::f64::consts::PI {
        angle -= 2.0 * std::f64::consts::PI;
    }
    while angle < -std::f64::consts::PI {
        angle += 2.0 * std::f64::consts::PI;
    }
    angle
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{VehicleConfig, VehiclePerturbation};
    use symthaea_core::genesis::GenesisSeed;

    fn setup() -> (BicycleModelSimulator, Road, PerturbationSchedule, VehicleController, VehicleHdcEncoder) {
        let initial_simulator = BicycleModelSimulator::at_speed(10.0);
        let road = Road::straight();
        let perturbations = PerturbationSchedule::none();
        let genesis = GenesisSeed::from_phrase("paired-lane-keep-test");
        let vehicle_config = VehicleConfig::default();
        let controller = VehicleController::new(&genesis, &vehicle_config);
        let encoder = VehicleHdcEncoder::new(&genesis, 32);
        (initial_simulator, road, perturbations, controller, encoder)
    }

    #[test]
    fn paired_rollout_clones_one_initial_state_and_retains_both_trajectories() {
        let (sim, road, perturbations, mut controller, mut encoder) = setup();
        let result = run_paired_lane_keep_rollout(
            "lane-keep-fixture-001",
            "straight-road-v1",
            "a".repeat(64),
            &sim,
            &road,
            &perturbations,
            10.0,
            0.005,
            8,
            &mut controller,
            &mut encoder,
        )
        .unwrap();

        assert!(result.profile.identity_is_valid());
        assert!(result.output_identity_is_valid());
        assert_eq!(result.profile.road_geometry_digest, road.profile_digest().unwrap());
        assert_eq!(result.frames.len(), 8);
        assert_eq!(result.advanced_metrics.steps, 8);
        assert_eq!(result.baseline_metrics.steps, 8);
        assert_eq!(
            result.frames[0].advanced_plant_state_before.speed,
            result.frames[0].baseline_plant_state_before.speed
        );
        assert_eq!(
            result.frames[0].advanced_plant_state_before.position_x,
            result.frames[0].baseline_plant_state_before.position_x
        );
        assert_eq!(
            result.actor_capabilities.iter().map(String::as_str).collect::<Vec<_>>(),
            vec![
                "no-pedestrian-actors",
                "no-cyclist-actors",
                "no-dynamic-traffic-actors",
                "no-intersection-collision-observable"
            ]
        );
    }

    #[test]
    fn identical_frozen_runs_produce_identical_output_receipts() {
        let (sim, road, perturbations, mut controller, mut encoder) = setup();
        let first = run_paired_lane_keep_rollout(
            "lane-keep-replay-001",
            "straight-road-v1",
            "9".repeat(64),
            &sim,
            &road,
            &perturbations,
            10.0,
            0.005,
            16,
            &mut controller,
            &mut encoder,
        )
        .unwrap();
        let second = run_paired_lane_keep_rollout(
            "lane-keep-replay-001",
            "straight-road-v1",
            "9".repeat(64),
            &sim,
            &road,
            &perturbations,
            10.0,
            0.005,
            16,
            &mut controller,
            &mut encoder,
        )
        .unwrap();
        assert_eq!(first.profile.experiment_profile_digest, second.profile.experiment_profile_digest);
        assert_eq!(first.run_output_digest, second.run_output_digest);
    }

    #[test]
    fn paired_rollout_applies_identical_crosswind_to_both_plants() {
        let (sim, road, _, mut controller, mut encoder) = setup();
        let (weights, _) = controller.output_projection();
        controller.set_output_projection(&vec![0.0; weights.len()], &[0.0, -20.0, -20.0]);
        let perturbations = PerturbationSchedule {
            events: vec![(0, VehiclePerturbation::Crosswind { force_n: 100.0 })],
            clear_step: Some(4),
        };
        let result = run_paired_lane_keep_rollout(
            "lane-keep-crosswind-001",
            "straight-road-v1",
            "b".repeat(64),
            &sim,
            &road,
            &perturbations,
            10.0,
            0.005,
            4,
            &mut controller,
            &mut encoder,
        )
        .unwrap();
        assert_eq!(
            result.frames[0].advanced_plant_state_after.lateral_velocity,
            result.frames[0].baseline_plant_state_after.lateral_velocity
        );
        assert_eq!(
            result.frames[0].advanced_plant_state_after.lateral_accel,
            result.frames[0].baseline_plant_state_after.lateral_accel
        );
        assert_eq!(
            result.profile.perturbation_schedule_digest,
            blake3::hash(&serde_json::to_vec(&perturbations).unwrap())
                .to_hex()
                .to_string()
        );
    }

    #[test]
    fn paired_rollout_binds_stanley_baseline_policy_and_digest() {
        let (sim, road, perturbations, mut controller, mut encoder) = setup();
        let baseline = VehicleClassicalBaseline::StanleyRoad(
            crate::classical_baselines::StanleyRoadConfig::default(),
        );
        let result = run_paired_lane_keep_rollout_with_baseline(
            "lane-keep-stanley-001",
            "straight-road-v1",
            "8".repeat(64),
            &sim,
            &road,
            &perturbations,
            10.0,
            0.005,
            4,
            &baseline,
            &mut controller,
            &mut encoder,
        )
        .unwrap();
        assert_eq!(result.profile.baseline_policy_id, baseline.policy_id());
        assert_eq!(
            result.profile.baseline_controller_digest,
            baseline.controller_digest().unwrap()
        );
        assert_eq!(result.frames.len(), 4);
        assert!(result.output_identity_is_valid());
    }

    #[test]
    fn paired_rollout_binds_pure_pursuit_baseline_policy_and_digest() {
        let (sim, road, perturbations, mut controller, mut encoder) = setup();
        let baseline = VehicleClassicalBaseline::PurePursuitRoad(
            crate::classical_baselines::PurePursuitRoadConfig::default(),
        );
        let result = run_paired_lane_keep_rollout_with_baseline(
            "lane-keep-pure-pursuit-001",
            "straight-road-v1",
            "7".repeat(64),
            &sim,
            &road,
            &perturbations,
            10.0,
            0.005,
            4,
            &baseline,
            &mut controller,
            &mut encoder,
        )
        .unwrap();
        assert_eq!(result.profile.baseline_policy_id, baseline.policy_id());
        assert_eq!(
            result.profile.baseline_controller_digest,
            baseline.controller_digest().unwrap()
        );
        assert_eq!(result.frames.len(), 4);
        assert!(result.output_identity_is_valid());
    }

    #[test]
    fn output_digest_and_profile_digest_detect_tampering() {
        let (sim, road, perturbations, mut controller, mut encoder) = setup();
        let mut result = run_paired_lane_keep_rollout(
            "lane-keep-profile-digest",
            "straight-road-v1",
            "e".repeat(64),
            &sim,
            &road,
            &perturbations,
            10.0,
            0.005,
            2,
            &mut controller,
            &mut encoder,
        ).unwrap();
        assert!(result.profile.identity_is_valid());
        result.profile.road_profile_id.push_str("-tampered");
        assert!(!result.output_identity_is_valid());
    }

    #[test]
    fn output_digest_detects_frame_mutation() {
        let (sim, road, perturbations, mut controller, mut encoder) = setup();
        let mut result = run_paired_lane_keep_rollout(
            "lane-keep-fixture-digest",
            "straight-road-v1",
            "c".repeat(64),
            &sim,
            &road,
            &perturbations,
            10.0,
            0.005,
            2,
            &mut controller,
            &mut encoder,
        )
        .unwrap();
        result.frames[0].advanced_command.throttle = 0.99;
        assert!(!result.output_identity_is_valid());
    }

    #[test]
    fn invalid_or_nonfinite_chassis_parameters_fail_closed() {
        let mut params = crate::types::ChassisParams::default();
        params.mass = f64::NAN;
        let sim = BicycleModelSimulator::with_params(params);
        let ( _, road, perturbations, mut controller, mut encoder) = setup();
        assert_eq!(
            run_paired_lane_keep_rollout(
                "candidate",
                "road",
                "d".repeat(64),
                &sim,
                &road,
                &perturbations,
                10.0,
                0.005,
                2,
                &mut controller,
                &mut encoder,
            ).err(),
            Some(VehiclePairedRolloutError::InvalidRolloutProfile)
        );
    }

    #[test]
    fn invalid_perturbation_parameters_are_rejected_before_stepping() {
        let (sim, road, _, mut controller, mut encoder) = setup();
        let invalid = PerturbationSchedule {
            events: vec![(0, VehiclePerturbation::TireBlowout {
                tire: "unknown-wheel".into(),
                grip_reduction: 0.5,
            })],
            clear_step: Some(2),
        };
        assert_eq!(
            run_paired_lane_keep_rollout(
                "candidate",
                "road",
                "f".repeat(64),
                &sim,
                &road,
                &invalid,
                10.0,
                0.005,
                2,
                &mut controller,
                &mut encoder,
            ).err(),
            Some(VehiclePairedRolloutError::InvalidRolloutProfile)
        );
    }

    #[test]
    fn rejects_invalid_profiles_before_stepping() {
        let (sim, road, perturbations, mut controller, mut encoder) = setup();
        assert_eq!(
            run_paired_lane_keep_rollout(
                "candidate",
                "road",
                "bad-digest",
                &sim,
                &road,
                &perturbations,
                10.0,
                0.005,
                8,
                &mut controller,
                &mut encoder,
            ).err(),
            Some(VehiclePairedRolloutError::InvalidAdvancedControllerDigest)
        );
    }
}
