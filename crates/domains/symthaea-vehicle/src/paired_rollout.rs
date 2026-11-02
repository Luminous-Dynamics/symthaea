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

use symthaea_core::genesis::GenesisSeed;

use crate::controller::VehicleController;
use crate::encoder::VehicleHdcEncoder;
use crate::perturbations::PerturbationSchedule;
use crate::road::Road;
use crate::simulator::{BicycleModelSimulator, VehiclePhysicsSimulator};
use crate::types::{
    VehicleCommand, VehicleConfig, VehicleState, pd_cruise_baseline_with_road,
};
use crate::shadow_evaluation::pd_road_baseline_digest;

pub const VEHICLE_PAIRED_ROLLOUT_SCHEMA: &str = "symthaea.vehicle.paired-rollout.v0.1";
pub const MAX_PAIRED_ROLLOUT_STEPS: usize = 100_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VehiclePairedRolloutError {
    EmptyCandidateId,
    EmptyRoadProfileId,
    InvalidAdvancedControllerDigest,
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
    final_progress_m: f64,
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
        self.final_progress_m = projection.s;
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
    let candidate_id = candidate_id.into();
    let road_profile_id = road_profile_id.into();
    let advanced_controller_digest = advanced_controller_digest.into();
    let baseline_controller_digest = pd_road_baseline_digest();

    if candidate_id.trim().is_empty() {
        return Err(VehiclePairedRolloutError::EmptyCandidateId);
    }
    if road_profile_id.trim().is_empty() {
        return Err(VehiclePairedRolloutError::EmptyRoadProfileId);
    }
    if !digest_is_valid(&advanced_controller_digest) {
        return Err(VehiclePairedRolloutError::InvalidAdvancedControllerDigest);
    }
    if advanced_controller_digest.eq_ignore_ascii_case(&baseline_controller_digest) {
        return Err(VehiclePairedRolloutError::ControllerIdentityAliased);
    }
    if !target_speed_mps.is_finite() || target_speed_mps < 0.0 {
        return Err(VehiclePairedRolloutError::InvalidTargetSpeed);
    }
    if !timestep_s.is_finite() || timestep_s <= 0.0 || timestep_s > 0.1 {
        return Err(VehiclePairedRolloutError::InvalidTimestep);
    }
    if steps == 0 || steps > MAX_PAIRED_ROLLOUT_STEPS {
        return Err(VehiclePairedRolloutError::InvalidStepBudget);
    }
    if !state_is_finite(initial_simulator.state()) {
        return Err(VehiclePairedRolloutError::InvalidInitialState);
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
        advanced_controller_digest,
        baseline_policy_id: crate::shadow_evaluation::PD_ROAD_BASELINE_POLICY_ID.into(),
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
        let baseline_command = pd_cruise_baseline_with_road(
            &baseline_observation,
            target_speed_mps,
            baseline_projection.road_heading,
        );
        if !command_is_valid(&baseline_command) {
            return Err(VehiclePairedRolloutError::BaselineCommandInvalid);
        }

        advanced_accumulator.observe(
            &advanced_observation_for_metrics(&advanced_plant_before, road),
            road,
            target_speed_mps,
            &advanced_command,
        );
        baseline_accumulator.observe(
            &advanced_observation_for_metrics(&baseline_plant_before, road),
            road,
            target_speed_mps,
            &baseline_command,
        );

        advanced_simulator.step(&advanced_command, timestep_s);
        baseline_simulator.step(&baseline_command, timestep_s);

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

    Ok(VehiclePairedRolloutResult {
        schema_version: VEHICLE_PAIRED_ROLLOUT_SCHEMA.into(),
        profile,
        advanced_metrics,
        baseline_metrics,
        advanced_minus_baseline,
        frames,
        actor_capabilities: vec![
            "no-pedestrian-actors".into(),
            "no-cyclist-actors".into(),
            "no-dynamic-traffic-actors".into(),
            "no-intersection-collision-observable".into(),
        ],
    })
}

fn advanced_observation_for_metrics(state: &VehicleState, road: &Road) -> VehicleState {
    let mut metric_state = state.clone();
    road.apply(&mut metric_state);
    metric_state
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
            result.actor_capabilities,
            vec![
                "no-pedestrian-actors",
                "no-cyclist-actors",
                "no-dynamic-traffic-actors",
                "no-intersection-collision-observable"
            ]
        );
    }

    #[test]
    fn paired_rollout_uses_the_same_perturbation_schedule() {
        let (sim, road, _, mut controller, mut encoder) = setup();
        let perturbations = PerturbationSchedule::crosswind();
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
        assert_eq!(result.frames[0].advanced_plant_state_before.position_y,
                   result.frames[0].baseline_plant_state_before.position_y);
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
