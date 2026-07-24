// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Constrained morphology-aware whole-body command allocation.
//!
//! This module is deliberately honest about its scope: it is not a complete
//! articulated rigid-body inverse-dynamics solver. It is a deterministic,
//! projected dynamics allocator that distributes torso/COM objectives across
//! the available joints while enforcing one-step position, velocity, command,
//! and support-phase constraints. It provides a stable contract that a future
//! sparse QP inverse-dynamics backend can implement without changing callers.

use serde::{Deserialize, Serialize};

use crate::contact::{BipedSupport, ContactFrame};
use crate::footstep::{FootSide, FootstepPlan};
use crate::morphology::HumanoidMorphology;
use crate::types::{HumanoidCommand, HumanoidState};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct WholeBodyObjective {
    pub desired_sagittal_com_accel_mps2: f64,
    pub desired_lateral_com_accel_mps2: f64,
    pub desired_torso_pitch: f64,
    pub desired_torso_roll: f64,
    pub desired_support_ratio: f64,
    pub planned_step: Option<FootstepPlan>,
}

impl Default for WholeBodyObjective {
    fn default() -> Self {
        Self {
            desired_sagittal_com_accel_mps2: 0.0,
            desired_lateral_com_accel_mps2: 0.0,
            desired_torso_pitch: 0.0,
            desired_torso_roll: 0.0,
            desired_support_ratio: 1.0,
            planned_step: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WholeBodyConstraintConfig {
    pub prediction_horizon_s: f64,
    pub max_joint_speed_rad_s: f64,
    pub max_command_delta: f32,
    pub single_support_authority: f32,
    pub flight_authority: f32,
    pub posture_gain: f64,
    pub com_accel_gain: f64,
    pub swing_foot_gain: f64,
    pub projection_iterations: usize,
    pub feasibility_tolerance: f64,
}

impl Default for WholeBodyConstraintConfig {
    fn default() -> Self {
        Self {
            prediction_horizon_s: 0.08,
            max_joint_speed_rad_s: 12.0,
            max_command_delta: 0.32,
            single_support_authority: 0.88,
            flight_authority: 0.55,
            posture_gain: 0.55,
            com_accel_gain: 0.12,
            swing_foot_gain: 0.28,
            projection_iterations: 3,
            feasibility_tolerance: 1e-5,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct WholeBodyControlReport {
    pub active_constraints: usize,
    pub maximum_joint_utilization: f64,
    pub mean_command_change: f32,
    pub objective_residual: f64,
    pub support_authority: f32,
    pub feasible: bool,
}

pub struct ConstrainedWholeBodyController {
    morphology: HumanoidMorphology,
    config: WholeBodyConstraintConfig,
    limits: Vec<[f64; 2]>,
    inertias: Vec<f64>,
    damping: Vec<f64>,
    torque_scales: Vec<f64>,
    names: Vec<String>,
}

impl ConstrainedWholeBodyController {
    pub fn new(morphology: HumanoidMorphology) -> Self {
        Self::with_config(morphology, WholeBodyConstraintConfig::default())
    }

    pub fn with_config(morphology: HumanoidMorphology, config: WholeBodyConstraintConfig) -> Self {
        Self {
            morphology,
            config,
            limits: morphology.joint_limits(),
            inertias: morphology.joint_inertias(),
            damping: morphology.joint_damping(),
            torque_scales: morphology.joint_torque_scales(),
            names: morphology.joint_names(),
        }
    }

    pub fn allocate(
        &self,
        state: &HumanoidState,
        contacts: &ContactFrame,
        nominal: &HumanoidCommand,
        objective: WholeBodyObjective,
    ) -> (HumanoidCommand, WholeBodyControlReport) {
        let n = self.morphology.num_actuators();
        if state.joint_angles.len() != n
            || state.joint_velocities.len() != n
            || nominal.num_actuators() != n
        {
            return (
                HumanoidCommand::zero_for(n),
                WholeBodyControlReport {
                    active_constraints: n,
                    maximum_joint_utilization: f64::INFINITY,
                    mean_command_change: 0.0,
                    objective_residual: f64::INFINITY,
                    support_authority: 0.0,
                    feasible: false,
                },
            );
        }

        let support_authority = match contacts.support() {
            BipedSupport::Double => 1.0,
            BipedSupport::Right | BipedSupport::Left => self.config.single_support_authority,
            BipedSupport::Flight => self.config.flight_authority,
        }
        .clamp(0.0, 1.0);
        let mut values = nominal.torques.clone();
        let before = values.clone();
        self.apply_objectives(&mut values, state, objective, support_authority);

        let mut active_constraints = 0usize;
        let mut maximum_joint_utilization = 0.0f64;
        for _ in 0..self.config.projection_iterations.max(1) {
            for i in 0..n {
                let requested = values[i];
                let projected =
                    self.project_joint(i, state, requested, before[i], support_authority);
                if (projected - requested).abs() > self.config.feasibility_tolerance as f32 {
                    active_constraints += 1;
                }
                values[i] = projected;
                maximum_joint_utilization = maximum_joint_utilization
                    .max(joint_utilization(state.joint_angles[i], self.limits[i]));
            }
        }

        let mean_command_change = values
            .iter()
            .zip(before.iter())
            .map(|(after, before)| (after - before).abs())
            .sum::<f32>()
            / n.max(1) as f32;
        let objective_residual = self.objective_residual(state, &values, objective);
        let feasible = values.iter().all(|value| value.is_finite())
            && maximum_joint_utilization <= 1.0 + self.config.feasibility_tolerance;

        (
            HumanoidCommand { torques: values },
            WholeBodyControlReport {
                active_constraints,
                maximum_joint_utilization,
                mean_command_change,
                objective_residual,
                support_authority,
                feasible,
            },
        )
    }

    fn apply_objectives(
        &self,
        values: &mut [f32],
        state: &HumanoidState,
        objective: WholeBodyObjective,
        support_authority: f32,
    ) {
        let pitch_error = objective.desired_torso_pitch - state.torso_vertical[1];
        let roll_error = objective.desired_torso_roll - state.torso_vertical[0];
        let sagittal = (self.config.posture_gain * pitch_error
            + self.config.com_accel_gain * objective.desired_sagittal_com_accel_mps2)
            as f32
            * support_authority;
        let lateral = (self.config.posture_gain * roll_error
            + self.config.com_accel_gain * objective.desired_lateral_com_accel_mps2)
            as f32
            * support_authority;
        let support_gain = objective.desired_support_ratio.clamp(0.25, 1.25) as f32;

        for (index, name) in self.names.iter().enumerate() {
            let correction = match name.as_str() {
                "abdomen_y" => -0.55 * sagittal,
                "abdomen_x" => -0.55 * lateral,
                "right_hip_y" | "left_hip_y" => 0.42 * sagittal,
                "right_ankle_y" | "left_ankle_y" => -0.62 * sagittal,
                "right_hip_x" | "left_hip_x" => 0.38 * lateral,
                "right_ankle_x" | "left_ankle_x" => -0.58 * lateral,
                "right_knee" | "left_knee" => -0.06 * (support_gain - 1.0),
                _ => 0.0,
            };
            values[index] += correction;
        }

        if let Some(plan) = objective.planned_step.filter(|plan| plan.feasible) {
            let forward_error = plan.target_world_m[0] - state.root_position[0];
            let lateral_error = plan.target_world_m[1] - state.root_position[1];
            let swing_scale = self.config.swing_foot_gain as f32 * plan.confidence as f32;
            let right = matches!(plan.swing_foot, FootSide::Right);
            for (index, name) in self.names.iter().enumerate() {
                let is_swing_joint = if right {
                    name.starts_with("right_")
                } else {
                    name.starts_with("left_")
                };
                if !is_swing_joint {
                    continue;
                }
                let correction = match name.as_str() {
                    "right_hip_y" | "left_hip_y" => forward_error as f32 * swing_scale,
                    "right_knee" | "left_knee" => -forward_error.abs() as f32 * 0.7 * swing_scale,
                    "right_ankle_y" | "left_ankle_y" => -forward_error as f32 * 0.45 * swing_scale,
                    "right_hip_x" | "left_hip_x" => lateral_error as f32 * swing_scale,
                    "right_ankle_x" | "left_ankle_x" => -lateral_error as f32 * 0.4 * swing_scale,
                    _ => 0.0,
                };
                values[index] += correction;
            }
        }
    }

    fn project_joint(
        &self,
        index: usize,
        state: &HumanoidState,
        requested: f32,
        nominal: f32,
        support_authority: f32,
    ) -> f32 {
        let horizon = self.config.prediction_horizon_s.max(1e-4);
        let q = state.joint_angles[index];
        let qd = state.joint_velocities[index];
        let inertia = self.inertias[index].max(1e-5);
        let damping = self.damping[index].max(0.0);
        let scale = self.torque_scales[index].max(1e-5);
        let [lower, upper] = self.limits[index];

        let command_from_position = |target_q: f64| {
            let required_accel = 2.0 * (target_q - q - qd * horizon) / (horizon * horizon);
            ((required_accel * inertia + damping * qd) / scale) as f32
        };
        let mut min_command = command_from_position(lower);
        let mut max_command = command_from_position(upper);
        if min_command > max_command {
            std::mem::swap(&mut min_command, &mut max_command);
        }

        let velocity_limit = self.config.max_joint_speed_rad_s.max(0.1);
        let min_velocity_command =
            (((-velocity_limit - qd) / horizon * inertia + damping * qd) / scale) as f32;
        let max_velocity_command =
            (((velocity_limit - qd) / horizon * inertia + damping * qd) / scale) as f32;
        min_command = min_command
            .max(min_velocity_command)
            .max(-support_authority);
        max_command = max_command.min(max_velocity_command).min(support_authority);

        let delta = self.config.max_command_delta.max(0.0);
        min_command = min_command.max(nominal - delta);
        max_command = max_command.min(nominal + delta);
        if min_command > max_command {
            return (0.5 * (min_command + max_command)).clamp(-1.0, 1.0);
        }
        requested.clamp(min_command.max(-1.0), max_command.min(1.0))
    }

    fn objective_residual(
        &self,
        state: &HumanoidState,
        values: &[f32],
        objective: WholeBodyObjective,
    ) -> f64 {
        let mut predicted_pitch = state.torso_vertical[1];
        let mut predicted_roll = state.torso_vertical[0];
        for (index, name) in self.names.iter().enumerate() {
            match name.as_str() {
                "abdomen_y" | "right_hip_y" | "left_hip_y" => {
                    predicted_pitch += values[index] as f64 * 0.02;
                }
                "abdomen_x" | "right_hip_x" | "left_hip_x" => {
                    predicted_roll += values[index] as f64 * 0.02;
                }
                _ => {}
            }
        }
        (objective.desired_torso_pitch - predicted_pitch).abs()
            + (objective.desired_torso_roll - predicted_roll).abs()
    }
}

fn joint_utilization(position: f64, limits: [f64; 2]) -> f64 {
    let center = 0.5 * (limits[0] + limits[1]);
    let half_range = 0.5 * (limits[1] - limits[0]).abs().max(1e-9);
    ((position - center).abs() / half_range).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocator_rejects_mismatched_dimensions() {
        let state = HumanoidState::standing();
        let nominal = HumanoidCommand::zero_for(20);
        let contacts = ContactFrame::estimated_from_state(&state, 0.05);
        let (_, report) = ConstrainedWholeBodyController::new(HumanoidMorphology::Dmc21).allocate(
            &state,
            &contacts,
            &nominal,
            WholeBodyObjective::default(),
        );
        assert!(!report.feasible);
    }

    #[test]
    fn allocator_limits_command_change() {
        let mut state = HumanoidState::standing();
        state.extremities[8] = 0.0;
        state.extremities[11] = 0.0;
        let nominal = HumanoidCommand::zero();
        let contacts = ContactFrame::estimated_from_state(&state, 0.05);
        let objective = WholeBodyObjective {
            desired_sagittal_com_accel_mps2: 10.0,
            desired_torso_pitch: 1.0,
            ..WholeBodyObjective::default()
        };
        let (command, report) = ConstrainedWholeBodyController::new(HumanoidMorphology::Dmc21)
            .allocate(&state, &contacts, &nominal, objective);
        assert!(report.feasible);
        assert!(
            command
                .torques
                .iter()
                .all(|value| value.abs() <= 0.32 + 1e-5)
        );
    }

    #[test]
    fn predicted_joint_limit_clips_outward_command() {
        let mut state = HumanoidState::standing();
        state.joint_angles[6] = -2.79;
        state.joint_velocities[6] = -4.0;
        state.extremities[8] = 0.0;
        state.extremities[11] = 0.0;
        let mut nominal = HumanoidCommand::zero();
        nominal.torques[6] = -1.0;
        let contacts = ContactFrame::estimated_from_state(&state, 0.05);
        let (command, report) = ConstrainedWholeBodyController::new(HumanoidMorphology::Dmc21)
            .allocate(&state, &contacts, &nominal, WholeBodyObjective::default());
        assert!(report.active_constraints > 0);
        assert!(command.torques[6] > -1.0);
    }
}
