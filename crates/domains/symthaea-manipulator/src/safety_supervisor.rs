// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Final-authority manipulator safety supervisor.
//!
//! Every nominal command must pass through this module immediately before the
//! physics or hardware backend. Cognition, training, moral gating, admittance,
//! and task controllers may request motion; only this supervisor may authorize
//! the applied actuator command.

use symthaea_core::embodiment::MotorSafetyLevel;

use crate::kinematics::ManipulatorKinematics;
use crate::types::{ManipulatorCommand, ManipulatorState, NUM_JOINTS};
use crate::workspace_safety::{ManipulatorSafetyLimits, WorkspaceBoundary};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyIntervention {
    None,
    SanitizedNonFiniteCommand,
    InvalidState,
    RedGravityHold,
    OrangeRetreat,
    WorkspaceRetreat,
    ForceLimitHold,
    JointSpeedBrake,
    TorqueLimited,
    TorqueSlewLimited,
}

#[derive(Debug, Clone, Copy)]
pub struct SafetyDecision {
    pub command: ManipulatorCommand,
    pub intervention: SafetyIntervention,
    pub workspace_clearance_m: f64,
    pub measured_force_n: f64,
}

#[derive(Debug, Clone)]
pub struct ManipulatorSafetySupervisor {
    workspace: WorkspaceBoundary,
    home_joint_angles: [f64; NUM_JOINTS],
    last_applied: ManipulatorCommand,
    max_normalized_torque_slew_per_s: f32,
    prediction_horizon_s: f64,
    red_latched: bool,
    last_intervention: SafetyIntervention,
}

impl ManipulatorSafetySupervisor {
    pub fn new(home_joint_angles: [f64; NUM_JOINTS]) -> Self {
        Self {
            workspace: WorkspaceBoundary::default(),
            home_joint_angles,
            last_applied: ManipulatorCommand::zero(),
            max_normalized_torque_slew_per_s: 8.0,
            prediction_horizon_s: 0.05,
            red_latched: false,
            last_intervention: SafetyIntervention::None,
        }
    }

    pub fn workspace(&self) -> &WorkspaceBoundary {
        &self.workspace
    }

    pub fn workspace_mut(&mut self) -> &mut WorkspaceBoundary {
        &mut self.workspace
    }

    pub fn last_applied_command(&self) -> ManipulatorCommand {
        self.last_applied
    }

    pub fn last_intervention(&self) -> SafetyIntervention {
        self.last_intervention
    }

    pub fn red_latched(&self) -> bool {
        self.red_latched
    }

    /// Clear a latched invalid-state fault. Callers should only expose this
    /// through an authenticated, operator-visible reset ceremony.
    pub fn clear_red_latch(&mut self) {
        self.red_latched = false;
    }

    pub fn reset(&mut self) {
        self.last_applied = ManipulatorCommand::zero();
        self.red_latched = false;
        self.last_intervention = SafetyIntervention::None;
    }

    #[allow(clippy::too_many_arguments)]
    pub fn supervise(
        &mut self,
        nominal: ManipulatorCommand,
        state: &ManipulatorState,
        kinematics: &ManipulatorKinematics,
        max_torques_nm: &[f64; NUM_JOINTS],
        gravity_hold: [f32; NUM_JOINTS],
        level: MotorSafetyLevel,
        dt: f64,
    ) -> SafetyDecision {
        let measured_force_n = norm3(&state.end_effector_force);
        let clearance = self.predicted_clearance(state, kinematics, level);

        if !state.is_finite() || !dt.is_finite() || dt <= 0.0 {
            self.red_latched = true;
            return self.finish(
                gravity_command(gravity_hold, state.gripper_opening),
                SafetyIntervention::InvalidState,
                clearance,
                measured_force_n,
            );
        }

        if self.red_latched || matches!(level, MotorSafetyLevel::Red) {
            return self.finish(
                gravity_command(gravity_hold, state.gripper_opening),
                SafetyIntervention::RedGravityHold,
                clearance,
                measured_force_n,
            );
        }

        let limits = ManipulatorSafetyLimits::for_level(level);
        if measured_force_n > limits.max_force_n + f64::EPSILON {
            return self.finish(
                gravity_command(gravity_hold, state.gripper_opening),
                SafetyIntervention::ForceLimitHold,
                clearance,
                measured_force_n,
            );
        }

        let mut command = nominal.clamped();
        let mut intervention = if nominal.is_finite() {
            SafetyIntervention::None
        } else {
            SafetyIntervention::SanitizedNonFiniteCommand
        };

        if limits.retreat_required {
            command = self.retreat_command(state, max_torques_nm, gravity_hold, limits);
            intervention = SafetyIntervention::OrangeRetreat;
        } else if clearance < 0.0 {
            command = self.retreat_command(state, max_torques_nm, gravity_hold, limits);
            intervention = SafetyIntervention::WorkspaceRetreat;
        } else {
            if apply_joint_speed_brake(&mut command, state, limits) {
                intervention = SafetyIntervention::JointSpeedBrake;
            }
            if clamp_torque(&mut command, limits.normalized_torque_cap) {
                intervention = SafetyIntervention::TorqueLimited;
            }
        }

        if self.apply_torque_slew_limit(&mut command, dt)
            && matches!(intervention, SafetyIntervention::None)
        {
            intervention = SafetyIntervention::TorqueSlewLimited;
        }

        self.finish(command.clamped(), intervention, clearance, measured_force_n)
    }

    fn predicted_clearance(
        &self,
        state: &ManipulatorState,
        kinematics: &ManipulatorKinematics,
        level: MotorSafetyLevel,
    ) -> f64 {
        let jac = kinematics.jacobian(&state.joint_angles, 1e-6);
        let mut predicted = state.end_effector_position;
        for axis in 0..3 {
            let velocity = (0..NUM_JOINTS)
                .map(|joint| jac[axis][joint] * state.joint_velocities[joint])
                .sum::<f64>();
            predicted[axis] += velocity * self.prediction_horizon_s;
        }
        self.workspace.clearance(&predicted, level)
    }

    fn retreat_command(
        &self,
        state: &ManipulatorState,
        max_torques_nm: &[f64; NUM_JOINTS],
        gravity_hold: [f32; NUM_JOINTS],
        limits: ManipulatorSafetyLimits,
    ) -> ManipulatorCommand {
        let mut command = ManipulatorCommand {
            joint_torques: gravity_hold,
            gripper: state.gripper_opening as f32,
        };
        for joint in 0..NUM_JOINTS {
            let position_error = self.home_joint_angles[joint] - state.joint_angles[joint];
            let corrective_nm = 20.0 * position_error - 5.0 * state.joint_velocities[joint];
            let normalized = corrective_nm / max_torques_nm[joint];
            command.joint_torques[joint] = (gravity_hold[joint] as f64 + normalized) as f32;
        }
        clamp_torque(&mut command, limits.normalized_torque_cap);
        command
    }

    fn apply_torque_slew_limit(&self, command: &mut ManipulatorCommand, dt: f64) -> bool {
        let max_delta = (self.max_normalized_torque_slew_per_s * dt as f32).max(0.0);
        let mut changed = false;
        for joint in 0..NUM_JOINTS {
            let previous = self.last_applied.joint_torques[joint];
            let bounded =
                command.joint_torques[joint].clamp(previous - max_delta, previous + max_delta);
            changed |= bounded != command.joint_torques[joint];
            command.joint_torques[joint] = bounded;
        }
        changed
    }

    fn finish(
        &mut self,
        command: ManipulatorCommand,
        intervention: SafetyIntervention,
        workspace_clearance_m: f64,
        measured_force_n: f64,
    ) -> SafetyDecision {
        self.last_applied = command;
        self.last_intervention = intervention;
        SafetyDecision {
            command,
            intervention,
            workspace_clearance_m,
            measured_force_n,
        }
    }
}

fn gravity_command(gravity_hold: [f32; NUM_JOINTS], gripper_opening: f64) -> ManipulatorCommand {
    ManipulatorCommand {
        joint_torques: gravity_hold,
        gripper: gripper_opening.clamp(0.0, 1.0) as f32,
    }
    .clamped()
}

fn apply_joint_speed_brake(
    command: &mut ManipulatorCommand,
    state: &ManipulatorState,
    limits: ManipulatorSafetyLimits,
) -> bool {
    let mut changed = false;
    for joint in 0..NUM_JOINTS {
        let velocity = state.joint_velocities[joint];
        if velocity.abs() > limits.max_joint_speed_rad_s {
            let requested = command.joint_torques[joint];
            if requested * velocity as f32 > 0.0 {
                command.joint_torques[joint] = 0.0;
                changed = true;
            }
        }
    }
    changed
}

fn clamp_torque(command: &mut ManipulatorCommand, cap: f32) -> bool {
    let mut changed = false;
    for torque in &mut command.joint_torques {
        let bounded = torque.clamp(-cap, cap);
        changed |= bounded != *torque;
        *torque = bounded;
    }
    changed
}

fn norm3(vector: &[f64; 3]) -> f64 {
    (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::ManipulatorPhysicsSimulator;
    use crate::simulator::SimpleManipulatorSimulator;

    fn fixture() -> (
        ManipulatorSafetySupervisor,
        SimpleManipulatorSimulator,
        ManipulatorCommand,
    ) {
        let simulator = SimpleManipulatorSimulator::new();
        let supervisor = ManipulatorSafetySupervisor::new(simulator.state().joint_angles);
        let command = ManipulatorCommand {
            joint_torques: [1.0; NUM_JOINTS],
            gripper: 0.2,
        };
        (supervisor, simulator, command)
    }

    #[test]
    fn red_cannot_be_bypassed_by_nominal_command() {
        let (mut supervisor, simulator, command) = fixture();
        let decision = supervisor.supervise(
            command,
            simulator.state(),
            simulator.kinematics(),
            simulator.max_torques(),
            simulator.gravity_compensation_torques(),
            MotorSafetyLevel::Red,
            0.002,
        );
        assert_eq!(decision.intervention, SafetyIntervention::RedGravityHold);
        assert_eq!(
            decision.command.joint_torques,
            simulator.gravity_compensation_torques()
        );
        assert_eq!(
            decision.command.gripper,
            simulator.state().gripper_opening as f32
        );
    }

    #[test]
    fn orange_generates_home_directed_retreat() {
        let (mut supervisor, mut simulator, command) = fixture();
        let mut disturbance = ManipulatorCommand::zero();
        disturbance.joint_torques[0] = 0.5;
        for _ in 0..20 {
            simulator.step(&disturbance, 0.002);
        }
        let decision = supervisor.supervise(
            command,
            simulator.state(),
            simulator.kinematics(),
            simulator.max_torques(),
            simulator.gravity_compensation_torques(),
            MotorSafetyLevel::Orange,
            0.002,
        );
        assert_eq!(decision.intervention, SafetyIntervention::OrangeRetreat);
        assert!(decision
            .command
            .joint_torques
            .iter()
            .all(|t| t.abs() <= 0.2));
    }

    #[test]
    fn human_zone_forces_workspace_retreat() {
        let (mut supervisor, simulator, command) = fixture();
        let position = simulator.state().end_effector_position;
        supervisor
            .workspace_mut()
            .human_zones
            .push([position[0], position[1], position[2], 0.2]);
        let decision = supervisor.supervise(
            command,
            simulator.state(),
            simulator.kinematics(),
            simulator.max_torques(),
            simulator.gravity_compensation_torques(),
            MotorSafetyLevel::Green,
            0.002,
        );
        assert!(decision.workspace_clearance_m < 0.0);
        assert_ne!(decision.intervention, SafetyIntervention::None);
    }

    #[test]
    fn non_finite_state_latches_red() {
        let (mut supervisor, simulator, command) = fixture();
        let mut state = simulator.state().clone();
        state.joint_velocities[0] = f64::NAN;
        let _ = supervisor.supervise(
            command,
            &state,
            simulator.kinematics(),
            simulator.max_torques(),
            simulator.gravity_compensation_torques(),
            MotorSafetyLevel::Green,
            0.002,
        );
        assert!(supervisor.red_latched());

        let decision = supervisor.supervise(
            command,
            simulator.state(),
            simulator.kinematics(),
            simulator.max_torques(),
            simulator.gravity_compensation_torques(),
            MotorSafetyLevel::Green,
            0.002,
        );
        assert_eq!(decision.intervention, SafetyIntervention::RedGravityHold);
    }
}
