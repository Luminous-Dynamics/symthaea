// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Black-box evidence that requested action alone is insufficient to identify
//! the transition dynamics of a backend that performs local actuation adaptation.
//!
//! The current `SimpleHumanoidSimulator` can enable a private actuator model that
//! delays and perturbs commands inside `step()`. R3.11 does not expose that private
//! post-adapter command. Instead it runs two otherwise identical simulator traces
//! with the exact same requested command sequence and isolates the backend-local
//! actuation model as the changed condition.
//!
//! A divergent physical trajectory under that controlled contrast is evidence
//! that an action-conditioned learner must either observe the applied-action
//! boundary or explicitly condition on the backend actuation model. It is not a
//! reconstruction of the hidden applied command and is not hardware execution
//! evidence.

use serde::{Deserialize, Serialize};

use crate::morphology::HumanoidMorphology;
use crate::semantic_action::{
    SemanticHumanoidActuationErrorV1, SemanticHumanoidActuationFrameV1,
};
use crate::simulator::{HumanoidPhysicsSimulator, SimpleHumanoidSimulator};
use crate::types::{ActuationMode, HumanoidCommand, HumanoidState};

/// Configuration for a controlled requested-action causality contrast.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanoidBackendActionConfoundConfigV1 {
    pub schema_id: String,
    pub morphology: HumanoidMorphology,
    /// Same normalized torque command supplied to both simulator traces at every step.
    pub requested_command: HumanoidCommand,
    pub dt_seconds: f64,
    pub steps: usize,
    pub reset_seed: u64,
    /// Noise parameter used by `SimpleHumanoidSimulator::with_actuator_noise`.
    /// A positive value also enables that model's private command-delay path.
    pub actuator_noise_std: f64,
}

impl HumanoidBackendActionConfoundConfigV1 {
    pub const SCHEMA_ID: &'static str = "symthaea.humanoid.backend-action-confound-config.v1";

    pub fn new(
        morphology: HumanoidMorphology,
        requested_command: HumanoidCommand,
        dt_seconds: f64,
        steps: usize,
        reset_seed: u64,
        actuator_noise_std: f64,
    ) -> Self {
        Self {
            schema_id: Self::SCHEMA_ID.to_string(),
            morphology,
            requested_command,
            dt_seconds,
            steps,
            reset_seed,
            actuator_noise_std,
        }
    }

    pub fn validate(&self) -> Result<(), HumanoidBackendActionConfoundErrorV1> {
        if self.schema_id != Self::SCHEMA_ID {
            return Err(HumanoidBackendActionConfoundErrorV1::SchemaMismatch);
        }
        if !self.dt_seconds.is_finite() || self.dt_seconds <= 0.0 {
            return Err(HumanoidBackendActionConfoundErrorV1::InvalidDt);
        }
        if self.steps == 0 || self.steps > 10_000 {
            return Err(HumanoidBackendActionConfoundErrorV1::InvalidStepCount);
        }
        if !self.actuator_noise_std.is_finite() || self.actuator_noise_std < 0.0 {
            return Err(HumanoidBackendActionConfoundErrorV1::InvalidActuatorNoise);
        }
        self.requested_command
            .validate_for(
                self.morphology.num_actuators(),
                ActuationMode::NormalizedTorque,
            )
            .map_err(|_| HumanoidBackendActionConfoundErrorV1::InvalidRequestedCommand)?;
        Ok(())
    }
}

/// Controlled black-box evidence for a backend-local actuation confound.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanoidBackendActionConfoundReportV1 {
    pub schema_id: String,
    pub morphology: HumanoidMorphology,
    pub requested_action: SemanticHumanoidActuationFrameV1,
    pub dt_seconds: f64,
    pub steps: usize,
    pub reset_seed: u64,
    pub actuator_noise_std: f64,
    /// Both traces receive the exact same `requested_command` on every step.
    pub identical_requested_sequence: bool,
    /// First 1-based simulation step where compared true-state channels differ.
    pub first_divergence_step: Option<usize>,
    pub final_timestamp_abs_delta: f64,
    pub root_position_l2_delta: f64,
    pub root_linear_velocity_l2_delta: f64,
    pub root_angular_velocity_l2_delta: f64,
    pub joint_position_l2_delta: f64,
    pub joint_velocity_l2_delta: f64,
    pub policy_channel_l2_delta: f64,
    pub max_abs_compared_state_delta: f64,
}

impl HumanoidBackendActionConfoundReportV1 {
    pub const SCHEMA_ID: &'static str = "symthaea.humanoid.backend-action-confound-report.v1";

    pub fn physical_trajectory_diverged(&self) -> bool {
        self.first_divergence_step.is_some() && self.max_abs_compared_state_delta > 0.0
    }

    /// The probe controls simulation step count/dt; timestamp disagreement would
    /// indicate a harness problem rather than action-model evidence.
    pub fn timestamps_match(&self) -> bool {
        self.final_timestamp_abs_delta <= f64::EPSILON * (self.steps as f64 + 1.0) * 16.0
    }
}

/// Run two deterministic simulator traces from the same initial state and with
/// the same requested action sequence. The only configured contrast is whether
/// the backend-local actuator delay/noise path is enabled.
pub fn measure_backend_action_confound_v1(
    config: &HumanoidBackendActionConfoundConfigV1,
) -> Result<HumanoidBackendActionConfoundReportV1, HumanoidBackendActionConfoundErrorV1> {
    config.validate()?;

    let initial_state = HumanoidState::standing_for(config.morphology);
    let requested_action =
        SemanticHumanoidActuationFrameV1::project_requested_normalized_torque_command(
            &config.requested_command,
            &initial_state,
            config.morphology,
        )?;

    let mut baseline = SimpleHumanoidSimulator::new_for(config.morphology);
    let mut adapted = SimpleHumanoidSimulator::new_for(config.morphology)
        .with_actuator_noise(config.actuator_noise_std);

    baseline.reset_with_perturbation(0.0, config.reset_seed);
    adapted.reset_with_perturbation(0.0, config.reset_seed);

    let mut first_divergence_step = None;
    for step in 1..=config.steps {
        baseline.step(&config.requested_command, config.dt_seconds);
        adapted.step(&config.requested_command, config.dt_seconds);

        if first_divergence_step.is_none()
            && max_compared_state_delta(baseline.true_state(), adapted.true_state()) > 0.0
        {
            first_divergence_step = Some(step);
        }
    }

    let left = baseline.true_state();
    let right = adapted.true_state();
    left.validate_for(config.morphology)
        .map_err(|_| HumanoidBackendActionConfoundErrorV1::InvalidBaselineState)?;
    right
        .validate_for(config.morphology)
        .map_err(|_| HumanoidBackendActionConfoundErrorV1::InvalidAdaptedState)?;

    Ok(HumanoidBackendActionConfoundReportV1 {
        schema_id: HumanoidBackendActionConfoundReportV1::SCHEMA_ID.to_string(),
        morphology: config.morphology,
        requested_action,
        dt_seconds: config.dt_seconds,
        steps: config.steps,
        reset_seed: config.reset_seed,
        actuator_noise_std: config.actuator_noise_std,
        identical_requested_sequence: true,
        first_divergence_step,
        final_timestamp_abs_delta: (left.timestamp - right.timestamp).abs(),
        root_position_l2_delta: l2_fixed(&left.root_position, &right.root_position),
        root_linear_velocity_l2_delta: l2_fixed(
            &left.root_linear_velocity,
            &right.root_linear_velocity,
        ),
        root_angular_velocity_l2_delta: l2_fixed(
            &left.root_angular_velocity,
            &right.root_angular_velocity,
        ),
        joint_position_l2_delta: l2_slice(&left.joint_angles, &right.joint_angles),
        joint_velocity_l2_delta: l2_slice(&left.joint_velocities, &right.joint_velocities),
        policy_channel_l2_delta: l2_f32_slice(&left.to_channels(), &right.to_channels()),
        max_abs_compared_state_delta: max_compared_state_delta(left, right),
    })
}

fn l2_fixed<const N: usize>(left: &[f64; N], right: &[f64; N]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn l2_slice(left: &[f64], right: &[f64]) -> f64 {
    if left.len() != right.len() {
        return f64::INFINITY;
    }
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn l2_f32_slice(left: &[f32], right: &[f32]) -> f64 {
    if left.len() != right.len() {
        return f64::INFINITY;
    }
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| {
            let delta = f64::from(*a) - f64::from(*b);
            delta * delta
        })
        .sum::<f64>()
        .sqrt()
}

fn max_compared_state_delta(left: &HumanoidState, right: &HumanoidState) -> f64 {
    let mut max_delta = (left.root_height - right.root_height).abs();
    for (a, b) in left.root_position.iter().zip(right.root_position.iter()) {
        max_delta = max_delta.max((a - b).abs());
    }
    for (a, b) in left
        .root_linear_velocity
        .iter()
        .zip(right.root_linear_velocity.iter())
    {
        max_delta = max_delta.max((a - b).abs());
    }
    for (a, b) in left
        .root_angular_velocity
        .iter()
        .zip(right.root_angular_velocity.iter())
    {
        max_delta = max_delta.max((a - b).abs());
    }
    for (a, b) in left.joint_angles.iter().zip(right.joint_angles.iter()) {
        max_delta = max_delta.max((a - b).abs());
    }
    for (a, b) in left
        .joint_velocities
        .iter()
        .zip(right.joint_velocities.iter())
    {
        max_delta = max_delta.max((a - b).abs());
    }
    max_delta
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HumanoidBackendActionConfoundErrorV1 {
    SchemaMismatch,
    InvalidDt,
    InvalidStepCount,
    InvalidActuatorNoise,
    InvalidRequestedCommand,
    InvalidBaselineState,
    InvalidAdaptedState,
    SemanticAction(SemanticHumanoidActuationErrorV1),
}

impl From<SemanticHumanoidActuationErrorV1> for HumanoidBackendActionConfoundErrorV1 {
    fn from(value: SemanticHumanoidActuationErrorV1) -> Self {
        Self::SemanticAction(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn command() -> HumanoidCommand {
        let mut torques = vec![0.0; HumanoidMorphology::Dmc21.num_actuators()];
        torques[0] = 0.5;
        torques[5] = -0.25;
        HumanoidCommand { torques }
    }

    #[test]
    fn zero_adaptation_is_an_exact_negative_control() {
        let config = HumanoidBackendActionConfoundConfigV1::new(
            HumanoidMorphology::Dmc21,
            command(),
            0.01,
            5,
            7,
            0.0,
        );
        let report = measure_backend_action_confound_v1(&config).unwrap();

        assert!(report.identical_requested_sequence);
        assert_eq!(report.first_divergence_step, None);
        assert_eq!(report.max_abs_compared_state_delta, 0.0);
        assert_eq!(report.policy_channel_l2_delta, 0.0);
        assert!(report.timestamps_match());
    }

    #[test]
    fn backend_local_actuation_adaptation_breaks_requested_action_sufficiency() {
        let config = HumanoidBackendActionConfoundConfigV1::new(
            HumanoidMorphology::Dmc21,
            command(),
            0.01,
            5,
            7,
            0.03,
        );
        let report = measure_backend_action_confound_v1(&config).unwrap();

        assert!(report.identical_requested_sequence);
        assert!(report.physical_trajectory_diverged());
        assert!(report.first_divergence_step.is_some());
        assert!(report.joint_velocity_l2_delta > 0.0);
        assert!(report.policy_channel_l2_delta > 0.0);
        assert!(report.max_abs_compared_state_delta > 0.0);
        assert!(report.timestamps_match());
        assert_eq!(report.requested_action.values[0].value, 50.0);
    }

    #[test]
    fn probe_rejects_malformed_experiments() {
        let mut config = HumanoidBackendActionConfoundConfigV1::new(
            HumanoidMorphology::Dmc21,
            command(),
            0.01,
            5,
            7,
            0.03,
        );
        config.dt_seconds = 0.0;
        assert_eq!(
            measure_backend_action_confound_v1(&config).unwrap_err(),
            HumanoidBackendActionConfoundErrorV1::InvalidDt
        );

        config.dt_seconds = 0.01;
        config.actuator_noise_std = f64::NAN;
        assert_eq!(
            measure_backend_action_confound_v1(&config).unwrap_err(),
            HumanoidBackendActionConfoundErrorV1::InvalidActuatorNoise
        );
    }
}
