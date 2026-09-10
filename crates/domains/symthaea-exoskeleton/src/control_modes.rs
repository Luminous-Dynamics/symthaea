// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Dual-mode exosuit controller proposals.
//!
//! This module proposes either human-assist or human-resistance torques. The
//! proposals remain non-authoritative until admitted by `SpaceExosuitSafetyKernel`.

use serde::{Deserialize, Serialize};

use crate::space_exosuit::{
    AssistDecision, SpaceEnvironmentState, SpaceExosuitSafetyKernel, SuitSafetyState,
};
use crate::types::{ExoskeletonCommand, NUM_ACTUATORS};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExosuitControlMode {
    /// Move with the wearer to reduce human workload.
    EvaAssist,
    /// Oppose the wearer's motion to create bounded exercise resistance.
    ExerciseResistance,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DualModeRequest {
    /// Estimated human-intent torque direction/magnitude, normalized [-1, 1].
    pub normalized_human_intent: [f32; NUM_ACTUATORS],
    /// Requested assist/resistance fraction in [0, 1].
    pub gain: f32,
    pub stiffness_gain: f32,
    pub damping_gain: f32,
}

impl DualModeRequest {
    pub fn is_valid(&self) -> bool {
        self.normalized_human_intent
            .iter()
            .all(|v| v.is_finite() && (-1.0..=1.0).contains(v))
            && self.gain.is_finite()
            && (0.0..=1.0).contains(&self.gain)
            && self.stiffness_gain.is_finite()
            && (0.0..=1.0).contains(&self.stiffness_gain)
            && self.damping_gain.is_finite()
            && (0.0..=1.0).contains(&self.damping_gain)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DualModeController;

impl DualModeController {
    pub fn propose(
        &self,
        mode: ExosuitControlMode,
        request: &DualModeRequest,
    ) -> Option<ExoskeletonCommand> {
        if !request.is_valid() {
            return None;
        }
        let direction = match mode {
            ExosuitControlMode::EvaAssist => 1.0,
            ExosuitControlMode::ExerciseResistance => -1.0,
        };
        let mut joint_torques = [0.0; NUM_ACTUATORS];
        for (out, intent) in joint_torques
            .iter_mut()
            .zip(request.normalized_human_intent)
        {
            *out = direction * request.gain * intent;
        }
        Some(ExoskeletonCommand {
            joint_torques,
            stiffness_gain: request.stiffness_gain,
            damping_gain: request.damping_gain,
        })
    }

    /// Propose, then pass through the deterministic certified-assist boundary.
    pub fn evaluate(
        &self,
        mode: ExosuitControlMode,
        request: &DualModeRequest,
        kernel: &SpaceExosuitSafetyKernel,
        environment: &SpaceEnvironmentState,
        suit: &SuitSafetyState,
    ) -> AssistDecision {
        match self.propose(mode, request) {
            Some(command) => kernel.evaluate(environment, suit, &command),
            None => kernel.evaluate(
                environment,
                suit,
                &ExoskeletonCommand {
                    joint_torques: [f32::NAN; NUM_ACTUATORS],
                    stiffness_gain: f32::NAN,
                    damping_gain: f32::NAN,
                },
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::space_exosuit::CertifiedAssistEnvelope;

    fn nominal_suit() -> SuitSafetyState {
        SuitSafetyState {
            suit_pressure_pa: 30_000.0,
            oxygen_partial_pressure_pa: 21_000.0,
            co2_partial_pressure_pa: 500.0,
            wearer_core_temperature_k: 310.0,
            assist_battery_soc: 0.8,
            pressure_integrity: 0.99,
            assist_electronics_health: 0.99,
        }
    }

    #[test]
    fn assist_and_exercise_have_opposite_torque_signs() {
        let c = DualModeController;
        let request = DualModeRequest {
            normalized_human_intent: [0.5; NUM_ACTUATORS],
            gain: 0.4,
            stiffness_gain: 0.2,
            damping_gain: 0.2,
        };
        let assist = c.propose(ExosuitControlMode::EvaAssist, &request).unwrap();
        let exercise = c
            .propose(ExosuitControlMode::ExerciseResistance, &request)
            .unwrap();
        assert!(assist.joint_torques[0] > 0.0);
        assert!(exercise.joint_torques[0] < 0.0);
    }

    #[test]
    fn both_modes_remain_inside_same_safety_kernel() {
        let c = DualModeController;
        let kernel = SpaceExosuitSafetyKernel::new(CertifiedAssistEnvelope::simulation_reference());
        let env = SpaceEnvironmentState::lunar_surface(250.0, 0.0);
        let request = DualModeRequest {
            normalized_human_intent: [1.0; NUM_ACTUATORS],
            gain: 1.0,
            stiffness_gain: 1.0,
            damping_gain: 1.0,
        };
        for mode in [ExosuitControlMode::EvaAssist, ExosuitControlMode::ExerciseResistance] {
            let decision = c.evaluate(mode, &request, &kernel, &env, &nominal_suit());
            assert!(decision.permitted);
            assert!(decision.command.joint_torques.iter().all(|t| t.abs() <= 0.35));
        }
    }
}
