// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Capture-point and protective-posture recovery primitives.
//!
//! This is a deterministic recovery layer, not a replacement for a full rigid
//! body inverse-dynamics solver. It converts support geometry and COM velocity
//! into bounded whole-body corrections that the learned residual policy can
//! refine without owning the final safety authority.

use serde::{Deserialize, Serialize};

use crate::contact::{BipedSupport, ContactFrame};
use crate::morphology::HumanoidMorphology;
use crate::types::{HumanoidCommand, HumanoidState};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecoveryMode {
    Nominal,
    CaptureStep,
    ProtectiveCrouch,
    Fallen,
}

impl Default for RecoveryMode {
    fn default() -> Self {
        Self::Nominal
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureRecoveryConfig {
    pub gravity_mps2: f64,
    pub nominal_com_height_m: f64,
    pub warning_margin_m: f64,
    pub capture_margin_m: f64,
    pub sagittal_gain: f64,
    pub lateral_gain: f64,
    pub crouch_authority: f32,
    pub max_correction: f32,
}

impl Default for CaptureRecoveryConfig {
    fn default() -> Self {
        Self {
            gravity_mps2: 9.81,
            nominal_com_height_m: 0.85,
            warning_margin_m: 0.035,
            capture_margin_m: -0.015,
            sagittal_gain: 1.6,
            lateral_gain: 1.8,
            crouch_authority: 0.22,
            max_correction: 0.38,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RecoveryReport {
    pub mode: RecoveryMode,
    pub capture_point_world_m: [f64; 2],
    pub support_margin_m: f64,
    pub effort: f32,
}

pub struct CapturePointRecoveryController {
    morphology: HumanoidMorphology,
    config: CaptureRecoveryConfig,
}

impl CapturePointRecoveryController {
    pub fn new(morphology: HumanoidMorphology) -> Self {
        Self::with_config(morphology, CaptureRecoveryConfig::default())
    }

    pub fn with_config(morphology: HumanoidMorphology, config: CaptureRecoveryConfig) -> Self {
        Self { morphology, config }
    }

    pub fn correction(
        &self,
        state: &HumanoidState,
        contacts: &ContactFrame,
    ) -> (HumanoidCommand, RecoveryReport) {
        let n = self.morphology.num_actuators();
        let mut command = HumanoidCommand::zero_for(n);
        if state.joint_angles.len() != n {
            return (
                command,
                RecoveryReport {
                    mode: RecoveryMode::Fallen,
                    capture_point_world_m: [state.root_position[0], state.root_position[1]],
                    support_margin_m: f64::NEG_INFINITY,
                    effort: 0.0,
                },
            );
        }

        let com_height = state
            .root_height
            .clamp(0.35, 1.4)
            .max(self.config.nominal_com_height_m * 0.5);
        let omega = (self.config.gravity_mps2 / com_height).sqrt().max(1e-6);
        let capture_point = [
            state.root_position[0] + state.com_velocity[0] / omega,
            state.root_position[1] + state.com_velocity[1] / omega,
        ];
        let margin = contacts.support_margin_m(capture_point);
        let fallen = state.uprightness() < 0.22 || state.head_height < 0.55;
        let mode = if fallen {
            RecoveryMode::Fallen
        } else if matches!(contacts.support(), BipedSupport::Flight) {
            RecoveryMode::ProtectiveCrouch
        } else if margin < self.config.capture_margin_m {
            RecoveryMode::CaptureStep
        } else if margin < self.config.warning_margin_m {
            RecoveryMode::ProtectiveCrouch
        } else {
            RecoveryMode::Nominal
        };

        if mode == RecoveryMode::Nominal || mode == RecoveryMode::Fallen {
            return (
                command,
                RecoveryReport {
                    mode,
                    capture_point_world_m: capture_point,
                    support_margin_m: margin,
                    effort: 0.0,
                },
            );
        }

        let support_center = contacts
            .center_of_pressure_world_m()
            .unwrap_or([state.root_position[0], state.root_position[1]]);
        let sagittal = ((capture_point[0] - support_center[0]) * self.config.sagittal_gain).clamp(
            -(self.config.max_correction as f64),
            self.config.max_correction as f64,
        ) as f32;
        let lateral = ((capture_point[1] - support_center[1]) * self.config.lateral_gain).clamp(
            -(self.config.max_correction as f64),
            self.config.max_correction as f64,
        ) as f32;

        let names = self.morphology.joint_names();
        let mut effort = 0.0f32;
        for (index, name) in names.iter().enumerate() {
            let value = match name.as_str() {
                "right_ankle_y" | "left_ankle_y" => -0.75 * sagittal,
                "right_hip_y" | "left_hip_y" => 0.55 * sagittal,
                "right_ankle_x" | "left_ankle_x" => -0.70 * lateral,
                "right_hip_x" | "left_hip_x" => 0.45 * lateral,
                "right_knee" | "left_knee" if mode == RecoveryMode::ProtectiveCrouch => {
                    -self.config.crouch_authority
                }
                "abdomen_y" => -0.35 * sagittal,
                "abdomen_x" => -0.35 * lateral,
                _ => 0.0,
            };
            command.torques[index] =
                value.clamp(-self.config.max_correction, self.config.max_correction);
            effort += command.torques[index].abs();
        }

        (
            command,
            RecoveryReport {
                mode,
                capture_point_world_m: capture_point,
                support_margin_m: margin,
                effort: effort / n.max(1) as f32,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_com_motion_outside_support_requests_capture_step() {
        let mut state = HumanoidState::standing();
        state.extremities[6..9].copy_from_slice(&[0.0, -0.1, 0.0]);
        state.extremities[9..12].copy_from_slice(&[0.0, 0.1, 0.0]);
        state.com_velocity[0] = 2.0;
        let contacts = ContactFrame::estimated_from_state(&state, 0.03);
        let (_, report) = CapturePointRecoveryController::new(HumanoidMorphology::Dmc21)
            .correction(&state, &contacts);
        assert_eq!(report.mode, RecoveryMode::CaptureStep);
        assert!(report.effort > 0.0);
    }

    #[test]
    fn stable_double_support_remains_nominal() {
        let mut state = HumanoidState::standing();
        state.extremities[6..9].copy_from_slice(&[0.0, -0.1, 0.0]);
        state.extremities[9..12].copy_from_slice(&[0.0, 0.1, 0.0]);
        let contacts = ContactFrame::estimated_from_state(&state, 0.03);
        let (_, report) = CapturePointRecoveryController::new(HumanoidMorphology::Dmc21)
            .correction(&state, &contacts);
        assert_eq!(report.mode, RecoveryMode::Nominal);
    }
}
