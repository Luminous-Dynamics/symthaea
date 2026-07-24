// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Protective falling and deterministic get-up sequencing.
//!
//! Falling is treated as a first-class control mode rather than only an episode
//! termination condition. The state machine bounds authority, protects the
//! head and limbs, waits for motion to settle, and only then attempts a staged
//! rise. It remains subordinate to the final safety projector and hardware HAL.

use serde::{Deserialize, Serialize};

use crate::contact::ContactFrame;
use crate::morphology::HumanoidMorphology;
use crate::multi_contact::MultiContactFrame;
use crate::types::{HumanoidCommand, HumanoidState};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FallProtectionPhase {
    Upright,
    Bracing,
    ImpactProtection,
    Settling,
    GetUpReady,
    Rising,
    Faulted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FallOrientation {
    Upright,
    Front,
    Back,
    LeftSide,
    RightSide,
    Unknown,
}

impl Default for FallOrientation {
    fn default() -> Self {
        Self::Unknown
    }
}

impl FallOrientation {
    pub fn from_state(state: &HumanoidState) -> Self {
        let up = state.torso_vertical;
        if up.iter().any(|value| !value.is_finite()) {
            return Self::Unknown;
        }
        if up[2] > 0.65 {
            return Self::Upright;
        }
        if up[0].abs() >= up[1].abs() {
            if up[0] >= 0.0 {
                Self::LeftSide
            } else {
                Self::RightSide
            }
        } else if up[1] >= 0.0 {
            Self::Back
        } else {
            Self::Front
        }
    }
}

impl Default for FallProtectionPhase {
    fn default() -> Self {
        Self::Upright
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallProtectionConfig {
    pub brace_uprightness: f64,
    pub brace_head_height_m: f64,
    pub impact_uprightness: f64,
    pub impact_head_height_m: f64,
    pub brace_angular_rate_rad_s: f64,
    pub settled_angular_rate_rad_s: f64,
    pub minimum_impact_hold_s: f64,
    pub settle_hold_s: f64,
    pub get_up_ready_hold_s: f64,
    pub rising_timeout_s: f64,
    pub recovered_uprightness: f64,
    pub recovered_head_height_m: f64,
    pub max_authority: f32,
}

impl Default for FallProtectionConfig {
    fn default() -> Self {
        Self {
            brace_uprightness: 0.62,
            brace_head_height_m: 0.92,
            impact_uprightness: 0.28,
            impact_head_height_m: 0.62,
            brace_angular_rate_rad_s: 2.2,
            settled_angular_rate_rad_s: 0.55,
            minimum_impact_hold_s: 0.18,
            settle_hold_s: 0.55,
            get_up_ready_hold_s: 0.35,
            rising_timeout_s: 4.5,
            recovered_uprightness: 0.82,
            recovered_head_height_m: 1.10,
            max_authority: 0.42,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FallProtectionReport {
    pub phase: FallProtectionPhase,
    pub orientation: FallOrientation,
    pub phase_elapsed_s: f64,
    pub intervention: bool,
    pub protective_effort: f32,
    pub get_up_progress: f64,
    pub active_contacts: usize,
    pub upper_body_support: bool,
    pub knee_support: bool,
    pub support_polygon_area_m2: f64,
}

pub struct FallProtectionController {
    morphology: HumanoidMorphology,
    config: FallProtectionConfig,
    phase: FallProtectionPhase,
    phase_elapsed_s: f64,
}

impl FallProtectionController {
    pub fn new(morphology: HumanoidMorphology) -> Self {
        Self::with_config(morphology, FallProtectionConfig::default())
    }

    pub const fn with_config(morphology: HumanoidMorphology, config: FallProtectionConfig) -> Self {
        Self {
            morphology,
            config,
            phase: FallProtectionPhase::Upright,
            phase_elapsed_s: 0.0,
        }
    }

    pub const fn phase(&self) -> FallProtectionPhase {
        self.phase
    }

    pub fn reset(&mut self) {
        self.phase = FallProtectionPhase::Upright;
        self.phase_elapsed_s = 0.0;
    }

    pub fn update(
        &mut self,
        state: &HumanoidState,
        contacts: &ContactFrame,
        dt: f64,
    ) -> (HumanoidCommand, FallProtectionReport) {
        let multi = MultiContactFrame::from_feet(contacts).with_protective_candidates(state);
        self.update_with_multi_contacts(state, &multi, dt)
    }

    pub fn update_with_multi_contacts(
        &mut self,
        state: &HumanoidState,
        contacts: &MultiContactFrame,
        dt: f64,
    ) -> (HumanoidCommand, FallProtectionReport) {
        self.phase_elapsed_s += dt.max(0.0);
        let angular_rate = state.angular_momentum();
        let orientation = FallOrientation::from_state(state);
        let settled = angular_rate <= self.config.settled_angular_rate_rad_s;
        let supported = contacts.active_count() > 0;

        let next = match self.phase {
            FallProtectionPhase::Upright => {
                if state.uprightness() < self.config.brace_uprightness
                    || state.head_height < self.config.brace_head_height_m
                    || angular_rate > self.config.brace_angular_rate_rad_s
                {
                    FallProtectionPhase::Bracing
                } else {
                    self.phase
                }
            }
            FallProtectionPhase::Bracing => {
                if state.uprightness() < self.config.impact_uprightness
                    || state.head_height < self.config.impact_head_height_m
                {
                    FallProtectionPhase::ImpactProtection
                } else if state.uprightness() > self.config.recovered_uprightness
                    && state.head_height > self.config.recovered_head_height_m
                    && settled
                {
                    FallProtectionPhase::Upright
                } else {
                    self.phase
                }
            }
            FallProtectionPhase::ImpactProtection => {
                if self.phase_elapsed_s >= self.config.minimum_impact_hold_s && settled {
                    FallProtectionPhase::Settling
                } else {
                    self.phase
                }
            }
            FallProtectionPhase::Settling => {
                if self.phase_elapsed_s >= self.config.settle_hold_s && settled && supported {
                    FallProtectionPhase::GetUpReady
                } else {
                    self.phase
                }
            }
            FallProtectionPhase::GetUpReady => {
                if !supported || !settled {
                    FallProtectionPhase::Settling
                } else if self.phase_elapsed_s >= self.config.get_up_ready_hold_s {
                    FallProtectionPhase::Rising
                } else {
                    self.phase
                }
            }
            FallProtectionPhase::Rising => {
                if state.uprightness() > self.config.recovered_uprightness
                    && state.head_height > self.config.recovered_head_height_m
                    && settled
                {
                    FallProtectionPhase::Upright
                } else if self.phase_elapsed_s >= self.config.rising_timeout_s {
                    FallProtectionPhase::Faulted
                } else {
                    self.phase
                }
            }
            FallProtectionPhase::Faulted => self.phase,
        };
        if next != self.phase {
            self.phase = next;
            self.phase_elapsed_s = 0.0;
        }

        let mut command = HumanoidCommand::zero_for(self.morphology.num_actuators());
        self.write_phase_command(&mut command, state, orientation);
        let protective_effort = command.control_effort();
        let get_up_progress = match self.phase {
            FallProtectionPhase::GetUpReady => 0.15,
            FallProtectionPhase::Rising => {
                let recovery_span =
                    (self.config.recovered_uprightness - self.config.impact_uprightness).max(1e-6);
                ((state.uprightness() - self.config.impact_uprightness) / recovery_span)
                    .clamp(0.15, 1.0)
            }
            FallProtectionPhase::Upright => 1.0,
            _ => 0.0,
        };
        (
            command,
            FallProtectionReport {
                phase: self.phase,
                orientation,
                phase_elapsed_s: self.phase_elapsed_s,
                intervention: self.phase != FallProtectionPhase::Upright,
                protective_effort,
                get_up_progress,
                active_contacts: contacts.active_count(),
                upper_body_support: contacts.has_upper_body_support(),
                knee_support: contacts.has_knee_support(),
                support_polygon_area_m2: contacts.support_polygon_area_m2(),
            },
        )
    }

    fn write_phase_command(
        &self,
        command: &mut HumanoidCommand,
        state: &HumanoidState,
        orientation: FallOrientation,
    ) {
        let authority = self.config.max_authority.clamp(0.0, 1.0);
        let names = self.morphology.joint_names();
        for (index, name) in names.iter().enumerate() {
            let value = match self.phase {
                FallProtectionPhase::Upright | FallProtectionPhase::Faulted => 0.0,
                FallProtectionPhase::Bracing => match name.as_str() {
                    "right_knee" | "left_knee" => -0.55 * authority,
                    "right_hip_y" | "left_hip_y" => 0.32 * authority,
                    "abdomen_y" => -0.35 * authority,
                    "right_shoulder1" | "left_shoulder1" => -0.30 * authority,
                    "right_elbow" | "left_elbow" => -0.42 * authority,
                    _ => 0.0,
                },
                FallProtectionPhase::ImpactProtection => match name.as_str() {
                    "right_knee" | "left_knee" => -0.80 * authority,
                    "right_hip_y" | "left_hip_y" => 0.48 * authority,
                    "abdomen_y" => -0.52 * authority,
                    "right_shoulder1" | "left_shoulder1" => -0.48 * authority,
                    "right_elbow" | "left_elbow" => -0.78 * authority,
                    "neck_pitch" => -0.35 * authority,
                    _ => 0.0,
                },
                FallProtectionPhase::Settling => {
                    let velocity = state.joint_velocities.get(index).copied().unwrap_or(0.0);
                    (-0.04 * velocity as f32).clamp(-0.18 * authority, 0.18 * authority)
                }
                FallProtectionPhase::GetUpReady => {
                    get_up_command(orientation, false, name, authority, self.phase_elapsed_s)
                }
                FallProtectionPhase::Rising => {
                    get_up_command(orientation, true, name, authority, self.phase_elapsed_s)
                }
            };
            command.torques[index] = value.clamp(-authority, authority);
        }
    }
}

fn get_up_command(
    orientation: FallOrientation,
    rising: bool,
    joint: &str,
    authority: f32,
    elapsed_s: f64,
) -> f32 {
    let stage = if rising {
        (elapsed_s / 1.1).clamp(0.0, 1.0) as f32
    } else {
        0.0
    };
    match orientation {
        FallOrientation::Front => match joint {
            "right_knee" | "left_knee" => (-0.72 + 0.72 * stage) * authority,
            "right_hip_y" | "left_hip_y" => (0.48 - 0.75 * stage) * authority,
            "right_shoulder1" | "left_shoulder1" => (-0.55 + 0.55 * stage) * authority,
            "right_elbow" | "left_elbow" => (-0.75 + 0.95 * stage) * authority,
            "abdomen_y" => (-0.28 + 0.68 * stage) * authority,
            "right_ankle_y" | "left_ankle_y" if rising => 0.24 * authority,
            _ => 0.0,
        },
        FallOrientation::Back => match joint {
            "right_knee" | "left_knee" => (-0.82 + 1.35 * stage) * authority,
            "right_hip_y" | "left_hip_y" => (0.60 - 1.00 * stage) * authority,
            "right_shoulder1" | "left_shoulder1" => (0.48 - 0.20 * stage) * authority,
            "right_elbow" | "left_elbow" => -0.38 * authority,
            "abdomen_y" => (-0.48 + 0.88 * stage) * authority,
            "right_ankle_y" | "left_ankle_y" if rising => 0.30 * authority,
            _ => 0.0,
        },
        FallOrientation::LeftSide | FallOrientation::RightSide => {
            let roll_sign = if orientation == FallOrientation::LeftSide {
                -1.0
            } else {
                1.0
            };
            match joint {
                "abdomen_x" => roll_sign * (0.60 - 0.40 * stage) * authority,
                "right_hip_x" | "left_hip_x" => -roll_sign * 0.42 * authority,
                "right_shoulder1" | "left_shoulder1" => -0.40 * authority,
                "right_elbow" | "left_elbow" => -0.55 * authority,
                "right_knee" | "left_knee" => -0.58 * authority,
                _ => 0.0,
            }
        }
        FallOrientation::Upright => match joint {
            "right_knee" | "left_knee" if rising => 0.55 * authority,
            "right_hip_y" | "left_hip_y" if rising => -0.30 * authority,
            "abdomen_y" if rising => 0.28 * authority,
            _ => 0.0,
        },
        FallOrientation::Unknown => match joint {
            "right_knee" | "left_knee" => -0.50 * authority,
            "right_elbow" | "left_elbow" => -0.45 * authority,
            _ => 0.0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orientation_classifies_dominant_lateral_axis() {
        let mut state = HumanoidState::standing();
        state.torso_vertical = [0.9, 0.1, 0.1];
        assert_eq!(
            FallOrientation::from_state(&state),
            FallOrientation::LeftSide
        );
        state.torso_vertical = [0.1, -0.9, 0.1];
        assert_eq!(FallOrientation::from_state(&state), FallOrientation::Front);
    }

    #[test]
    fn falling_state_enters_bracing_then_impact_protection() {
        let mut controller = FallProtectionController::new(HumanoidMorphology::Dmc21);
        let mut state = HumanoidState::standing();
        state.torso_vertical[2] = 0.5;
        let contacts = ContactFrame::estimated_from_state(&state, 0.05);
        let (_, first) = controller.update(&state, &contacts, 0.025);
        assert_eq!(first.phase, FallProtectionPhase::Bracing);
        state.torso_vertical[2] = 0.1;
        state.head_height = 0.4;
        let (command, second) = controller.update(&state, &contacts, 0.025);
        assert_eq!(second.phase, FallProtectionPhase::ImpactProtection);
        assert!(command.control_effort() > 0.0);
    }

    #[test]
    fn rising_timeout_latches_a_fault() {
        let mut controller = FallProtectionController::new(HumanoidMorphology::Dmc21);
        controller.phase = FallProtectionPhase::Rising;
        controller.phase_elapsed_s = controller.config.rising_timeout_s;
        let mut state = HumanoidState::standing();
        state.torso_vertical[2] = 0.2;
        state.head_height = 0.5;
        let contacts = ContactFrame::estimated_from_state(&state, 0.05);
        let (_, report) = controller.update(&state, &contacts, 0.025);
        assert_eq!(report.phase, FallProtectionPhase::Faulted);
    }

    #[test]
    fn upright_state_has_no_protective_authority() {
        let mut controller = FallProtectionController::new(HumanoidMorphology::Dmc21);
        let state = HumanoidState::standing();
        let contacts = ContactFrame::estimated_from_state(&state, 0.05);
        let (command, report) = controller.update(&state, &contacts, 0.025);
        assert_eq!(report.phase, FallProtectionPhase::Upright);
        assert_eq!(command.control_effort(), 0.0);
    }
}
