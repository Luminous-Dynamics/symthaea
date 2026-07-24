// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Predictive, reachability-constrained footstep planning.
//!
//! The planner intentionally remains deterministic and lightweight. It predicts
//! the unstable capture motion over a short horizon, chooses the unloaded foot,
//! and projects the requested landing point into an anatomical reach envelope.
//! A future QP/MPC solver can consume the same plan contract without changing
//! the higher-level recovery state machine.

use serde::{Deserialize, Serialize};

use crate::contact::{BipedSupport, ContactFrame};
use crate::types::HumanoidState;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FootSide {
    Right,
    Left,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FootstepPlannerConfig {
    pub gravity_mps2: f64,
    pub nominal_com_height_m: f64,
    pub prediction_horizon_s: f64,
    pub nominal_step_duration_s: f64,
    pub min_step_duration_s: f64,
    pub max_step_duration_s: f64,
    pub max_forward_reach_m: f64,
    pub max_backward_reach_m: f64,
    pub max_lateral_reach_m: f64,
    pub nominal_stance_half_width_m: f64,
    pub minimum_clearance_m: f64,
}

impl Default for FootstepPlannerConfig {
    fn default() -> Self {
        Self {
            gravity_mps2: 9.81,
            nominal_com_height_m: 0.85,
            prediction_horizon_s: 0.28,
            nominal_step_duration_s: 0.34,
            min_step_duration_s: 0.20,
            max_step_duration_s: 0.55,
            max_forward_reach_m: 0.48,
            max_backward_reach_m: 0.28,
            max_lateral_reach_m: 0.30,
            nominal_stance_half_width_m: 0.10,
            minimum_clearance_m: 0.055,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FootstepPlan {
    pub swing_foot: FootSide,
    pub target_world_m: [f64; 3],
    pub predicted_capture_point_world_m: [f64; 2],
    pub duration_s: f64,
    pub clearance_m: f64,
    pub reach_utilization: f64,
    pub feasible: bool,
    pub confidence: f64,
}

pub struct ModelPredictiveFootstepPlanner {
    config: FootstepPlannerConfig,
}

impl ModelPredictiveFootstepPlanner {
    pub fn new() -> Self {
        Self::with_config(FootstepPlannerConfig::default())
    }

    pub const fn with_config(config: FootstepPlannerConfig) -> Self {
        Self { config }
    }

    pub fn plan(&self, state: &HumanoidState, contacts: &ContactFrame) -> FootstepPlan {
        let support = contacts.support();
        let swing_foot = choose_swing_foot(state, support, contacts);
        let support_center = contacts
            .center_of_pressure_world_m()
            .unwrap_or([state.root_position[0], state.root_position[1]]);
        let com_height = state
            .root_height
            .clamp(0.35, 1.4)
            .max(self.config.nominal_com_height_m * 0.5);
        let omega = (self.config.gravity_mps2 / com_height).sqrt().max(1e-6);
        let capture_point = [
            state.root_position[0] + state.com_velocity[0] / omega,
            state.root_position[1] + state.com_velocity[1] / omega,
        ];
        let horizon = self.config.prediction_horizon_s.max(0.0);
        let unstable_growth = (omega * horizon).exp().clamp(1.0, 8.0);
        let predicted = [
            support_center[0] + (capture_point[0] - support_center[0]) * unstable_growth,
            support_center[1] + (capture_point[1] - support_center[1]) * unstable_growth,
        ];

        let lateral_bias = match swing_foot {
            FootSide::Right => -self.config.nominal_stance_half_width_m,
            FootSide::Left => self.config.nominal_stance_half_width_m,
        };
        let requested = [predicted[0], predicted[1] + lateral_bias];
        let dx = requested[0] - state.root_position[0];
        let dy = requested[1] - state.root_position[1];
        let clamped_dx = dx.clamp(
            -self.config.max_backward_reach_m,
            self.config.max_forward_reach_m,
        );
        let clamped_dy = dy.clamp(
            -self.config.max_lateral_reach_m,
            self.config.max_lateral_reach_m,
        );
        let sagittal_limit = if dx >= 0.0 {
            self.config.max_forward_reach_m
        } else {
            self.config.max_backward_reach_m
        }
        .max(1e-6);
        let sagittal_utilization = (dx.abs() / sagittal_limit).max(0.0);
        let lateral_utilization = (dy.abs() / self.config.max_lateral_reach_m.max(1e-6)).max(0.0);
        let reach_utilization = sagittal_utilization.max(lateral_utilization);
        let feasible = reach_utilization <= 1.0 && !matches!(support, BipedSupport::Flight);

        let speed = state.horizontal_speed();
        let duration_s = (self.config.nominal_step_duration_s / (1.0 + 0.22 * speed)).clamp(
            self.config.min_step_duration_s,
            self.config.max_step_duration_s,
        );
        let source_confidence = match swing_foot {
            FootSide::Right => contacts.left.confidence,
            FootSide::Left => contacts.right.confidence,
        };
        let confidence = if feasible {
            (source_confidence * (1.0 - 0.35 * reach_utilization)).clamp(0.0, 1.0)
        } else {
            0.0
        };

        FootstepPlan {
            swing_foot,
            target_world_m: [
                state.root_position[0] + clamped_dx,
                state.root_position[1] + clamped_dy,
                0.0,
            ],
            predicted_capture_point_world_m: predicted,
            duration_s,
            clearance_m: self.config.minimum_clearance_m,
            reach_utilization,
            feasible,
            confidence,
        }
    }
}

impl Default for ModelPredictiveFootstepPlanner {
    fn default() -> Self {
        Self::new()
    }
}

fn choose_swing_foot(
    state: &HumanoidState,
    support: BipedSupport,
    contacts: &ContactFrame,
) -> FootSide {
    match support {
        BipedSupport::Right => FootSide::Left,
        BipedSupport::Left => FootSide::Right,
        BipedSupport::Double => {
            let lateral_capture = state.root_position[1] + state.com_velocity[1] * 0.2;
            let right_distance =
                (lateral_capture - contacts.right.center_of_pressure_world_m[1]).abs();
            let left_distance =
                (lateral_capture - contacts.left.center_of_pressure_world_m[1]).abs();
            if right_distance <= left_distance {
                FootSide::Left
            } else {
                FootSide::Right
            }
        }
        BipedSupport::Flight => {
            if state.com_velocity[1] >= 0.0 {
                FootSide::Left
            } else {
                FootSide::Right
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::HumanoidState;

    fn standing_contacts(state: &HumanoidState) -> ContactFrame {
        ContactFrame::estimated_from_state(state, 0.03)
    }

    #[test]
    fn forward_capture_motion_requests_a_forward_step() {
        let mut state = HumanoidState::standing();
        state.extremities[6..9].copy_from_slice(&[0.0, -0.1, 0.0]);
        state.extremities[9..12].copy_from_slice(&[0.0, 0.1, 0.0]);
        state.com_velocity[0] = 0.8;
        let plan = ModelPredictiveFootstepPlanner::new().plan(&state, &standing_contacts(&state));
        assert!(plan.target_world_m[0] > state.root_position[0]);
        assert!(plan.duration_s > 0.0);
    }

    #[test]
    fn unreachable_capture_target_is_projected_and_marked_infeasible() {
        let mut state = HumanoidState::standing();
        state.extremities[6..9].copy_from_slice(&[0.0, -0.1, 0.0]);
        state.extremities[9..12].copy_from_slice(&[0.0, 0.1, 0.0]);
        state.com_velocity[0] = 8.0;
        let plan = ModelPredictiveFootstepPlanner::new().plan(&state, &standing_contacts(&state));
        assert!(!plan.feasible);
        assert!(plan.target_world_m[0] <= state.root_position[0] + 0.48 + 1e-9);
    }

    #[test]
    fn single_support_selects_the_unloaded_foot() {
        let mut state = HumanoidState::standing();
        state.extremities[6..9].copy_from_slice(&[0.0, -0.1, 0.0]);
        state.extremities[9..12].copy_from_slice(&[0.0, 0.1, 0.2]);
        let plan = ModelPredictiveFootstepPlanner::new().plan(&state, &standing_contacts(&state));
        assert_eq!(plan.swing_foot, FootSide::Left);
    }
}
