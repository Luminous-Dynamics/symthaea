// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Receding-horizon terrain-aware footstep sequence planning.
//!
//! A bounded deterministic beam search evaluates alternating footsteps over a
//! short horizon. Costs include capture-point error, reach, slope, friction,
//! terrain confidence, height discontinuity, and step-to-step smoothness. Only
//! the first step is executed; the horizon is replanned on every control tick.

use serde::{Deserialize, Serialize};

use crate::contact::{BipedSupport, ContactFrame};
use crate::footstep::{FootSide, FootstepPlan, ModelPredictiveFootstepPlanner};
use crate::terrain::TerrainProbe;
use crate::types::HumanoidState;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainMpcConfig {
    pub horizon_steps: usize,
    pub beam_width: usize,
    pub forward_offsets_m: Vec<f64>,
    pub lateral_offsets_m: Vec<f64>,
    pub maximum_step_height_m: f64,
    pub minimum_normal_z: f64,
    pub minimum_friction: f64,
    pub minimum_confidence: f64,
    pub capture_cost: f64,
    pub reach_cost: f64,
    pub slope_cost: f64,
    pub low_friction_cost: f64,
    pub uncertainty_cost: f64,
    pub height_uncertainty_cost: f64,
    pub normal_uncertainty_cost: f64,
    pub friction_uncertainty_cost: f64,
    pub stale_evidence_cost: f64,
    pub maximum_evidence_age_s: f64,
    pub maximum_height_std_m: f64,
    pub height_change_cost: f64,
    pub smoothness_cost: f64,
    pub terminal_velocity_decay: f64,
}

impl Default for TerrainMpcConfig {
    fn default() -> Self {
        Self {
            horizon_steps: 3,
            beam_width: 12,
            forward_offsets_m: vec![-0.10, 0.0, 0.10],
            lateral_offsets_m: vec![-0.06, 0.0, 0.06],
            maximum_step_height_m: 0.18,
            minimum_normal_z: 0.72,
            minimum_friction: 0.35,
            minimum_confidence: 0.40,
            capture_cost: 7.0,
            reach_cost: 2.5,
            slope_cost: 2.0,
            low_friction_cost: 2.5,
            uncertainty_cost: 3.0,
            height_uncertainty_cost: 9.0,
            normal_uncertainty_cost: 2.0,
            friction_uncertainty_cost: 1.5,
            stale_evidence_cost: 2.5,
            maximum_evidence_age_s: 0.25,
            maximum_height_std_m: 0.12,
            height_change_cost: 1.5,
            smoothness_cost: 0.8,
            terminal_velocity_decay: 0.55,
        }
    }
}

impl TerrainMpcConfig {
    pub fn validate(&self) -> bool {
        self.horizon_steps > 0
            && self.beam_width > 0
            && !self.forward_offsets_m.is_empty()
            && !self.lateral_offsets_m.is_empty()
            && self
                .forward_offsets_m
                .iter()
                .chain(self.lateral_offsets_m.iter())
                .all(|value| value.is_finite())
            && self.maximum_step_height_m.is_finite()
            && self.maximum_step_height_m >= 0.0
            && self.minimum_normal_z.is_finite()
            && (0.0..=1.0).contains(&self.minimum_normal_z)
            && self.minimum_friction.is_finite()
            && self.minimum_friction >= 0.0
            && self.minimum_confidence.is_finite()
            && (0.0..=1.0).contains(&self.minimum_confidence)
            && self.maximum_evidence_age_s.is_finite()
            && self.maximum_evidence_age_s >= 0.0
            && self.maximum_height_std_m.is_finite()
            && self.maximum_height_std_m >= 0.0
            && [
                self.capture_cost,
                self.reach_cost,
                self.slope_cost,
                self.low_friction_cost,
                self.uncertainty_cost,
                self.height_uncertainty_cost,
                self.normal_uncertainty_cost,
                self.friction_uncertainty_cost,
                self.stale_evidence_cost,
                self.height_change_cost,
                self.smoothness_cost,
                self.terminal_velocity_decay,
            ]
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainMpcPlan {
    pub footsteps: Vec<FootstepPlan>,
    pub total_cost: f64,
    pub minimum_terrain_confidence: f64,
    pub maximum_slope_rad: f64,
    pub maximum_height_std_m: f64,
    pub maximum_evidence_age_s: f64,
    pub feasible: bool,
    pub evaluated_candidates: usize,
}

impl TerrainMpcPlan {
    pub fn first_step(&self) -> Option<FootstepPlan> {
        self.footsteps.first().copied()
    }
}

#[derive(Clone)]
struct CandidateSequence {
    steps: Vec<FootstepPlan>,
    cost: f64,
    minimum_confidence: f64,
    maximum_slope_rad: f64,
    maximum_height_std_m: f64,
    maximum_evidence_age_s: f64,
    anchor: [f64; 3],
    previous_delta: [f64; 2],
    next_foot: FootSide,
}

pub struct RecedingHorizonTerrainPlanner {
    config: TerrainMpcConfig,
    capture_planner: ModelPredictiveFootstepPlanner,
}

impl RecedingHorizonTerrainPlanner {
    pub fn new() -> Self {
        Self::with_config(TerrainMpcConfig::default())
    }

    pub fn with_config(config: TerrainMpcConfig) -> Self {
        Self {
            config,
            capture_planner: ModelPredictiveFootstepPlanner::new(),
        }
    }

    pub fn plan<T: TerrainProbe + ?Sized>(
        &self,
        state: &HumanoidState,
        contacts: &ContactFrame,
        terrain: &T,
    ) -> TerrainMpcPlan {
        if !self.config.validate() || matches!(contacts.support(), BipedSupport::Flight) {
            return TerrainMpcPlan {
                footsteps: Vec::new(),
                total_cost: f64::INFINITY,
                minimum_terrain_confidence: 0.0,
                maximum_slope_rad: f64::INFINITY,
                maximum_height_std_m: f64::INFINITY,
                maximum_evidence_age_s: f64::INFINITY,
                feasible: false,
                evaluated_candidates: 0,
            };
        }
        let capture = self.capture_planner.plan(state, contacts);
        let initial_anchor = support_anchor(state, contacts, capture.swing_foot);
        let mut frontier = vec![CandidateSequence {
            steps: Vec::new(),
            cost: 0.0,
            minimum_confidence: 1.0,
            maximum_slope_rad: 0.0,
            maximum_height_std_m: 0.0,
            maximum_evidence_age_s: 0.0,
            anchor: initial_anchor,
            previous_delta: [0.0; 2],
            next_foot: capture.swing_foot,
        }];
        let mut evaluated = 0usize;

        for depth in 0..self.config.horizon_steps {
            let mut expanded = Vec::new();
            for sequence in &frontier {
                let nominal = nominal_target(state, &capture, sequence, depth, &self.config);
                for forward_offset in &self.config.forward_offsets_m {
                    for lateral_offset in &self.config.lateral_offsets_m {
                        evaluated += 1;
                        let target_xy = [nominal[0] + forward_offset, nominal[1] + lateral_offset];
                        let sample = terrain.sample(target_xy);
                        if !sample.validate() {
                            continue;
                        }
                        let delta = [
                            target_xy[0] - sequence.anchor[0],
                            target_xy[1] - sequence.anchor[1],
                        ];
                        let reach = reach_utilization(delta);
                        let height_delta = sample.height_m - sequence.anchor[2];
                        let slope = sample.normal_world[2].clamp(-1.0, 1.0).acos();
                        let feasible = reach <= 1.0
                            && height_delta.abs() <= self.config.maximum_step_height_m
                            && sample.normal_world[2] >= self.config.minimum_normal_z
                            && sample.friction >= self.config.minimum_friction
                            && sample.confidence >= self.config.minimum_confidence
                            && sample.height_std_m <= self.config.maximum_height_std_m
                            && sample.age_s <= self.config.maximum_evidence_age_s;
                        if !feasible {
                            continue;
                        }
                        let capture_error = if depth == 0 {
                            (target_xy[0] - capture.predicted_capture_point_world_m[0])
                                .hypot(target_xy[1] - capture.predicted_capture_point_world_m[1])
                        } else {
                            0.25 * (target_xy[0] - capture.predicted_capture_point_world_m[0])
                                .hypot(target_xy[1] - capture.predicted_capture_point_world_m[1])
                        };
                        let smoothness = (delta[0] - sequence.previous_delta[0])
                            .hypot(delta[1] - sequence.previous_delta[1]);
                        let incremental_cost = self.config.capture_cost * capture_error.powi(2)
                            + self.config.reach_cost * reach.powi(4)
                            + self.config.slope_cost * slope.powi(2)
                            + self.config.low_friction_cost
                                * (1.0 / sample.friction.max(0.05) - 1.0).max(0.0)
                            + self.config.uncertainty_cost * (1.0 - sample.confidence)
                            + self.config.height_uncertainty_cost * sample.height_std_m.powi(2)
                            + self.config.normal_uncertainty_cost * sample.normal_std_rad.powi(2)
                            + self.config.friction_uncertainty_cost * sample.friction_std.powi(2)
                            + self.config.stale_evidence_cost * sample.age_s.powi(2)
                            + self.config.height_change_cost * height_delta.abs()
                            + self.config.smoothness_cost * smoothness;
                        let mut steps = sequence.steps.clone();
                        steps.push(FootstepPlan {
                            swing_foot: sequence.next_foot,
                            target_world_m: [target_xy[0], target_xy[1], sample.height_m],
                            predicted_capture_point_world_m: capture
                                .predicted_capture_point_world_m,
                            duration_s: capture.duration_s,
                            clearance_m: capture.clearance_m + height_delta.max(0.0),
                            reach_utilization: reach,
                            feasible: true,
                            confidence: (sample.confidence * (1.0 - 0.35 * reach)).clamp(0.0, 1.0),
                        });
                        expanded.push(CandidateSequence {
                            steps,
                            cost: sequence.cost + incremental_cost,
                            minimum_confidence: sequence.minimum_confidence.min(sample.confidence),
                            maximum_slope_rad: sequence.maximum_slope_rad.max(slope),
                            maximum_height_std_m: sequence
                                .maximum_height_std_m
                                .max(sample.height_std_m),
                            maximum_evidence_age_s: sequence
                                .maximum_evidence_age_s
                                .max(sample.age_s),
                            anchor: [target_xy[0], target_xy[1], sample.height_m],
                            previous_delta: delta,
                            next_foot: opposite(sequence.next_foot),
                        });
                    }
                }
            }
            expanded.sort_by(|left, right| {
                left.cost
                    .partial_cmp(&right.cost)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            expanded.truncate(self.config.beam_width);
            if expanded.is_empty() {
                break;
            }
            frontier = expanded;
        }

        let best = frontier
            .into_iter()
            .filter(|candidate| candidate.steps.len() == self.config.horizon_steps)
            .min_by(|left, right| {
                left.cost
                    .partial_cmp(&right.cost)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        match best {
            Some(best) => TerrainMpcPlan {
                footsteps: best.steps,
                total_cost: best.cost,
                minimum_terrain_confidence: best.minimum_confidence,
                maximum_slope_rad: best.maximum_slope_rad,
                maximum_height_std_m: best.maximum_height_std_m,
                maximum_evidence_age_s: best.maximum_evidence_age_s,
                feasible: true,
                evaluated_candidates: evaluated,
            },
            None => TerrainMpcPlan {
                footsteps: Vec::new(),
                total_cost: f64::INFINITY,
                minimum_terrain_confidence: 0.0,
                maximum_slope_rad: f64::INFINITY,
                maximum_height_std_m: f64::INFINITY,
                maximum_evidence_age_s: f64::INFINITY,
                feasible: false,
                evaluated_candidates: evaluated,
            },
        }
    }
}

impl Default for RecedingHorizonTerrainPlanner {
    fn default() -> Self {
        Self::new()
    }
}

fn support_anchor(
    state: &HumanoidState,
    contacts: &ContactFrame,
    swing_foot: FootSide,
) -> [f64; 3] {
    let stance = match swing_foot {
        FootSide::Right => contacts.left,
        FootSide::Left => contacts.right,
    };
    if stance.in_contact {
        stance.point_world_m
    } else {
        [state.root_position[0], state.root_position[1], 0.0]
    }
}

fn nominal_target(
    state: &HumanoidState,
    capture: &FootstepPlan,
    sequence: &CandidateSequence,
    depth: usize,
    config: &TerrainMpcConfig,
) -> [f64; 2] {
    if depth == 0 {
        return [capture.target_world_m[0], capture.target_world_m[1]];
    }
    let decay = config
        .terminal_velocity_decay
        .clamp(0.0, 1.0)
        .powi(depth as i32);
    let lateral = match sequence.next_foot {
        FootSide::Right => -0.20,
        FootSide::Left => 0.20,
    };
    [
        sequence.anchor[0] + state.com_velocity[0] * capture.duration_s * decay,
        state.root_position[1] + lateral,
    ]
}

fn reach_utilization(delta: [f64; 2]) -> f64 {
    let sagittal_limit = if delta[0] >= 0.0 { 0.50 } else { 0.30 };
    (delta[0].abs() / sagittal_limit)
        .max(delta[1].abs() / 0.34)
        .max(0.0)
}

fn opposite(side: FootSide) -> FootSide {
    match side {
        FootSide::Right => FootSide::Left,
        FootSide::Left => FootSide::Right,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::{FlatTerrain, HeightFieldTerrain};

    #[test]
    fn flat_ground_produces_full_horizon() {
        let mut state = HumanoidState::standing();
        state.com_velocity[0] = 0.5;
        let contacts = ContactFrame::estimated_from_state(&state, 0.05);
        let plan = RecedingHorizonTerrainPlanner::new().plan(&state, &contacts, &FlatTerrain);
        assert!(plan.feasible);
        assert_eq!(plan.footsteps.len(), 3);
        assert!(plan.evaluated_candidates > 0);
    }

    #[test]
    fn unknown_terrain_fails_closed() {
        let state = HumanoidState::standing();
        let contacts = ContactFrame::estimated_from_state(&state, 0.05);
        let terrain = HeightFieldTerrain {
            origin_world_m: [10.0, 10.0],
            cell_size_m: 0.1,
            width: 2,
            height: 2,
            heights_m: vec![0.0; 4],
            friction: 1.0,
            compliance: 0.0,
        };
        let plan = RecedingHorizonTerrainPlanner::new().plan(&state, &contacts, &terrain);
        assert!(!plan.feasible);
        assert!(plan.footsteps.is_empty());
    }

    #[test]
    fn planner_alternates_swing_feet() {
        let state = HumanoidState::standing();
        let contacts = ContactFrame::estimated_from_state(&state, 0.05);
        let plan = RecedingHorizonTerrainPlanner::new().plan(&state, &contacts, &FlatTerrain);
        assert_ne!(plan.footsteps[0].swing_foot, plan.footsteps[1].swing_foot);
        assert_eq!(plan.footsteps[0].swing_foot, plan.footsteps[2].swing_foot);
    }
}
