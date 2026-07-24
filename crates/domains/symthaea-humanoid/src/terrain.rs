// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Terrain queries and clearance-aware swing-foot trajectories.

use serde::{Deserialize, Serialize};

use crate::footstep::{FootSide, FootstepPlan};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TerrainEvidenceSource {
    Analytic,
    SimulatorTruth,
    VisionEstimate,
    Fused,
    Unknown,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TerrainSample {
    pub height_m: f64,
    pub normal_world: [f64; 3],
    pub friction: f64,
    pub compliance: f64,
    pub confidence: f64,
    /// One-sigma vertical uncertainty.
    #[serde(default)]
    pub height_std_m: f64,
    /// One-sigma surface-normal angular uncertainty.
    #[serde(default)]
    pub normal_std_rad: f64,
    /// One-sigma friction uncertainty.
    #[serde(default)]
    pub friction_std: f64,
    /// Age of the evidence when sampled.
    #[serde(default)]
    pub age_s: f64,
    #[serde(default = "default_terrain_source")]
    pub source: TerrainEvidenceSource,
}

impl TerrainSample {
    pub const fn flat() -> Self {
        Self {
            height_m: 0.0,
            normal_world: [0.0, 0.0, 1.0],
            friction: 1.0,
            compliance: 0.0,
            confidence: 1.0,
            height_std_m: 0.0,
            normal_std_rad: 0.0,
            friction_std: 0.0,
            age_s: 0.0,
            source: TerrainEvidenceSource::Analytic,
        }
    }

    pub fn validate(&self) -> bool {
        self.height_m.is_finite()
            && self.normal_world.iter().all(|value| value.is_finite())
            && self.friction.is_finite()
            && self.friction >= 0.0
            && self.compliance.is_finite()
            && self.compliance >= 0.0
            && self.confidence.is_finite()
            && (0.0..=1.0).contains(&self.confidence)
            && self.height_std_m.is_finite()
            && self.height_std_m >= 0.0
            && self.normal_std_rad.is_finite()
            && self.normal_std_rad >= 0.0
            && self.friction_std.is_finite()
            && self.friction_std >= 0.0
            && self.age_s.is_finite()
            && self.age_s >= 0.0
            && vector_norm(self.normal_world) > 1.0e-9
    }
}

pub trait TerrainProbe {
    fn sample(&self, world_xy_m: [f64; 2]) -> TerrainSample;
}

impl<T> TerrainProbe for T
where
    T: crate::simulator::HumanoidPhysicsSimulator + ?Sized,
{
    fn sample(&self, world_xy_m: [f64; 2]) -> TerrainSample {
        self.terrain_sample(world_xy_m)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FlatTerrain;

impl TerrainProbe for FlatTerrain {
    fn sample(&self, _world_xy_m: [f64; 2]) -> TerrainSample {
        TerrainSample::flat()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeightFieldTerrain {
    pub origin_world_m: [f64; 2],
    pub cell_size_m: f64,
    pub width: usize,
    pub height: usize,
    pub heights_m: Vec<f64>,
    pub friction: f64,
    pub compliance: f64,
}

impl HeightFieldTerrain {
    pub fn validate(&self) -> bool {
        self.width >= 2
            && self.height >= 2
            && self.heights_m.len() == self.width * self.height
            && self.cell_size_m.is_finite()
            && self.cell_size_m > 0.0
            && self.heights_m.iter().all(|value| value.is_finite())
            && self.friction.is_finite()
            && self.friction >= 0.0
            && self.compliance.is_finite()
            && self.compliance >= 0.0
    }

    fn height_at_index(&self, x: usize, y: usize) -> f64 {
        self.heights_m[y * self.width + x]
    }
}

impl TerrainProbe for HeightFieldTerrain {
    fn sample(&self, world_xy_m: [f64; 2]) -> TerrainSample {
        if !self.validate() {
            return TerrainSample {
                confidence: 0.0,
                ..TerrainSample::flat()
            };
        }
        let raw_gx = (world_xy_m[0] - self.origin_world_m[0]) / self.cell_size_m;
        let raw_gy = (world_xy_m[1] - self.origin_world_m[1]) / self.cell_size_m;
        let inside = raw_gx >= 0.0
            && raw_gx <= self.width.saturating_sub(1) as f64
            && raw_gy >= 0.0
            && raw_gy <= self.height.saturating_sub(1) as f64;
        let gx = raw_gx.clamp(0.0, self.width.saturating_sub(1) as f64);
        let gy = raw_gy.clamp(0.0, self.height.saturating_sub(1) as f64);
        let x0 = gx.floor() as usize;
        let y0 = gy.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let tx = gx - x0 as f64;
        let ty = gy - y0 as f64;
        let h00 = self.height_at_index(x0, y0);
        let h10 = self.height_at_index(x1, y0);
        let h01 = self.height_at_index(x0, y1);
        let h11 = self.height_at_index(x1, y1);
        let hx0 = h00 + (h10 - h00) * tx;
        let hx1 = h01 + (h11 - h01) * tx;
        let height_m = hx0 + (hx1 - hx0) * ty;
        let dzdx = ((h10 - h00) * (1.0 - ty) + (h11 - h01) * ty) / self.cell_size_m;
        let dzdy = ((h01 - h00) * (1.0 - tx) + (h11 - h10) * tx) / self.cell_size_m;
        let normal_world = normalize([-dzdx, -dzdy, 1.0]);
        TerrainSample {
            height_m,
            normal_world,
            friction: self.friction,
            compliance: self.compliance,
            confidence: if inside { 1.0 } else { 0.0 },
            height_std_m: if inside { 0.002 } else { 0.25 },
            normal_std_rad: if inside { 0.01 } else { 1.0 },
            friction_std: if inside { 0.02 } else { 1.0 },
            age_s: 0.0,
            source: TerrainEvidenceSource::Analytic,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SwingTrajectoryConfig {
    pub nominal_clearance_m: f64,
    pub terrain_clearance_margin_m: f64,
    pub maximum_clearance_m: f64,
    pub touchdown_velocity_mps: f64,
    pub minimum_terrain_confidence: f64,
    pub path_samples: usize,
}

impl Default for SwingTrajectoryConfig {
    fn default() -> Self {
        Self {
            nominal_clearance_m: 0.07,
            terrain_clearance_margin_m: 0.035,
            maximum_clearance_m: 0.24,
            touchdown_velocity_mps: -0.12,
            minimum_terrain_confidence: 0.5,
            path_samples: 9,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SwingTrajectory {
    pub foot: FootSide,
    pub start_world_m: [f64; 3],
    pub apex_world_m: [f64; 3],
    pub target_world_m: [f64; 3],
    pub target_normal_world: [f64; 3],
    pub duration_s: f64,
    pub touchdown_velocity_mps: f64,
    pub maximum_obstacle_height_m: f64,
    pub clearance_m: f64,
    pub terrain_confidence: f64,
    pub feasible: bool,
}

impl SwingTrajectory {
    pub fn sample(&self, phase: f64) -> [f64; 3] {
        let t = phase.clamp(0.0, 1.0);
        let one_minus_t = 1.0 - t;
        [
            one_minus_t * one_minus_t * self.start_world_m[0]
                + 2.0 * one_minus_t * t * self.apex_world_m[0]
                + t * t * self.target_world_m[0],
            one_minus_t * one_minus_t * self.start_world_m[1]
                + 2.0 * one_minus_t * t * self.apex_world_m[1]
                + t * t * self.target_world_m[1],
            one_minus_t * one_minus_t * self.start_world_m[2]
                + 2.0 * one_minus_t * t * self.apex_world_m[2]
                + t * t * self.target_world_m[2],
        ]
    }
}

pub struct TerrainAwareSwingPlanner {
    config: SwingTrajectoryConfig,
}

impl TerrainAwareSwingPlanner {
    pub fn new() -> Self {
        Self::with_config(SwingTrajectoryConfig::default())
    }

    pub const fn with_config(config: SwingTrajectoryConfig) -> Self {
        Self { config }
    }

    pub fn plan<T: TerrainProbe + ?Sized>(
        &self,
        footstep: &FootstepPlan,
        start_world_m: [f64; 3],
        terrain: &T,
    ) -> SwingTrajectory {
        let samples = self.config.path_samples.max(3);
        let mut maximum_obstacle_height_m = f64::NEG_INFINITY;
        let mut minimum_confidence = 1.0f64;
        for index in 0..samples {
            let t = index as f64 / (samples - 1) as f64;
            let xy = [
                start_world_m[0] + (footstep.target_world_m[0] - start_world_m[0]) * t,
                start_world_m[1] + (footstep.target_world_m[1] - start_world_m[1]) * t,
            ];
            let sample = terrain.sample(xy);
            if !sample.validate() {
                minimum_confidence = 0.0;
                continue;
            }
            maximum_obstacle_height_m = maximum_obstacle_height_m.max(sample.height_m);
            minimum_confidence = minimum_confidence.min(sample.confidence);
        }
        if !maximum_obstacle_height_m.is_finite() {
            maximum_obstacle_height_m = start_world_m[2].min(footstep.target_world_m[2]);
        }
        let target_sample =
            terrain.sample([footstep.target_world_m[0], footstep.target_world_m[1]]);
        let target_height = if target_sample.validate() {
            target_sample.height_m
        } else {
            footstep.target_world_m[2]
        };
        let target_world_m = [
            footstep.target_world_m[0],
            footstep.target_world_m[1],
            target_height,
        ];
        let base_height = start_world_m[2].max(target_world_m[2]);
        let required_clearance = (maximum_obstacle_height_m - base_height).max(0.0)
            + self.config.terrain_clearance_margin_m;
        let clearance_m = self
            .config
            .nominal_clearance_m
            .max(footstep.clearance_m)
            .max(required_clearance)
            .min(self.config.maximum_clearance_m);
        let apex_world_m = [
            0.5 * (start_world_m[0] + target_world_m[0]),
            0.5 * (start_world_m[1] + target_world_m[1]),
            base_height + clearance_m,
        ];
        let feasible = footstep.feasible
            && minimum_confidence >= self.config.minimum_terrain_confidence
            && required_clearance <= self.config.maximum_clearance_m;
        SwingTrajectory {
            foot: footstep.swing_foot,
            start_world_m,
            apex_world_m,
            target_world_m,
            target_normal_world: if target_sample.validate() {
                normalize(target_sample.normal_world)
            } else {
                [0.0, 0.0, 1.0]
            },
            duration_s: footstep.duration_s,
            touchdown_velocity_mps: self.config.touchdown_velocity_mps,
            maximum_obstacle_height_m,
            clearance_m,
            terrain_confidence: minimum_confidence,
            feasible,
        }
    }
}

impl Default for TerrainAwareSwingPlanner {
    fn default() -> Self {
        Self::new()
    }
}

const fn default_terrain_source() -> TerrainEvidenceSource {
    TerrainEvidenceSource::Unknown
}

fn vector_norm(vector: [f64; 3]) -> f64 {
    (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt()
}

fn normalize(vector: [f64; 3]) -> [f64; 3] {
    let norm = vector_norm(vector);
    if norm <= 1.0e-12 {
        [0.0, 0.0, 1.0]
    } else {
        [vector[0] / norm, vector[1] / norm, vector[2] / norm]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan() -> FootstepPlan {
        FootstepPlan {
            swing_foot: FootSide::Right,
            target_world_m: [0.4, -0.1, 0.0],
            predicted_capture_point_world_m: [0.35, -0.08],
            duration_s: 0.35,
            clearance_m: 0.06,
            reach_utilization: 0.5,
            feasible: true,
            confidence: 1.0,
        }
    }

    #[test]
    fn flat_terrain_produces_clearance_above_endpoints() {
        let trajectory =
            TerrainAwareSwingPlanner::new().plan(&plan(), [0.0, -0.1, 0.0], &FlatTerrain);
        assert!(trajectory.feasible);
        assert!(trajectory.apex_world_m[2] > trajectory.start_world_m[2]);
        assert!(trajectory.apex_world_m[2] > trajectory.target_world_m[2]);
    }

    #[test]
    fn height_field_raises_swing_over_obstacle() {
        let terrain = HeightFieldTerrain {
            origin_world_m: [0.0, -0.2],
            cell_size_m: 0.1,
            width: 5,
            height: 3,
            heights_m: vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            friction: 0.8,
            compliance: 0.0,
        };
        let trajectory = TerrainAwareSwingPlanner::new().plan(&plan(), [0.0, -0.1, 0.0], &terrain);
        assert!(trajectory.maximum_obstacle_height_m >= 0.12 - 1.0e-9);
        assert!(trajectory.apex_world_m[2] > 0.12);
    }

    #[test]
    fn bezier_samples_start_and_target_exactly() {
        let trajectory =
            TerrainAwareSwingPlanner::new().plan(&plan(), [0.0, -0.1, 0.0], &FlatTerrain);
        assert_eq!(trajectory.sample(0.0), trajectory.start_world_m);
        assert_eq!(trajectory.sample(1.0), trajectory.target_world_m);
    }
    #[test]
    fn samples_outside_height_field_are_untrusted() {
        let terrain = HeightFieldTerrain {
            origin_world_m: [0.0, 0.0],
            cell_size_m: 0.1,
            width: 2,
            height: 2,
            heights_m: vec![0.0; 4],
            friction: 0.8,
            compliance: 0.0,
        };
        assert_eq!(terrain.sample([10.0, 10.0]).confidence, 0.0);
    }
}
