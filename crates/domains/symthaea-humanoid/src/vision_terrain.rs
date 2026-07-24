// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Vision-uncertainty-aware terrain evidence and conservative fusion.

use serde::{Deserialize, Serialize};

use crate::terrain::{TerrainEvidenceSource, TerrainProbe, TerrainSample};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct VisionTerrainObservation {
    pub world_xy_m: [f64; 2],
    pub sampled_at_s: f64,
    pub height_m: f64,
    pub normal_world: [f64; 3],
    pub friction_estimate: f64,
    pub height_std_m: f64,
    pub normal_std_rad: f64,
    pub friction_std: f64,
    pub confidence: f64,
}

impl VisionTerrainObservation {
    pub fn validate(&self) -> bool {
        self.world_xy_m.iter().all(|value| value.is_finite())
            && self.sampled_at_s.is_finite()
            && self.height_m.is_finite()
            && self.normal_world.iter().all(|value| value.is_finite())
            && norm3(self.normal_world) > 1.0e-9
            && self.friction_estimate.is_finite()
            && self.friction_estimate >= 0.0
            && [self.height_std_m, self.normal_std_rad, self.friction_std]
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
            && self.confidence.is_finite()
            && (0.0..=1.0).contains(&self.confidence)
    }

    pub fn sample_at(&self, now_s: f64) -> TerrainSample {
        let age_s = if now_s.is_finite() && now_s >= self.sampled_at_s {
            now_s - self.sampled_at_s
        } else {
            f64::INFINITY
        };
        TerrainSample {
            height_m: self.height_m,
            normal_world: normalize3(self.normal_world),
            friction: self.friction_estimate,
            compliance: 0.0,
            confidence: if age_s.is_finite() {
                self.confidence
            } else {
                0.0
            },
            height_std_m: self.height_std_m,
            normal_std_rad: self.normal_std_rad,
            friction_std: self.friction_std,
            age_s,
            source: TerrainEvidenceSource::VisionEstimate,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TerrainFusionConfig {
    pub maximum_vision_age_s: f64,
    pub disagreement_sigma_limit: f64,
    pub uncertainty_floor: f64,
}

impl Default for TerrainFusionConfig {
    fn default() -> Self {
        Self {
            maximum_vision_age_s: 0.20,
            disagreement_sigma_limit: 4.0,
            uncertainty_floor: 1.0e-4,
        }
    }
}

pub fn fuse_terrain_samples(
    geometric: TerrainSample,
    vision: TerrainSample,
    config: TerrainFusionConfig,
) -> TerrainSample {
    if !geometric.validate() {
        return vision;
    }
    if !vision.validate()
        || vision.source != TerrainEvidenceSource::VisionEstimate
        || vision.age_s > config.maximum_vision_age_s.max(0.0)
    {
        return geometric;
    }
    let floor = config.uncertainty_floor.max(1.0e-9);
    let geometric_variance = geometric.height_std_m.max(floor).powi(2);
    let vision_variance = vision.height_std_m.max(floor).powi(2);
    let disagreement_sigma = (geometric.height_m - vision.height_m).abs()
        / (geometric_variance + vision_variance).sqrt().max(floor);
    if disagreement_sigma > config.disagreement_sigma_limit.max(0.0) {
        let mut rejected = geometric;
        rejected.confidence *= 0.75;
        rejected.height_std_m = rejected
            .height_std_m
            .max((geometric.height_m - vision.height_m).abs());
        return rejected;
    }
    let geometric_weight = 1.0 / geometric_variance;
    let vision_weight = 1.0 / vision_variance;
    let total_weight = geometric_weight + vision_weight;
    let normal = normalize3([
        geometric_weight * geometric.normal_world[0] + vision_weight * vision.normal_world[0],
        geometric_weight * geometric.normal_world[1] + vision_weight * vision.normal_world[1],
        geometric_weight * geometric.normal_world[2] + vision_weight * vision.normal_world[2],
    ]);
    TerrainSample {
        height_m: (geometric_weight * geometric.height_m + vision_weight * vision.height_m)
            / total_weight,
        normal_world: normal,
        friction: geometric.friction.min(vision.friction),
        compliance: geometric.compliance.max(vision.compliance),
        confidence: geometric.confidence.min(vision.confidence),
        height_std_m: (1.0 / total_weight).sqrt(),
        normal_std_rad: geometric.normal_std_rad.max(vision.normal_std_rad),
        friction_std: geometric.friction_std.max(vision.friction_std),
        age_s: geometric.age_s.max(vision.age_s),
        source: TerrainEvidenceSource::Fused,
    }
}

pub struct VisionFusedTerrain<'a, G: TerrainProbe + ?Sized> {
    geometric: &'a G,
    observations: &'a [VisionTerrainObservation],
    now_s: f64,
    config: TerrainFusionConfig,
    association_radius_m: f64,
}

impl<'a, G: TerrainProbe + ?Sized> VisionFusedTerrain<'a, G> {
    pub fn new(geometric: &'a G, observations: &'a [VisionTerrainObservation], now_s: f64) -> Self {
        Self {
            geometric,
            observations,
            now_s,
            config: TerrainFusionConfig::default(),
            association_radius_m: 0.08,
        }
    }

    pub fn with_config(mut self, config: TerrainFusionConfig) -> Self {
        self.config = config;
        self
    }
}

impl<G: TerrainProbe + ?Sized> TerrainProbe for VisionFusedTerrain<'_, G> {
    fn sample(&self, world_xy_m: [f64; 2]) -> TerrainSample {
        let geometric = self.geometric.sample(world_xy_m);
        let nearest = self
            .observations
            .iter()
            .filter(|observation| observation.validate())
            .min_by(|left, right| {
                distance2(left.world_xy_m, world_xy_m)
                    .partial_cmp(&distance2(right.world_xy_m, world_xy_m))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        match nearest {
            Some(observation)
                if distance2(observation.world_xy_m, world_xy_m)
                    <= self.association_radius_m.powi(2) =>
            {
                fuse_terrain_samples(geometric, observation.sample_at(self.now_s), self.config)
            }
            _ => geometric,
        }
    }
}

fn distance2(left: [f64; 2], right: [f64; 2]) -> f64 {
    (left[0] - right[0]).powi(2) + (left[1] - right[1]).powi(2)
}

fn norm3(vector: [f64; 3]) -> f64 {
    (vector[0].powi(2) + vector[1].powi(2) + vector[2].powi(2)).sqrt()
}

fn normalize3(vector: [f64; 3]) -> [f64; 3] {
    let norm = norm3(vector);
    if norm <= 1.0e-12 {
        [0.0, 0.0, 1.0]
    } else {
        [vector[0] / norm, vector[1] / norm, vector[2] / norm]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::FlatTerrain;

    #[test]
    fn stale_vision_does_not_override_geometric_truth() {
        let observation = VisionTerrainObservation {
            world_xy_m: [0.0, 0.0],
            sampled_at_s: 0.0,
            height_m: 1.0,
            normal_world: [0.0, 0.0, 1.0],
            friction_estimate: 0.8,
            height_std_m: 0.01,
            normal_std_rad: 0.02,
            friction_std: 0.1,
            confidence: 1.0,
        };
        let terrain = FlatTerrain;
        let observations = [observation];
        let fused = VisionFusedTerrain::new(&terrain, &observations, 1.0);
        assert_eq!(fused.sample([0.0, 0.0]).height_m, 0.0);
    }

    #[test]
    fn agreement_reduces_height_uncertainty() {
        let mut geometric = TerrainSample::flat();
        geometric.height_std_m = 0.04;
        let mut vision = TerrainSample::flat();
        vision.source = TerrainEvidenceSource::VisionEstimate;
        vision.height_m = 0.01;
        vision.height_std_m = 0.03;
        let fused = fuse_terrain_samples(geometric, vision, TerrainFusionConfig::default());
        assert_eq!(fused.source, TerrainEvidenceSource::Fused);
        assert!(fused.height_std_m < 0.03);
    }
}
