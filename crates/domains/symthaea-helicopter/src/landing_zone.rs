// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic emergency landing-zone evaluation.
//!
//! Candidate sites are rejected fail-closed on missing terrain, excessive
//! slope/roughness, geofence violations, blocked approach corridors, or
//! crosswind limits. Viable sites receive a normalized score with stable
//! tie-breaking so replay selects the same diversion target.

use serde::{Deserialize, Serialize};

use crate::terrain_safety::{AxisAlignedGeofence, TerrainProvider};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LandingZoneCandidate {
    pub candidate_id: String,
    /// East, north, and nominal touchdown altitude in the local frame.
    pub local_position_m: [f64; 3],
    /// Final approach direction of travel, radians clockwise from east.
    pub approach_heading_rad: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LandingZoneConfig {
    pub footprint_radius_m: f64,
    pub maximum_slope_deg: f64,
    pub maximum_roughness_m: f64,
    pub minimum_geofence_margin_m: f64,
    pub approach_length_m: f64,
    pub approach_samples: usize,
    pub approach_height_m: f64,
    pub maximum_crosswind_mps: f64,
    pub distance_scale_m: f64,
}

impl Default for LandingZoneConfig {
    fn default() -> Self {
        Self {
            footprint_radius_m: 8.0,
            maximum_slope_deg: 7.0,
            maximum_roughness_m: 0.75,
            minimum_geofence_margin_m: 5.0,
            approach_length_m: 60.0,
            approach_samples: 6,
            approach_height_m: 18.0,
            maximum_crosswind_mps: 10.0,
            distance_scale_m: 2_000.0,
        }
    }
}

impl LandingZoneConfig {
    pub fn validate(&self) -> bool {
        [
            self.footprint_radius_m,
            self.maximum_slope_deg,
            self.maximum_roughness_m,
            self.minimum_geofence_margin_m,
            self.approach_length_m,
            self.approach_height_m,
            self.maximum_crosswind_mps,
            self.distance_scale_m,
        ]
        .iter()
        .all(|value| value.is_finite() && *value > 0.0)
            && self.approach_samples > 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LandingZoneRejection {
    TerrainUnavailable,
    OutsideGeofence,
    InsufficientGeofenceMargin,
    ExcessiveSlope,
    ExcessiveRoughness,
    ApproachOutsideGeofence,
    ApproachTerrainConflict,
    ExcessiveCrosswind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LandingZoneAssessment {
    pub candidate_id: String,
    pub viable: bool,
    pub score: f64,
    pub ground_elevation_m: f64,
    pub maximum_slope_deg: f64,
    pub roughness_m: f64,
    pub geofence_margin_m: f64,
    pub crosswind_mps: f64,
    pub horizontal_distance_m: f64,
    pub rejections: Vec<LandingZoneRejection>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LandingZoneError {
    InvalidConfiguration,
    InvalidCandidate,
    InvalidVehicleState,
    NoCandidates,
    NoViableCandidate,
}

#[derive(Debug, Clone)]
pub struct LandingZoneEvaluator {
    config: LandingZoneConfig,
}

impl Default for LandingZoneEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl LandingZoneEvaluator {
    pub fn new() -> Self {
        Self {
            config: LandingZoneConfig::default(),
        }
    }

    pub fn with_config(config: LandingZoneConfig) -> Result<Self, LandingZoneError> {
        if !config.validate() {
            return Err(LandingZoneError::InvalidConfiguration);
        }
        Ok(Self { config })
    }

    pub fn config(&self) -> LandingZoneConfig {
        self.config
    }

    pub fn evaluate<T: TerrainProvider>(
        &self,
        candidate: &LandingZoneCandidate,
        vehicle_position_m: [f64; 3],
        wind_velocity_mps: [f64; 3],
        terrain: &T,
        geofence: &AxisAlignedGeofence,
    ) -> Result<LandingZoneAssessment, LandingZoneError> {
        if !self.config.validate() || !geofence.validate() {
            return Err(LandingZoneError::InvalidConfiguration);
        }
        if candidate.candidate_id.trim().is_empty()
            || !candidate
                .local_position_m
                .iter()
                .all(|value| value.is_finite())
            || !candidate.approach_heading_rad.is_finite()
        {
            return Err(LandingZoneError::InvalidCandidate);
        }
        if !vehicle_position_m.iter().all(|value| value.is_finite())
            || !wind_velocity_mps.iter().all(|value| value.is_finite())
        {
            return Err(LandingZoneError::InvalidVehicleState);
        }

        let east = candidate.local_position_m[0];
        let north = candidate.local_position_m[1];
        let mut rejections = Vec::new();
        let Some(center_ground) = terrain.elevation_m(east, north) else {
            return Ok(LandingZoneAssessment {
                candidate_id: candidate.candidate_id.clone(),
                viable: false,
                score: 0.0,
                ground_elevation_m: f64::NAN,
                maximum_slope_deg: f64::NAN,
                roughness_m: f64::NAN,
                geofence_margin_m: f64::NAN,
                crosswind_mps: f64::NAN,
                horizontal_distance_m: horizontal_distance(vehicle_position_m, [east, north, 0.0]),
                rejections: vec![LandingZoneRejection::TerrainUnavailable],
            });
        };

        let touchdown = [east, north, center_ground];
        if !geofence.contains(touchdown) {
            rejections.push(LandingZoneRejection::OutsideGeofence);
        }
        let geofence_margin_m = horizontal_geofence_margin(geofence, east, north);
        if geofence_margin_m < self.config.minimum_geofence_margin_m {
            rejections.push(LandingZoneRejection::InsufficientGeofenceMargin);
        }

        let directions = [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [
                std::f64::consts::FRAC_1_SQRT_2,
                std::f64::consts::FRAC_1_SQRT_2,
            ],
            [
                std::f64::consts::FRAC_1_SQRT_2,
                -std::f64::consts::FRAC_1_SQRT_2,
            ],
            [
                -std::f64::consts::FRAC_1_SQRT_2,
                std::f64::consts::FRAC_1_SQRT_2,
            ],
            [
                -std::f64::consts::FRAC_1_SQRT_2,
                -std::f64::consts::FRAC_1_SQRT_2,
            ],
        ];
        let mut minimum_ground = center_ground;
        let mut maximum_ground = center_ground;
        let mut maximum_slope_deg: f64 = 0.0;
        for direction in directions {
            let sample_east = east + direction[0] * self.config.footprint_radius_m;
            let sample_north = north + direction[1] * self.config.footprint_radius_m;
            let Some(sample_ground) = terrain.elevation_m(sample_east, sample_north) else {
                rejections.push(LandingZoneRejection::TerrainUnavailable);
                continue;
            };
            minimum_ground = minimum_ground.min(sample_ground);
            maximum_ground = maximum_ground.max(sample_ground);
            let slope_deg = ((sample_ground - center_ground).abs()
                / self.config.footprint_radius_m)
                .atan()
                .to_degrees();
            maximum_slope_deg = maximum_slope_deg.max(slope_deg);
        }
        let roughness_m = maximum_ground - minimum_ground;
        if maximum_slope_deg > self.config.maximum_slope_deg {
            rejections.push(LandingZoneRejection::ExcessiveSlope);
        }
        if roughness_m > self.config.maximum_roughness_m {
            rejections.push(LandingZoneRejection::ExcessiveRoughness);
        }

        let approach_unit = [
            candidate.approach_heading_rad.cos(),
            candidate.approach_heading_rad.sin(),
        ];
        for sample in 1..=self.config.approach_samples {
            let fraction = sample as f64 / self.config.approach_samples as f64;
            let distance = fraction * self.config.approach_length_m;
            let sample_east = east - approach_unit[0] * distance;
            let sample_north = north - approach_unit[1] * distance;
            let planned_altitude = center_ground + fraction * self.config.approach_height_m;
            if !geofence.contains([sample_east, sample_north, planned_altitude]) {
                rejections.push(LandingZoneRejection::ApproachOutsideGeofence);
                break;
            }
            match terrain.elevation_m(sample_east, sample_north) {
                Some(ground) if ground <= planned_altitude => {}
                Some(_) => {
                    rejections.push(LandingZoneRejection::ApproachTerrainConflict);
                    break;
                }
                None => {
                    rejections.push(LandingZoneRejection::TerrainUnavailable);
                    break;
                }
            }
        }

        let crosswind_mps = (wind_velocity_mps[0] * -approach_unit[1]
            + wind_velocity_mps[1] * approach_unit[0])
            .abs();
        if crosswind_mps > self.config.maximum_crosswind_mps {
            rejections.push(LandingZoneRejection::ExcessiveCrosswind);
        }
        rejections.sort_by_key(|reason| *reason as u8);
        rejections.dedup();

        let horizontal_distance_m = horizontal_distance(vehicle_position_m, touchdown);
        let slope_score = 1.0 - (maximum_slope_deg / self.config.maximum_slope_deg).clamp(0.0, 1.0);
        let roughness_score = 1.0 - (roughness_m / self.config.maximum_roughness_m).clamp(0.0, 1.0);
        let crosswind_score =
            1.0 - (crosswind_mps / self.config.maximum_crosswind_mps).clamp(0.0, 1.0);
        let distance_score =
            1.0 - (horizontal_distance_m / self.config.distance_scale_m).clamp(0.0, 1.0);
        let margin_score =
            (geofence_margin_m / (4.0 * self.config.minimum_geofence_margin_m)).clamp(0.0, 1.0);
        let viable = rejections.is_empty();
        let score = if viable {
            (0.30 * slope_score
                + 0.20 * roughness_score
                + 0.20 * crosswind_score
                + 0.20 * distance_score
                + 0.10 * margin_score)
                .clamp(0.0, 1.0)
        } else {
            0.0
        };

        Ok(LandingZoneAssessment {
            candidate_id: candidate.candidate_id.clone(),
            viable,
            score,
            ground_elevation_m: center_ground,
            maximum_slope_deg,
            roughness_m,
            geofence_margin_m,
            crosswind_mps,
            horizontal_distance_m,
            rejections,
        })
    }

    pub fn select_best<T: TerrainProvider>(
        &self,
        candidates: &[LandingZoneCandidate],
        vehicle_position_m: [f64; 3],
        wind_velocity_mps: [f64; 3],
        terrain: &T,
        geofence: &AxisAlignedGeofence,
    ) -> Result<LandingZoneAssessment, LandingZoneError> {
        if candidates.is_empty() {
            return Err(LandingZoneError::NoCandidates);
        }
        let mut viable = Vec::new();
        for candidate in candidates {
            let assessment = self.evaluate(
                candidate,
                vehicle_position_m,
                wind_velocity_mps,
                terrain,
                geofence,
            )?;
            if assessment.viable {
                viable.push(assessment);
            }
        }
        viable
            .into_iter()
            .max_by(|left, right| {
                left.score
                    .total_cmp(&right.score)
                    .then_with(|| right.candidate_id.cmp(&left.candidate_id))
            })
            .ok_or(LandingZoneError::NoViableCandidate)
    }
}

fn horizontal_distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    let east = a[0] - b[0];
    let north = a[1] - b[1];
    (east * east + north * north).sqrt()
}

fn horizontal_geofence_margin(geofence: &AxisAlignedGeofence, east_m: f64, north_m: f64) -> f64 {
    (east_m - geofence.min_east_m)
        .min(geofence.max_east_m - east_m)
        .min(north_m - geofence.min_north_m)
        .min(geofence.max_north_m - north_m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain_safety::{FlatTerrain, HeightGrid};

    fn fence() -> AxisAlignedGeofence {
        AxisAlignedGeofence {
            min_east_m: -100.0,
            max_east_m: 100.0,
            min_north_m: -100.0,
            max_north_m: 100.0,
            min_altitude_m: 0.0,
            max_altitude_m: 200.0,
        }
    }

    fn candidate(id: &str, east: f64, north: f64) -> LandingZoneCandidate {
        LandingZoneCandidate {
            candidate_id: id.to_string(),
            local_position_m: [east, north, 0.0],
            approach_heading_rad: 0.0,
        }
    }

    #[test]
    fn flat_centered_zone_is_viable() {
        let assessment = LandingZoneEvaluator::new()
            .evaluate(
                &candidate("flat", 0.0, 0.0),
                [50.0, 0.0, 30.0],
                [0.0; 3],
                &FlatTerrain::default(),
                &fence(),
            )
            .unwrap();
        assert!(assessment.viable);
        assert!(assessment.score > 0.8);
    }

    #[test]
    fn steep_or_rough_footprint_is_rejected() {
        let grid = HeightGrid {
            origin_east_m: -15.0,
            origin_north_m: -15.0,
            cell_size_m: 10.0,
            width: 3,
            height: 3,
            elevations_m: vec![0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
        };
        let assessment = LandingZoneEvaluator::with_config(LandingZoneConfig {
            footprint_radius_m: 10.0,
            approach_length_m: 10.0,
            approach_samples: 1,
            ..LandingZoneConfig::default()
        })
        .unwrap()
        .evaluate(
            &candidate("steep", 0.0, 0.0),
            [0.0, 0.0, 30.0],
            [0.0; 3],
            &grid,
            &fence(),
        )
        .unwrap();
        assert!(!assessment.viable);
        assert!(
            assessment
                .rejections
                .contains(&LandingZoneRejection::ExcessiveSlope)
        );
    }

    #[test]
    fn stable_tie_break_prefers_lexicographically_first_id() {
        let best = LandingZoneEvaluator::new()
            .select_best(
                &[candidate("bravo", 0.0, 0.0), candidate("alpha", 0.0, 0.0)],
                [50.0, 0.0, 30.0],
                [0.0; 3],
                &FlatTerrain::default(),
                &fence(),
            )
            .unwrap();
        assert_eq!(best.candidate_id, "alpha");
    }

    #[test]
    fn crosswind_limit_fails_closed() {
        let assessment = LandingZoneEvaluator::new()
            .evaluate(
                &candidate("windy", 0.0, 0.0),
                [0.0, 0.0, 30.0],
                [0.0, 20.0, 0.0],
                &FlatTerrain::default(),
                &fence(),
            )
            .unwrap();
        assert!(
            assessment
                .rejections
                .contains(&LandingZoneRejection::ExcessiveCrosswind)
        );
    }
}
