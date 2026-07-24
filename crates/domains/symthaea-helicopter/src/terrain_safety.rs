// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Terrain-clearance and geofence safety kernel.
//!
//! Mission guidance may request a waypoint, but an independent kernel decides
//! whether the current and projected trajectory remains inside the authorized
//! volume with sufficient terrain clearance. Missing terrain data fails closed.

use serde::{Deserialize, Serialize};

use crate::types::{HelicopterCommand, HelicopterState};

/// Terrain elevation source in the simulator's local world frame.
pub trait TerrainProvider {
    /// Ground elevation in meters at local east/north coordinates.
    fn elevation_m(&self, east_m: f64, north_m: f64) -> Option<f64>;
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FlatTerrain {
    pub elevation_m: f64,
}

impl Default for FlatTerrain {
    fn default() -> Self {
        Self { elevation_m: 0.0 }
    }
}

impl TerrainProvider for FlatTerrain {
    fn elevation_m(&self, _east_m: f64, _north_m: f64) -> Option<f64> {
        self.elevation_m.is_finite().then_some(self.elevation_m)
    }
}

/// Bounded height grid using nearest-cell sampling. Queries outside the grid
/// return `None` rather than assuming sea level.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeightGrid {
    pub origin_east_m: f64,
    pub origin_north_m: f64,
    pub cell_size_m: f64,
    pub width: usize,
    pub height: usize,
    pub elevations_m: Vec<f64>,
}

impl HeightGrid {
    pub fn validate(&self) -> bool {
        self.origin_east_m.is_finite()
            && self.origin_north_m.is_finite()
            && self.cell_size_m.is_finite()
            && self.cell_size_m > 0.0
            && self.width > 0
            && self.height > 0
            && self.elevations_m.len() == self.width.saturating_mul(self.height)
            && self.elevations_m.iter().all(|value| value.is_finite())
    }
}

impl TerrainProvider for HeightGrid {
    fn elevation_m(&self, east_m: f64, north_m: f64) -> Option<f64> {
        if !self.validate() || !east_m.is_finite() || !north_m.is_finite() {
            return None;
        }
        let x = ((east_m - self.origin_east_m) / self.cell_size_m).floor();
        let y = ((north_m - self.origin_north_m) / self.cell_size_m).floor();
        if x < 0.0 || y < 0.0 || x >= self.width as f64 || y >= self.height as f64 {
            return None;
        }
        self.elevations_m
            .get(y as usize * self.width + x as usize)
            .copied()
    }
}

/// Authorized local flight volume.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AxisAlignedGeofence {
    pub min_east_m: f64,
    pub max_east_m: f64,
    pub min_north_m: f64,
    pub max_north_m: f64,
    pub min_altitude_m: f64,
    pub max_altitude_m: f64,
}

impl AxisAlignedGeofence {
    pub fn validate(&self) -> bool {
        [
            self.min_east_m,
            self.max_east_m,
            self.min_north_m,
            self.max_north_m,
            self.min_altitude_m,
            self.max_altitude_m,
        ]
        .iter()
        .all(|value| value.is_finite())
            && self.min_east_m <= self.max_east_m
            && self.min_north_m <= self.max_north_m
            && self.min_altitude_m <= self.max_altitude_m
    }

    pub fn contains(&self, position: [f64; 3]) -> bool {
        self.validate()
            && position[0] >= self.min_east_m
            && position[0] <= self.max_east_m
            && position[1] >= self.min_north_m
            && position[1] <= self.max_north_m
            && position[2] >= self.min_altitude_m
            && position[2] <= self.max_altitude_m
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TerrainSafetyConfig {
    pub minimum_clearance_m: f64,
    pub lookahead_s: f64,
    pub emergency_climb_collective: f32,
    pub emergency_climb_thrust: f32,
}

impl Default for TerrainSafetyConfig {
    fn default() -> Self {
        Self {
            minimum_clearance_m: 15.0,
            lookahead_s: 3.0,
            emergency_climb_collective: 0.48,
            emergency_climb_thrust: 0.68,
        }
    }
}

impl TerrainSafetyConfig {
    pub fn validate(&self) -> bool {
        self.minimum_clearance_m.is_finite()
            && self.minimum_clearance_m >= 0.0
            && self.lookahead_s.is_finite()
            && self.lookahead_s >= 0.0
            && self.emergency_climb_collective.is_finite()
            && (0.0..=1.0).contains(&self.emergency_climb_collective)
            && self.emergency_climb_thrust.is_finite()
            && (0.0..=1.0).contains(&self.emergency_climb_thrust)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TerrainSafetyReason {
    Safe,
    InvalidConfiguration,
    TerrainUnavailable,
    OutsideGeofence,
    InsufficientCurrentClearance,
    ProjectedTerrainConflict,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TerrainSafetyAssessment {
    pub safe: bool,
    pub reason: TerrainSafetyReason,
    pub current_clearance_m: f64,
    pub projected_clearance_m: f64,
    pub projected_position: [f64; 3],
}

#[derive(Debug, Clone)]
pub struct TerrainSafetyKernel {
    config: TerrainSafetyConfig,
}

impl Default for TerrainSafetyKernel {
    fn default() -> Self {
        Self {
            config: TerrainSafetyConfig::default(),
        }
    }
}

impl TerrainSafetyKernel {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_config(config: TerrainSafetyConfig) -> Option<Self> {
        config.validate().then_some(Self { config })
    }

    pub fn config(&self) -> TerrainSafetyConfig {
        self.config
    }

    pub fn assess<T: TerrainProvider>(
        &self,
        state: &HelicopterState,
        terrain: &T,
        geofence: &AxisAlignedGeofence,
    ) -> TerrainSafetyAssessment {
        if !self.config.validate() || !geofence.validate() || !state.is_finite() {
            return TerrainSafetyAssessment {
                safe: false,
                reason: TerrainSafetyReason::InvalidConfiguration,
                current_clearance_m: f64::NAN,
                projected_clearance_m: f64::NAN,
                projected_position: state.position,
            };
        }

        let projected_position = [
            state.position[0] + state.linear_velocity[0] * self.config.lookahead_s,
            state.position[1] + state.linear_velocity[1] * self.config.lookahead_s,
            state.position[2] + state.linear_velocity[2] * self.config.lookahead_s,
        ];
        let Some(current_ground) = terrain.elevation_m(state.position[0], state.position[1]) else {
            return TerrainSafetyAssessment {
                safe: false,
                reason: TerrainSafetyReason::TerrainUnavailable,
                current_clearance_m: f64::NAN,
                projected_clearance_m: f64::NAN,
                projected_position,
            };
        };
        let Some(projected_ground) =
            terrain.elevation_m(projected_position[0], projected_position[1])
        else {
            return TerrainSafetyAssessment {
                safe: false,
                reason: TerrainSafetyReason::TerrainUnavailable,
                current_clearance_m: state.position[2] - current_ground,
                projected_clearance_m: f64::NAN,
                projected_position,
            };
        };

        let current_clearance_m = state.position[2] - current_ground;
        let projected_clearance_m = projected_position[2] - projected_ground;
        let reason = if !geofence.contains(state.position) || !geofence.contains(projected_position)
        {
            TerrainSafetyReason::OutsideGeofence
        } else if current_clearance_m < self.config.minimum_clearance_m {
            TerrainSafetyReason::InsufficientCurrentClearance
        } else if projected_clearance_m < self.config.minimum_clearance_m {
            TerrainSafetyReason::ProjectedTerrainConflict
        } else {
            TerrainSafetyReason::Safe
        };

        TerrainSafetyAssessment {
            safe: reason == TerrainSafetyReason::Safe,
            reason,
            current_clearance_m,
            projected_clearance_m,
            projected_position,
        }
    }

    /// Apply an independent escape command. Terrain conflicts prioritize climb;
    /// geofence conflicts brake horizontal motion while preserving lift.
    pub fn enforce(
        &self,
        state: &HelicopterState,
        requested: HelicopterCommand,
        assessment: TerrainSafetyAssessment,
    ) -> HelicopterCommand {
        if assessment.safe {
            return requested.clamped();
        }
        let mut command = requested;
        match assessment.reason {
            TerrainSafetyReason::OutsideGeofence => {
                command.cyclic_lon = (-0.10 * state.linear_velocity[0]) as f32;
                command.cyclic_lat = (0.10 * state.linear_velocity[1]) as f32;
                command.collective = command.collective.max(0.30);
                command.thrust = command.thrust.max(0.60);
            }
            TerrainSafetyReason::InsufficientCurrentClearance
            | TerrainSafetyReason::ProjectedTerrainConflict
            | TerrainSafetyReason::TerrainUnavailable
            | TerrainSafetyReason::InvalidConfiguration => {
                command.collective = command
                    .collective
                    .max(self.config.emergency_climb_collective);
                command.thrust = command.thrust.max(self.config.emergency_climb_thrust);
                command.cyclic_lon = (-0.08 * state.linear_velocity[0]) as f32;
                command.cyclic_lat = (0.08 * state.linear_velocity[1]) as f32;
            }
            TerrainSafetyReason::Safe => {}
        }
        command.clamped()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fence() -> AxisAlignedGeofence {
        AxisAlignedGeofence {
            min_east_m: -1_000.0,
            max_east_m: 1_000.0,
            min_north_m: -1_000.0,
            max_north_m: 1_000.0,
            min_altitude_m: 0.0,
            max_altitude_m: 500.0,
        }
    }

    #[test]
    fn safe_hover_passes() {
        let kernel = TerrainSafetyKernel::new();
        let state = HelicopterState::hover(20.0);
        let assessment = kernel.assess(&state, &FlatTerrain::default(), &fence());
        assert!(assessment.safe);
        assert_eq!(assessment.reason, TerrainSafetyReason::Safe);
    }

    #[test]
    fn descending_trajectory_detects_projected_conflict() {
        let kernel = TerrainSafetyKernel::new();
        let mut state = HelicopterState::hover(20.0);
        state.linear_velocity[2] = -3.0;
        let assessment = kernel.assess(&state, &FlatTerrain::default(), &fence());
        assert!(!assessment.safe);
        assert_eq!(
            assessment.reason,
            TerrainSafetyReason::ProjectedTerrainConflict
        );
        let command = kernel.enforce(&state, HelicopterCommand::zero(), assessment);
        assert!(command.collective >= 0.48);
        assert!(command.thrust >= 0.68);
    }

    #[test]
    fn unknown_height_grid_area_fails_closed() {
        let grid = HeightGrid {
            origin_east_m: 0.0,
            origin_north_m: 0.0,
            cell_size_m: 10.0,
            width: 1,
            height: 1,
            elevations_m: vec![0.0],
        };
        let mut state = HelicopterState::hover(20.0);
        state.position[0] = 100.0;
        let assessment = TerrainSafetyKernel::new().assess(&state, &grid, &fence());
        assert_eq!(assessment.reason, TerrainSafetyReason::TerrainUnavailable);
    }

    #[test]
    fn projected_geofence_exit_brakes_translation() {
        let kernel = TerrainSafetyKernel::new();
        let mut state = HelicopterState::hover(20.0);
        state.position[0] = 995.0;
        state.linear_velocity[0] = 10.0;
        let assessment = kernel.assess(&state, &FlatTerrain::default(), &fence());
        assert_eq!(assessment.reason, TerrainSafetyReason::OutsideGeofence);
        let command = kernel.enforce(&state, HelicopterCommand::hover(), assessment);
        assert!(command.cyclic_lon < 0.0);
        assert!(command.thrust >= 0.60);
    }
}
