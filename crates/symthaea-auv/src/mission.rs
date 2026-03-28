// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Water quality monitoring missions for commons stewardship.

use crate::types::ChemicalReadings;
use serde::{Deserialize, Serialize};

/// Mission types for AUV water stewardship.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuvMission {
    /// Patrol a depth transect, sampling water quality at intervals.
    DepthTransect {
        start_depth: f64,
        end_depth: f64,
        sample_interval_m: f64,
    },
    /// Patrol a horizontal grid at fixed depth.
    GridPatrol {
        center: [f64; 2],
        width: f64,
        height: f64,
        depth: f64,
        lane_spacing: f64,
    },
    /// Track a contamination plume to its source.
    ContaminationTracking {
        initial_reading: ChemicalReadings,
        gradient_follow: bool,
    },
    /// Station-keeping: hold position and monitor continuously.
    StationKeep {
        position: [f64; 3],
        duration_seconds: f64,
    },
}

/// Waypoint for mission navigation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Waypoint {
    pub position: [f64; 3],
    pub sample_at_waypoint: bool,
}

/// Generate depth transect waypoints.
pub fn depth_transect_waypoints(
    start: f64,
    end: f64,
    interval: f64,
    x: f64,
    y: f64,
) -> Vec<Waypoint> {
    let mut waypoints = Vec::new();
    let mut depth = start;
    let step = if end > start { interval } else { -interval };
    while (step > 0.0 && depth <= end) || (step < 0.0 && depth >= end) {
        waypoints.push(Waypoint {
            position: [x, y, depth],
            sample_at_waypoint: true,
        });
        depth += step;
    }
    waypoints
}

/// Generate horizontal grid waypoints (lawn-mower at fixed depth).
pub fn grid_patrol_waypoints(
    center: [f64; 2],
    width: f64,
    height: f64,
    depth: f64,
    lane_spacing: f64,
) -> Vec<Waypoint> {
    let mut waypoints = Vec::new();
    let x_start = center[0] - width / 2.0;
    let x_end = center[0] + width / 2.0;
    let y_start = center[1] - height / 2.0;
    let y_end = center[1] + height / 2.0;

    let mut y = y_start;
    let mut going_right = true;
    while y <= y_end {
        if going_right {
            waypoints.push(Waypoint {
                position: [x_start, y, depth],
                sample_at_waypoint: true,
            });
            waypoints.push(Waypoint {
                position: [x_end, y, depth],
                sample_at_waypoint: true,
            });
        } else {
            waypoints.push(Waypoint {
                position: [x_end, y, depth],
                sample_at_waypoint: true,
            });
            waypoints.push(Waypoint {
                position: [x_start, y, depth],
                sample_at_waypoint: true,
            });
        }
        going_right = !going_right;
        y += lane_spacing;
    }
    waypoints
}

/// Mission metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MissionMetrics {
    pub waypoints_visited: usize,
    pub total_waypoints: usize,
    pub samples_collected: usize,
    pub anomalies_detected: usize,
    pub distance_traveled: f64,
    pub mission_time: f64,
    pub mean_depth_error: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_depth_transect_waypoints() {
        let wps = depth_transect_waypoints(0.0, 50.0, 10.0, 0.0, 0.0);
        assert_eq!(wps.len(), 6); // 0, 10, 20, 30, 40, 50
        assert!(wps.iter().all(|w| w.sample_at_waypoint));
        assert!((wps[0].position[2] - 0.0).abs() < 1e-10);
        assert!((wps[5].position[2] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_grid_patrol_waypoints() {
        let wps = grid_patrol_waypoints([0.0, 0.0], 100.0, 50.0, 10.0, 25.0);
        assert!(wps.len() >= 4); // At least 2 lanes × 2 ends
        assert!(wps.iter().all(|w| (w.position[2] - 10.0).abs() < 1e-10));
    }

    #[test]
    fn test_grid_alternating_direction() {
        let wps = grid_patrol_waypoints([0.0, 0.0], 100.0, 100.0, 10.0, 50.0);
        if wps.len() >= 4 {
            // First lane goes right (x increases)
            assert!(wps[0].position[0] < wps[1].position[0]);
            // Second lane goes left (x decreases)
            assert!(wps[2].position[0] > wps[3].position[0]);
        }
    }
}
