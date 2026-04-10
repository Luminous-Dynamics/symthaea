// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Water quality monitoring missions for commons stewardship.

use crate::navigation_estimator::AuvNavigationEstimate;
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

/// Communication mode recommended by the mission tracker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuvCommsPolicy {
    LiveAcoustic,
    DelayTolerantStoreForward,
    EmergencyRecoveryBurst,
}

impl Default for AuvCommsPolicy {
    fn default() -> Self {
        Self::LiveAcoustic
    }
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
    pub mean_navigation_sigma_m: f64,
    pub final_navigation_position_error_m: f64,
    pub comms_policy: AuvCommsPolicy,
}

/// Tracks mission progress using runtime navigation estimates.
#[derive(Debug, Clone)]
pub struct MissionTracker {
    waypoints: Vec<Waypoint>,
    waypoint_tolerance_m: f64,
    current_waypoint: usize,
    last_position_m: Option<[f64; 3]>,
    sigma_accumulator_m: f64,
    sigma_samples: usize,
    metrics: MissionMetrics,
}

impl MissionTracker {
    pub fn new(waypoints: Vec<Waypoint>, waypoint_tolerance_m: f64) -> Self {
        Self {
            metrics: MissionMetrics {
                total_waypoints: waypoints.len(),
                ..MissionMetrics::default()
            },
            waypoints,
            waypoint_tolerance_m,
            current_waypoint: 0,
            last_position_m: None,
            sigma_accumulator_m: 0.0,
            sigma_samples: 0,
        }
    }

    pub fn from_mission(mission: &AuvMission) -> Self {
        let waypoints = match mission {
            AuvMission::DepthTransect {
                start_depth,
                end_depth,
                sample_interval_m,
            } => depth_transect_waypoints(*start_depth, *end_depth, *sample_interval_m, 0.0, 0.0),
            AuvMission::GridPatrol {
                center,
                width,
                height,
                depth,
                lane_spacing,
            } => grid_patrol_waypoints(*center, *width, *height, *depth, *lane_spacing),
            AuvMission::StationKeep {
                position,
                duration_seconds: _,
            } => vec![Waypoint {
                position: *position,
                sample_at_waypoint: false,
            }],
            AuvMission::ContaminationTracking { .. } => vec![],
        };

        Self::new(waypoints, 3.0)
    }

    pub fn update(
        &mut self,
        timestamp_s: f64,
        estimate: &AuvNavigationEstimate,
        true_position_m: Option<[f64; 3]>,
        samples_collected_delta: usize,
        anomalies_detected_delta: usize,
        blackout: bool,
    ) {
        if let Some(last_position_m) = self.last_position_m {
            let dx = estimate.position_m[0] - last_position_m[0];
            let dy = estimate.position_m[1] - last_position_m[1];
            let dz = estimate.position_m[2] - last_position_m[2];
            self.metrics.distance_traveled += (dx * dx + dy * dy + dz * dz).sqrt();
        }
        self.last_position_m = Some(estimate.position_m);
        self.metrics.mission_time = timestamp_s;
        self.metrics.samples_collected += samples_collected_delta;
        self.metrics.anomalies_detected += anomalies_detected_delta;
        self.metrics.mean_depth_error += estimate.position_m[2].abs();
        self.sigma_accumulator_m += estimate.position_sigma_m;
        self.sigma_samples += 1;
        self.metrics.mean_navigation_sigma_m = self.sigma_accumulator_m / self.sigma_samples as f64;

        if let Some(true_position_m) = true_position_m {
            let dx = estimate.position_m[0] - true_position_m[0];
            let dy = estimate.position_m[1] - true_position_m[1];
            let dz = estimate.position_m[2] - true_position_m[2];
            self.metrics.final_navigation_position_error_m = (dx * dx + dy * dy + dz * dz).sqrt();
            self.metrics.mean_depth_error = dz.abs();
        }

        while self.current_waypoint < self.waypoints.len() {
            let waypoint = &self.waypoints[self.current_waypoint];
            let dx = estimate.position_m[0] - waypoint.position[0];
            let dy = estimate.position_m[1] - waypoint.position[1];
            let dz = estimate.position_m[2] - waypoint.position[2];
            let distance = (dx * dx + dy * dy + dz * dz).sqrt();
            if distance > self.waypoint_tolerance_m {
                break;
            }
            self.metrics.waypoints_visited += 1;
            self.current_waypoint += 1;
        }

        self.metrics.comms_policy = mission_comms_policy(estimate.position_sigma_m, blackout);
    }

    pub fn metrics(&self) -> &MissionMetrics {
        &self.metrics
    }
}

/// Select communication behavior based on navigation confidence and blackout state.
pub fn mission_comms_policy(position_sigma_m: f64, blackout: bool) -> AuvCommsPolicy {
    if position_sigma_m > 25.0 {
        AuvCommsPolicy::EmergencyRecoveryBurst
    } else if blackout || position_sigma_m > 5.0 {
        AuvCommsPolicy::DelayTolerantStoreForward
    } else {
        AuvCommsPolicy::LiveAcoustic
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::navigation_estimator::AuvNavigationEstimate;

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

    #[test]
    fn mission_tracker_advances_waypoints_and_sets_policy() {
        let mission = AuvMission::StationKeep {
            position: [5.0, 0.0, 10.0],
            duration_seconds: 30.0,
        };
        let mut tracker = MissionTracker::from_mission(&mission);
        let estimate = AuvNavigationEstimate {
            position_m: [5.5, 0.0, 10.2],
            velocity_mps: [0.0, 0.0, 0.0],
            position_sigma_m: 2.0,
            update_count: 4,
        };
        tracker.update(12.0, &estimate, Some([5.0, 0.0, 10.0]), 1, 0, false);
        let metrics = tracker.metrics();
        assert_eq!(metrics.waypoints_visited, 1);
        assert_eq!(metrics.comms_policy, AuvCommsPolicy::LiveAcoustic);
    }

    #[test]
    fn degraded_navigation_switches_to_store_forward() {
        assert_eq!(
            mission_comms_policy(8.0, true),
            AuvCommsPolicy::DelayTolerantStoreForward
        );
        assert_eq!(
            mission_comms_policy(30.0, false),
            AuvCommsPolicy::EmergencyRecoveryBurst
        );
    }
}
