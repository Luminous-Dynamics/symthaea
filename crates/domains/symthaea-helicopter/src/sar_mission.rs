// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! SAR (Search and Rescue) mission runner for consciousness-coupled helicopter.
//!
//! Defines mission types, search patterns, and success metrics for helicopter
//! SAR operations governed by the Mycelix emergency cluster.
//!
//! ## Mission Types
//!
//! - **GridSearch**: Systematic coverage of a rectangular zone at altitude
//! - **HoverOverTarget**: Precision hover for winch deployment
//! - **PointToPoint**: Navigate between waypoints (e.g., base → incident site)
//! - **ReturnToBase**: Navigate home with fuel-aware altitude management
//!
//! ## Governance Integration
//!
//! Each mission has a `MissionAuthority` with 24-hour expiry (Essay 17).
//! Telemetry is periodically submitted as `SituationReport` entries.
//! Consciousness gating controls mission authority:
//! - Φ > 0.6: full mission authority
//! - Φ 0.3–0.6: search only, no winch deploy
//! - Φ < 0.3: return to base immediately

use serde::{Deserialize, Serialize};

use crate::simulator::HelicopterPhysicsSimulator;
use crate::types::{HelicopterCommand, HelicopterState};

// ── Mission Types ────────────────────────────────────────────────────────────

/// Geographic coordinate (latitude/longitude in degrees, altitude in meters).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GeoPoint {
    pub lat: f64,
    pub lon: f64,
    pub alt: f64,
}

impl GeoPoint {
    /// Distance to another point (flat-earth approximation for SAR range, meters).
    pub fn distance_to(&self, other: &GeoPoint) -> f64 {
        // ~111,320 m per degree latitude, ~111,320 * cos(lat) per degree longitude
        let dlat = (other.lat - self.lat) * 111_320.0;
        let dlon = (other.lon - self.lon) * 111_320.0 * self.lat.to_radians().cos();
        let dalt = other.alt - self.alt;
        (dlat * dlat + dlon * dlon + dalt * dalt).sqrt()
    }

    /// Convert to local meters relative to a mission origin
    /// (equirectangular approximation — fine for SAR-scale distances).
    ///
    /// Returns `[east_m, north_m, alt_m]` matching the simulator's world
    /// frame (x = east, y = north, z = altitude). Altitude passes through
    /// unchanged: the simulator's z is already meters above ground.
    pub fn to_local_meters(&self, origin: &GeoPoint) -> [f64; 3] {
        let east = (self.lon - origin.lon) * 111_320.0 * origin.lat.to_radians().cos();
        let north = (self.lat - origin.lat) * 111_320.0;
        [east, north, self.alt]
    }
}

/// SAR mission definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarMission {
    /// Unique mission identifier.
    pub mission_id: String,
    /// Mission type.
    pub mission_type: SarMissionType,
    /// Dispatching authority (agent public key hash).
    pub dispatched_by: String,
    /// Authority expiry (Unix timestamp, 24-hour max per Essay 17).
    pub authority_expires_at: u64,
    /// Mission priority (0 = highest).
    pub priority: u8,
}

/// Mission type with parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SarMissionType {
    /// Search a rectangular grid zone.
    GridSearch {
        /// Southwest corner of search zone.
        sw_corner: GeoPoint,
        /// Northeast corner of search zone.
        ne_corner: GeoPoint,
        /// Search altitude (meters AGL).
        altitude: f64,
        /// Lane spacing (meters between parallel passes).
        lane_spacing: f64,
    },
    /// Hover over a target for winch deployment.
    HoverOverTarget {
        /// Target position.
        target: GeoPoint,
        /// Required hover altitude (meters AGL).
        hover_altitude: f64,
        /// Maximum position error for winch clearance (meters).
        max_position_error: f64,
        /// Required hover duration (seconds).
        hold_duration: f64,
    },
    /// Navigate between waypoints.
    PointToPoint {
        /// Ordered waypoints to visit.
        waypoints: Vec<GeoPoint>,
        /// Cruise altitude (meters AGL).
        cruise_altitude: f64,
        /// Target speed (m/s).
        target_speed: f64,
    },
    /// Return to base.
    ReturnToBase {
        /// Base location.
        base: GeoPoint,
        /// Remaining fuel (fraction 0.0–1.0).
        fuel_remaining: f64,
    },
}

// ── Search Patterns ──────────────────────────────────────────────────────────

/// Generate waypoints for a grid search (lawn-mower pattern).
///
/// Produces a series of waypoints that systematically cover the rectangular
/// zone defined by `sw` (southwest corner) and `ne` (northeast corner).
pub fn grid_search_waypoints(
    sw: &GeoPoint,
    ne: &GeoPoint,
    altitude: f64,
    lane_spacing: f64,
) -> Vec<GeoPoint> {
    let mut waypoints = Vec::new();
    let lon_step = lane_spacing / (111_320.0 * sw.lat.to_radians().cos());
    let mut lon = sw.lon;
    let mut going_north = true;

    while lon <= ne.lon {
        if going_north {
            waypoints.push(GeoPoint {
                lat: sw.lat,
                lon,
                alt: altitude,
            });
            waypoints.push(GeoPoint {
                lat: ne.lat,
                lon,
                alt: altitude,
            });
        } else {
            waypoints.push(GeoPoint {
                lat: ne.lat,
                lon,
                alt: altitude,
            });
            waypoints.push(GeoPoint {
                lat: sw.lat,
                lon,
                alt: altitude,
            });
        }
        going_north = !going_north;
        lon += lon_step;
    }

    waypoints
}

/// Generate waypoints for an expanding square search pattern.
///
/// Starts at `center` and spirals outward in expanding squares.
pub fn expanding_square_waypoints(
    center: &GeoPoint,
    altitude: f64,
    initial_leg: f64,
    expansion: f64,
    num_legs: usize,
) -> Vec<GeoPoint> {
    let mut waypoints = vec![GeoPoint {
        lat: center.lat,
        lon: center.lon,
        alt: altitude,
    }];
    let lat_per_m = 1.0 / 111_320.0;
    let lon_per_m = 1.0 / (111_320.0 * center.lat.to_radians().cos());

    let mut current = *center;
    let mut leg_length = initial_leg;
    // Directions: N, E, S, W (repeat with expanding legs)
    let directions: [(f64, f64); 4] = [
        (lat_per_m, 0.0),  // North
        (0.0, lon_per_m),  // East
        (-lat_per_m, 0.0), // South
        (0.0, -lon_per_m), // West
    ];

    for i in 0..num_legs {
        let dir = directions[i % 4];
        current.lat += dir.0 * leg_length;
        current.lon += dir.1 * leg_length;
        current.alt = altitude;
        waypoints.push(current);

        // Expand every 2 legs (after completing N+E or S+W)
        if i % 2 == 1 {
            leg_length += expansion;
        }
    }

    waypoints
}

/// Route-validation failure before mission execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteValidationError {
    EmptyRoute,
    NonFiniteWaypoint,
    TerrainUnavailable,
    BelowMinimumClearance,
    OutsideGeofence,
}

/// Validate every waypoint against local terrain and the authorized flight
/// volume before control authority is granted.
pub fn validate_waypoint_route<T: crate::terrain_safety::TerrainProvider>(
    origin: &GeoPoint,
    waypoints: &[GeoPoint],
    terrain: &T,
    geofence: &crate::terrain_safety::AxisAlignedGeofence,
    minimum_clearance_m: f64,
) -> Result<(), RouteValidationError> {
    if waypoints.is_empty() {
        return Err(RouteValidationError::EmptyRoute);
    }
    if !minimum_clearance_m.is_finite() || minimum_clearance_m < 0.0 {
        return Err(RouteValidationError::NonFiniteWaypoint);
    }
    for waypoint in waypoints {
        if !waypoint.lat.is_finite() || !waypoint.lon.is_finite() || !waypoint.alt.is_finite() {
            return Err(RouteValidationError::NonFiniteWaypoint);
        }
        let local = waypoint.to_local_meters(origin);
        if !geofence.contains(local) {
            return Err(RouteValidationError::OutsideGeofence);
        }
        let ground = terrain
            .elevation_m(local[0], local[1])
            .ok_or(RouteValidationError::TerrainUnavailable)?;
        if local[2] - ground < minimum_clearance_m {
            return Err(RouteValidationError::BelowMinimumClearance);
        }
    }
    Ok(())
}

// ── Mission Execution ────────────────────────────────────────────────────────

/// Mission execution state.
#[derive(Debug, Clone)]
pub struct MissionState {
    /// Current waypoint index.
    pub current_waypoint: usize,
    /// Total waypoints in mission.
    pub total_waypoints: usize,
    /// Distance to current target (meters).
    pub distance_to_target: f64,
    /// Mission elapsed time (seconds).
    pub elapsed_seconds: f64,
    /// Whether mission is complete.
    pub complete: bool,
    /// Hover hold time accumulated (for HoverOverTarget).
    pub hover_hold_time: f64,
    /// Number of simulation steps.
    pub steps: usize,
}

/// Mission metrics collected during execution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MissionMetrics {
    /// Total flight time (seconds).
    pub flight_time: f64,
    /// Total distance covered (meters).
    pub distance_covered: f64,
    /// Mean altitude error from target (meters).
    pub mean_altitude_error: f64,
    /// Mean position error from waypoint (meters).
    pub mean_position_error: f64,
    /// Mean control effort (0.0–1.0).
    pub mean_control_effort: f32,
    /// Fraction of waypoints reached (within tolerance).
    pub waypoint_completion: f64,
    /// Whether mission succeeded overall.
    pub success: bool,
    /// Number of steps executed.
    pub steps: usize,
}

/// Run a grid search mission on the given simulator.
///
/// Returns mission metrics. The simulator must be pre-initialized at the
/// mission start position; `origin` is the geographic point corresponding
/// to the simulator's local frame origin `[0, 0, z]` — all waypoints are
/// converted to local meters relative to it.
///
/// (Before Tier 2.5 this multiplied *absolute* lat/lon by 111,320 and
/// subtracted local meters, producing ~10⁷ m position errors for any
/// non-zero coordinates.)
pub fn run_grid_search<S: HelicopterPhysicsSimulator>(
    sim: &mut S,
    controller: &mut crate::controller::HelicopterController,
    encoder: &mut crate::encoder::HelicopterHdcEncoder,
    origin: &GeoPoint,
    waypoints: &[GeoPoint],
    dt: f64,
    max_steps: usize,
    waypoint_tolerance: f64,
) -> MissionMetrics {
    let mut metrics = MissionMetrics::default();
    let mut current_wp = 0;
    let mut altitude_error_sum = 0.0;
    let mut position_error_sum = 0.0;
    let mut effort_sum = 0.0f32;
    let mut prev_pos = sim.state().position;

    for step in 0..max_steps {
        if current_wp >= waypoints.len() {
            metrics.success = true;
            break;
        }

        let target = &waypoints[current_wp];
        let state = sim.state();

        // Geo → local: waypoint in meters relative to the mission origin,
        // compared against the simulator's local position.
        let local = target.to_local_meters(origin);
        let dx = local[0] - state.position[0];
        let dy = local[1] - state.position[1];
        let dz = local[2] - state.position[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();

        altitude_error_sum += dz.abs();
        position_error_sum += dist;

        // Check waypoint reached
        if dist < waypoint_tolerance {
            current_wp += 1;
        }

        // Explicit target-conditioned guidance. The learned controller is
        // retained as a bounded residual rather than being asked to infer an
        // unseen waypoint from vehicle state alone.
        let reference = crate::guidance::FlightReference::hold(local, 0.0);
        let guidance = crate::guidance::position_hold_command(
            sim.state(),
            &reference,
            &crate::guidance::GuidanceConfig::default(),
        );
        let hv = encoder.encode_with_dt(sim.state(), dt);
        let learned = controller.forward(&hv, dt as f32);
        let cmd = guidance.blend(learned, 0.10);
        effort_sum += cmd.control_effort();

        sim.step(&cmd, dt);

        // Track distance after integration.
        let pos = sim.state().position;
        let step_dist = ((pos[0] - prev_pos[0]).powi(2)
            + (pos[1] - prev_pos[1]).powi(2)
            + (pos[2] - prev_pos[2]).powi(2))
        .sqrt();
        metrics.distance_covered += step_dist;
        prev_pos = pos;
        metrics.steps = step + 1;
    }

    let n = metrics.steps.max(1) as f64;
    metrics.flight_time = metrics.steps as f64 * dt;
    metrics.mean_altitude_error = altitude_error_sum / n;
    metrics.mean_position_error = position_error_sum / n;
    metrics.mean_control_effort = effort_sum / n as f32;
    metrics.waypoint_completion = current_wp as f64 / waypoints.len().max(1) as f64;

    metrics
}

// ── Telemetry ────────────────────────────────────────────────────────────────

/// Telemetry report for Mycelix SituationReport submission.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarTelemetryReport {
    /// Mission ID.
    pub mission_id: String,
    /// Current position.
    pub position: [f64; 3],
    /// Current altitude (meters).
    pub altitude: f64,
    /// Battery/fuel level (0.0–1.0).
    pub fuel_level: f64,
    /// Current consciousness level (Phi).
    pub consciousness_level: f64,
    /// Safety level (Green/Yellow/Orange/Red).
    pub safety_level: String,
    /// Mission progress (0.0–1.0).
    pub mission_progress: f64,
    /// Waypoints completed.
    pub waypoints_completed: usize,
    /// Waypoints remaining.
    pub waypoints_remaining: usize,
    /// Timestamp (seconds since mission start).
    pub elapsed_seconds: f64,
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geo_point_distance() {
        let a = GeoPoint {
            lat: 0.0,
            lon: 0.0,
            alt: 0.0,
        };
        let b = GeoPoint {
            lat: 0.0,
            lon: 0.0,
            alt: 100.0,
        };
        assert!((a.distance_to(&b) - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_geo_point_distance_horizontal() {
        let a = GeoPoint {
            lat: 30.0,
            lon: -97.0,
            alt: 0.0,
        };
        let b = GeoPoint {
            lat: 30.001,
            lon: -97.0,
            alt: 0.0,
        };
        let dist = a.distance_to(&b);
        // ~111m per 0.001 degree latitude
        assert!(dist > 100.0 && dist < 120.0, "dist={dist}");
    }

    #[test]
    fn test_grid_search_waypoints() {
        let sw = GeoPoint {
            lat: 30.0,
            lon: -97.0,
            alt: 0.0,
        };
        let ne = GeoPoint {
            lat: 30.005,
            lon: -96.995,
            alt: 0.0,
        };
        let waypoints = grid_search_waypoints(&sw, &ne, 50.0, 100.0);
        assert!(waypoints.len() >= 4, "Should have at least 2 lanes");
        assert!(waypoints.iter().all(|wp| (wp.alt - 50.0).abs() < 0.01));
    }

    #[test]
    fn test_grid_search_alternating_direction() {
        let sw = GeoPoint {
            lat: 0.0,
            lon: 0.0,
            alt: 0.0,
        };
        let ne = GeoPoint {
            lat: 1.0,
            lon: 1.0,
            alt: 0.0,
        };
        let waypoints = grid_search_waypoints(&sw, &ne, 50.0, 50_000.0);
        // First lane goes S→N, second goes N→S (lawn-mower)
        if waypoints.len() >= 4 {
            assert!(
                waypoints[0].lat < waypoints[1].lat,
                "First lane should go north"
            );
            assert!(
                waypoints[2].lat > waypoints[3].lat,
                "Second lane should go south"
            );
        }
    }

    #[test]
    fn test_expanding_square_waypoints() {
        let center = GeoPoint {
            lat: 30.0,
            lon: -97.0,
            alt: 0.0,
        };
        let waypoints = expanding_square_waypoints(&center, 50.0, 100.0, 50.0, 8);
        assert_eq!(waypoints.len(), 9); // center + 8 legs
        assert!(waypoints.iter().all(|wp| (wp.alt - 50.0).abs() < 0.01));
    }

    #[test]
    fn test_expanding_square_expands() {
        let center = GeoPoint {
            lat: 30.0,
            lon: -97.0,
            alt: 0.0,
        };
        let waypoints = expanding_square_waypoints(&center, 50.0, 100.0, 50.0, 8);
        // Later waypoints should be farther from center
        let dist_first = center.distance_to(&waypoints[1]);
        let dist_last = center.distance_to(&waypoints[waypoints.len() - 1]);
        assert!(
            dist_last > dist_first,
            "Square should expand: first={dist_first}, last={dist_last}"
        );
    }

    #[test]
    fn test_mission_metrics_default() {
        let m = MissionMetrics::default();
        assert!(!m.success);
        assert_eq!(m.steps, 0);
        assert_eq!(m.distance_covered, 0.0);
    }

    #[test]
    fn route_validation_rejects_low_or_outside_waypoints() {
        use crate::terrain_safety::{AxisAlignedGeofence, FlatTerrain};
        let origin = GeoPoint {
            lat: 0.0,
            lon: 0.0,
            alt: 0.0,
        };
        let fence = AxisAlignedGeofence {
            min_east_m: -100.0,
            max_east_m: 100.0,
            min_north_m: -100.0,
            max_north_m: 100.0,
            min_altitude_m: 0.0,
            max_altitude_m: 100.0,
        };
        let low = [GeoPoint {
            lat: 0.0,
            lon: 0.0,
            alt: 5.0,
        }];
        assert_eq!(
            validate_waypoint_route(&origin, &low, &FlatTerrain::default(), &fence, 15.0),
            Err(RouteValidationError::BelowMinimumClearance)
        );
        let outside = [GeoPoint {
            lat: 0.0,
            lon: 1_000.0 / 111_320.0,
            alt: 20.0,
        }];
        assert_eq!(
            validate_waypoint_route(&origin, &outside, &FlatTerrain::default(), &fence, 15.0),
            Err(RouteValidationError::OutsideGeofence)
        );
    }

    #[test]
    fn test_telemetry_report_serialization() {
        let report = SarTelemetryReport {
            mission_id: "SAR-001".to_string(),
            position: [0.0, 0.0, 20.0],
            altitude: 20.0,
            fuel_level: 0.8,
            consciousness_level: 0.7,
            safety_level: "Green".to_string(),
            mission_progress: 0.5,
            waypoints_completed: 3,
            waypoints_remaining: 3,
            elapsed_seconds: 120.0,
        };
        let json = serde_json::to_string(&report).unwrap();
        let restored: SarTelemetryReport = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.mission_id, "SAR-001");
        assert_eq!(restored.waypoints_completed, 3);
    }

    #[test]
    fn test_run_grid_search_basic() {
        use crate::controller::HelicopterController;
        use crate::encoder::HelicopterHdcEncoder;
        use crate::simulator::SimpleHelicopterSimulator;
        use crate::types::HelicopterConfig;

        let config = HelicopterConfig::default();
        let genesis = symthaea_core::genesis::GenesisSeed::from_phrase(&config.genesis_phrase);
        let mut sim = SimpleHelicopterSimulator::new();
        let mut ctrl = HelicopterController::new(&genesis, &config);
        let mut enc = HelicopterHdcEncoder::new(&genesis, 32);

        // Simple 2-waypoint mission at hover altitude
        let origin = GeoPoint {
            lat: 0.0,
            lon: 0.0,
            alt: 0.0,
        };
        let waypoints = vec![
            GeoPoint {
                lat: 0.0,
                lon: 0.0,
                alt: 20.0,
            },
            GeoPoint {
                lat: 0.0,
                lon: 0.0,
                alt: 30.0,
            },
        ];

        let metrics = run_grid_search(
            &mut sim,
            &mut ctrl,
            &mut enc,
            &origin,
            &waypoints,
            1.0 / 300.0,
            300,    // 1 second
            1000.0, // Very loose tolerance for testing
        );

        assert!(metrics.steps > 0);
        assert!(metrics.flight_time > 0.0);
        assert!(metrics.mean_control_effort >= 0.0);
    }

    #[test]
    fn test_to_local_meters_100m_north() {
        let origin = GeoPoint {
            lat: 30.0,
            lon: -97.0,
            alt: 0.0,
        };
        let north_100m = GeoPoint {
            lat: 30.0 + 100.0 / 111_320.0,
            lon: -97.0,
            alt: 20.0,
        };
        let local = north_100m.to_local_meters(&origin);
        assert!(
            local[0].abs() < 0.1,
            "east offset should be ~0: {}",
            local[0]
        );
        assert!(
            (local[1] - 100.0).abs() < 0.1,
            "north offset should be ~100m: {}",
            local[1]
        );
        assert!((local[2] - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_grid_search_position_error_is_local_not_geographic() {
        // Regression for the Tier 2.5 geo→local bug: a waypoint 100m north
        // of the mission origin must produce a ~100m error vector, not the
        // ~1e7 m garbage that absolute-degrees × 111,320 used to produce.
        use crate::controller::HelicopterController;
        use crate::encoder::HelicopterHdcEncoder;
        use crate::simulator::SimpleHelicopterSimulator;
        use crate::types::HelicopterConfig;

        let config = HelicopterConfig::default();
        let genesis = symthaea_core::genesis::GenesisSeed::from_phrase(&config.genesis_phrase);
        let mut sim = SimpleHelicopterSimulator::new(); // Hover at 20m, local [0,0,20]
        let mut ctrl = HelicopterController::new(&genesis, &config);
        let mut enc = HelicopterHdcEncoder::new(&genesis, 32);

        let origin = GeoPoint {
            lat: 30.0,
            lon: -97.0,
            alt: 0.0,
        };
        let waypoints = vec![GeoPoint {
            lat: 30.0 + 100.0 / 111_320.0, // 100m north of origin
            lon: -97.0,
            alt: 20.0, // Same altitude as the hovering sim
        }];

        let metrics = run_grid_search(
            &mut sim,
            &mut ctrl,
            &mut enc,
            &origin,
            &waypoints,
            1.0 / 300.0,
            30, // Short: position barely changes → error stays near initial
            1.0,
        );

        assert!(
            metrics.mean_position_error > 50.0 && metrics.mean_position_error < 200.0,
            "100m-north waypoint must give ~100m error, got {} m",
            metrics.mean_position_error
        );
    }
}
