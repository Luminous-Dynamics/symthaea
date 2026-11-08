// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Versioned classical road-tracking baselines for controlled vehicle evaluation.
//!
//! These baselines are simulation research comparators. They are not certified
//! production controllers and their profile digests are not binary attestations.

use serde::{Deserialize, Serialize};

use crate::shadow_evaluation::pd_road_baseline_digest;
use crate::road::Road;
use crate::types::{
    ChassisParams, VehicleCommand, VehicleState, pd_cruise_baseline_with_road,
};

pub const STANLEY_ROAD_BASELINE_POLICY_ID: &str = "symthaea.vehicle.stanley-road-baseline.v1";
pub const PURE_PURSUIT_ROAD_BASELINE_POLICY_ID: &str = "symthaea.vehicle.pure-pursuit-road-baseline.v1";

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct StanleyRoadConfig {
    /// Cross-track correction gain in atan(k * e / (v + softening)).
    pub cross_track_gain: f64,
    /// Positive low-speed regularizer in m/s.
    pub softening_speed_mps: f64,
    /// Multiplier on atan(wheelbase * road_curvature) feed-forward.
    pub curvature_feed_forward_gain: f64,
    /// Proportional longitudinal speed gain.
    pub speed_gain: f64,
    /// Speed-error deadband before braking in m/s.
    pub brake_deadband_mps: f64,
    /// Proportional braking gain beyond the deadband.
    pub brake_gain: f64,
}

impl Default for StanleyRoadConfig {
    fn default() -> Self {
        Self {
            cross_track_gain: 1.5,
            softening_speed_mps: 2.0,
            curvature_feed_forward_gain: 1.0,
            speed_gain: 0.3,
            brake_deadband_mps: 1.0,
            brake_gain: 0.2,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct PurePursuitRoadConfig {
    /// Minimum lookahead distance in meters.
    pub minimum_lookahead_m: f64,
    /// Additional lookahead per m/s of speed (seconds).
    pub speed_lookahead_gain_s: f64,
    /// Maximum lookahead distance in meters.
    pub maximum_lookahead_m: f64,
    /// Proportional longitudinal speed gain.
    pub speed_gain: f64,
    /// Speed-error deadband before braking in m/s.
    pub brake_deadband_mps: f64,
    /// Proportional braking gain beyond the deadband.
    pub brake_gain: f64,
}

impl Default for PurePursuitRoadConfig {
    fn default() -> Self {
        Self {
            minimum_lookahead_m: 4.0,
            speed_lookahead_gain_s: 0.8,
            maximum_lookahead_m: 18.0,
            speed_gain: 0.3,
            brake_deadband_mps: 1.0,
            brake_gain: 0.2,
        }
    }
}

impl PurePursuitRoadConfig {
    pub fn is_valid(&self) -> bool {
        [
            self.minimum_lookahead_m,
            self.speed_lookahead_gain_s,
            self.maximum_lookahead_m,
            self.speed_gain,
            self.brake_deadband_mps,
            self.brake_gain,
        ]
        .iter()
        .all(|value| value.is_finite())
            && self.minimum_lookahead_m > 0.0
            && self.speed_lookahead_gain_s >= 0.0
            && self.maximum_lookahead_m >= self.minimum_lookahead_m
            && self.speed_gain >= 0.0
            && self.brake_deadband_mps >= 0.0
            && self.brake_gain >= 0.0
    }

    pub fn digest(&self) -> Option<String> {
        if !self.is_valid() {
            return None;
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(PURE_PURSUIT_ROAD_BASELINE_POLICY_ID.as_bytes());
        for value in [
            self.minimum_lookahead_m,
            self.speed_lookahead_gain_s,
            self.maximum_lookahead_m,
            self.speed_gain,
            self.brake_deadband_mps,
            self.brake_gain,
        ] {
            hasher.update(&value.to_bits().to_le_bytes());
        }
        Some(hasher.finalize().to_hex().to_string())
    }
}

impl StanleyRoadConfig {
    pub fn is_valid(&self) -> bool {
        [
            self.cross_track_gain,
            self.softening_speed_mps,
            self.curvature_feed_forward_gain,
            self.speed_gain,
            self.brake_deadband_mps,
            self.brake_gain,
        ]
        .iter()
        .all(|value| value.is_finite())
            && self.cross_track_gain >= 0.0
            && self.softening_speed_mps > 0.0
            && self.curvature_feed_forward_gain >= 0.0
            && self.speed_gain >= 0.0
            && self.brake_deadband_mps >= 0.0
            && self.brake_gain >= 0.0
    }

    pub fn digest(&self) -> Option<String> {
        if !self.is_valid() {
            return None;
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(STANLEY_ROAD_BASELINE_POLICY_ID.as_bytes());
        for value in [
            self.cross_track_gain,
            self.softening_speed_mps,
            self.curvature_feed_forward_gain,
            self.speed_gain,
            self.brake_deadband_mps,
            self.brake_gain,
        ] {
            hasher.update(&value.to_bits().to_le_bytes());
        }
        Some(hasher.finalize().to_hex().to_string())
    }
}

/// Selectable classical comparator for paired simulator evaluations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum VehicleClassicalBaseline {
    /// Existing PD teacher used by the HDC-LTC imitation curriculum.
    PdRoadTeacher,
    /// Stanley-style cross-track/heading road tracker with a separate speed loop.
    StanleyRoad(StanleyRoadConfig),
    /// Pure-pursuit geometry tracker with speed-scaled path lookahead.
    PurePursuitRoad(PurePursuitRoadConfig),
}

impl Default for VehicleClassicalBaseline {
    fn default() -> Self {
        Self::PdRoadTeacher
    }
}

impl VehicleClassicalBaseline {
    pub fn policy_id(&self) -> &'static str {
        match self {
            Self::PdRoadTeacher => crate::shadow_evaluation::PD_ROAD_BASELINE_POLICY_ID,
            Self::StanleyRoad(_) => STANLEY_ROAD_BASELINE_POLICY_ID,
            Self::PurePursuitRoad(_) => PURE_PURSUIT_ROAD_BASELINE_POLICY_ID,
        }
    }

    /// Stable parameter-profile fingerprint; not a build or binary attestation.
    pub fn controller_digest(&self) -> Option<String> {
        match self {
            Self::PdRoadTeacher => Some(pd_road_baseline_digest()),
            Self::StanleyRoad(config) => config.digest(),
            Self::PurePursuitRoad(config) => config.digest(),
        }
    }

    pub fn is_valid(&self) -> bool {
        self.controller_digest().is_some()
    }

    pub fn command(
        &self,
        state: &VehicleState,
        road: &Road,
        target_speed_mps: f64,
        chassis: &ChassisParams,
    ) -> Option<VehicleCommand> {
        if road.profile_digest().is_none()
            || !target_speed_mps.is_finite()
            || !(0.0..=80.0).contains(&target_speed_mps)
            || !chassis.max_steering_angle.is_finite()
            || chassis.max_steering_angle <= 0.0
            || !chassis.wheelbase.is_finite()
            || chassis.wheelbase <= 0.0
            || !state.speed.is_finite()
            || !(0.0..80.0).contains(&state.speed)
            || !state.heading.is_finite()
            || !state.position_x.is_finite()
            || !state.position_y.is_finite()
            || !state.lateral_offset.is_finite()
            || !state.curvature_ahead.is_finite()
        {
            return None;
        }

        let projection = road.project(state.position_x, state.position_y);
        if !projection.s.is_finite() || !projection.road_heading.is_finite() {
            return None;
        }

        match self {
            Self::PdRoadTeacher => {
                Some(pd_cruise_baseline_with_road(state, target_speed_mps, projection.road_heading))
            }
            Self::StanleyRoad(config) => {
                if !config.is_valid()
                    || !state.speed.is_finite()
                    || !state.heading.is_finite()
                    || !state.lateral_offset.is_finite()
                    || !state.curvature_ahead.is_finite()
                {
                    return None;
                }

                let heading_error = wrap_angle(projection.road_heading - state.heading);
                let cross_track_correction = (config.cross_track_gain * projection.lateral_offset)
                    .atan2(state.speed.max(0.0) + config.softening_speed_mps);
                let road_curvature = road.curvature_at(projection.s + road.lookahead);
                let curvature_feed_forward = (chassis.wheelbase * road_curvature
                    * config.curvature_feed_forward_gain)
                    .atan();
                let steering_angle =
                    heading_error - cross_track_correction + curvature_feed_forward;
                let steering =
                    (steering_angle / chassis.max_steering_angle).clamp(-1.0, 1.0) as f32;

                let speed_error = target_speed_mps - state.speed;
                let throttle = if speed_error > 0.0 {
                    (speed_error * config.speed_gain).clamp(0.0, 1.0) as f32
                } else {
                    0.0
                };
                let brake = if speed_error < -config.brake_deadband_mps {
                    ((-speed_error - config.brake_deadband_mps) * config.brake_gain)
                        .clamp(0.0, 1.0) as f32
                } else {
                    0.0
                };

                Some(VehicleCommand {
                    steering,
                    throttle,
                    brake,
                })
            }
            Self::PurePursuitRoad(config) => {
                if !config.is_valid() {
                    return None;
                }
                let lookahead_m = (config.minimum_lookahead_m
                    + config.speed_lookahead_gain_s * state.speed)
                    .clamp(
                        config.minimum_lookahead_m,
                        config.maximum_lookahead_m,
                    );
                let (target_x, target_y, _) =
                    road.position_at(projection.s + lookahead_m);
                let dx = target_x - state.position_x;
                let dy = target_y - state.position_y;
                let chord_m = dx.hypot(dy);
                if !chord_m.is_finite() || chord_m < 0.05 {
                    return None;
                }
                let alpha = wrap_angle(dy.atan2(dx) - state.heading);
                let curvature_per_m = 2.0 * alpha.sin() / chord_m;
                let steering_angle = (chassis.wheelbase * curvature_per_m).atan();
                let steering =
                    (steering_angle / chassis.max_steering_angle).clamp(-1.0, 1.0) as f32;

                let speed_error = target_speed_mps - state.speed;
                let throttle = if speed_error > 0.0 {
                    (speed_error * config.speed_gain).clamp(0.0, 1.0) as f32
                } else {
                    0.0
                };
                let brake = if speed_error < -config.brake_deadband_mps {
                    ((-speed_error - config.brake_deadband_mps) * config.brake_gain)
                        .clamp(0.0, 1.0) as f32
                } else {
                    0.0
                };
                Some(VehicleCommand {
                    steering,
                    throttle,
                    brake,
                })
            }
        }
    }
}

fn wrap_angle(mut angle: f64) -> f64 {
    while angle > std::f64::consts::PI {
        angle -= 2.0 * std::f64::consts::PI;
    }
    while angle < -std::f64::consts::PI {
        angle += 2.0 * std::f64::consts::PI;
    }
    angle
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chassis() -> ChassisParams {
        ChassisParams::default()
    }

    #[test]
    fn stanley_turns_toward_center_from_positive_and_negative_offsets() {
        let policy = VehicleClassicalBaseline::StanleyRoad(StanleyRoadConfig::default());
        let mut left = VehicleState::cruising(10.0);
        left.position_y = 1.0;
        left.lateral_offset = 1.0;
        let mut right = VehicleState::cruising(10.0);
        right.position_y = -1.0;
        right.lateral_offset = -1.0;

        let road = Road::straight();
        let left_command = policy.command(&left, &road, 10.0, &chassis()).unwrap();
        let right_command = policy.command(&right, &road, 10.0, &chassis()).unwrap();

        assert!(left_command.steering < 0.0);
        assert!(right_command.steering > 0.0);
    }

    #[test]
    fn stanley_has_zero_steering_on_centered_straight_road() {
        let policy = VehicleClassicalBaseline::StanleyRoad(StanleyRoadConfig::default());
        let state = VehicleState::cruising(10.0);
        let command = policy.command(&state, &Road::straight(), 10.0, &chassis()).unwrap();
        assert!(command.steering.abs() < 1e-7);
        assert_eq!(command.throttle, 0.0);
        assert_eq!(command.brake, 0.0);
    }

    #[test]
    fn stanley_steering_includes_heading_and_curvature_feed_forward() {
        let policy = VehicleClassicalBaseline::StanleyRoad(StanleyRoadConfig::default());
        let mut state = VehicleState::cruising(10.0);
        state.heading = -0.1;
        let curved_road = Road::from_segments(&[
            crate::road::RoadSegment::Arc {
                radius: 50.0,
                sweep_angle: 1.0,
            },
        ]);
        let command = policy.command(&state, &curved_road, 10.0, &chassis()).unwrap();
        assert!(command.steering > 0.0);
    }

    #[test]
    fn pure_pursuit_turns_toward_center_from_both_lane_offsets() {
        let policy = VehicleClassicalBaseline::PurePursuitRoad(
            PurePursuitRoadConfig::default(),
        );
        let road = Road::straight();
        let mut left = VehicleState::cruising(10.0);
        left.position_y = 1.5;
        left.lateral_offset = 1.5;
        let mut right = VehicleState::cruising(10.0);
        right.position_y = -1.5;
        right.lateral_offset = -1.5;
        let left_command = policy.command(&left, &road, 10.0, &chassis()).unwrap();
        let right_command = policy.command(&right, &road, 10.0, &chassis()).unwrap();
        assert!(left_command.steering < 0.0);
        assert!(right_command.steering > 0.0);
    }

    #[test]
    fn pure_pursuit_is_centered_on_a_straight_road() {
        let policy = VehicleClassicalBaseline::PurePursuitRoad(
            PurePursuitRoadConfig::default(),
        );
        let command = policy
            .command(
                &VehicleState::cruising(10.0),
                &Road::straight(),
                10.0,
                &chassis(),
            )
            .unwrap();
        assert!(command.steering.abs() < 1e-6);
    }

    #[test]
    fn pure_pursuit_tracks_left_curvature_with_positive_steering() {
        let policy = VehicleClassicalBaseline::PurePursuitRoad(
            PurePursuitRoadConfig::default(),
        );
        let road = Road::from_segments(&[
            crate::road::RoadSegment::Arc {
                radius: 100.0,
                sweep_angle: 1.0,
            },
        ]);
        let command = policy
            .command(&VehicleState::cruising(10.0), &road, 10.0, &chassis())
            .unwrap();
        assert!(command.steering > 0.0);
    }

    #[test]
    fn pure_pursuit_digest_and_lookahead_configuration_are_validated() {
        let config = PurePursuitRoadConfig::default();
        let mut changed = config;
        changed.maximum_lookahead_m += 1.0;
        assert_ne!(config.digest(), changed.digest());
        let invalid = PurePursuitRoadConfig {
            maximum_lookahead_m: 2.0,
            ..config
        };
        assert!(invalid.digest().is_none());
    }

    #[test]
    fn baseline_configuration_digest_is_stable_and_parameter_sensitive() {
        let first = StanleyRoadConfig::default();
        let mut second = first;
        second.cross_track_gain += 0.1;
        assert_eq!(first.digest(), first.digest());
        assert_ne!(first.digest(), second.digest());
        let pd_baseline = VehicleClassicalBaseline::PdRoadTeacher;
        assert_eq!(
            pd_baseline.controller_digest(),
            Some(pd_road_baseline_digest())
        );
        assert!(
            VehicleClassicalBaseline::PurePursuitRoad(PurePursuitRoadConfig::default())
                .controller_digest()
                .is_some()
        );
    }

    #[test]
    fn invalid_baseline_parameters_are_rejected() {
        let policy = VehicleClassicalBaseline::StanleyRoad(StanleyRoadConfig {
            softening_speed_mps: 0.0,
            ..StanleyRoadConfig::default()
        });
        assert!(!policy.is_valid());
        assert!(policy.controller_digest().is_none());
    }
}
