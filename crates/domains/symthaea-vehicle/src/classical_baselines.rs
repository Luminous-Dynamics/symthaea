// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Versioned classical road-tracking baselines for controlled vehicle evaluation.
//!
//! These baselines are simulation research comparators. They are not certified
//! production controllers and their profile digests are not binary attestations.

use serde::{Deserialize, Serialize};

use crate::shadow_evaluation::pd_road_baseline_digest;
use crate::types::{
    ChassisParams, VehicleCommand, VehicleState, pd_cruise_baseline_with_road,
};

pub const STANLEY_ROAD_BASELINE_POLICY_ID: &str = "symthaea.vehicle.stanley-road-baseline.v1";

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
        }
    }

    /// Stable parameter-profile fingerprint; not a build or binary attestation.
    pub fn controller_digest(&self) -> Option<String> {
        match self {
            Self::PdRoadTeacher => Some(pd_road_baseline_digest()),
            Self::StanleyRoad(config) => config.digest(),
        }
    }

    pub fn is_valid(&self) -> bool {
        self.controller_digest().is_some()
    }

    pub fn command(
        &self,
        state: &VehicleState,
        road_heading_rad: f64,
        target_speed_mps: f64,
        chassis: &ChassisParams,
    ) -> Option<VehicleCommand> {
        if !road_heading_rad.is_finite()
            || !target_speed_mps.is_finite()
            || target_speed_mps < 0.0
            || !chassis.max_steering_angle.is_finite()
            || chassis.max_steering_angle <= 0.0
            || !chassis.wheelbase.is_finite()
            || chassis.wheelbase <= 0.0
        {
            return None;
        }

        match self {
            Self::PdRoadTeacher => {
                Some(pd_cruise_baseline_with_road(state, target_speed_mps, road_heading_rad))
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

                let heading_error = wrap_angle(road_heading_rad - state.heading);
                let cross_track_correction = (config.cross_track_gain * state.lateral_offset)
                    .atan2(state.speed.max(0.0) + config.softening_speed_mps);
                let curvature_feed_forward = (chassis.wheelbase * state.curvature_ahead
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
        left.lateral_offset = 1.0;
        let mut right = VehicleState::cruising(10.0);
        right.lateral_offset = -1.0;

        let left_command = policy.command(&left, 0.0, 10.0, &chassis()).unwrap();
        let right_command = policy.command(&right, 0.0, 10.0, &chassis()).unwrap();

        assert!(left_command.steering < 0.0);
        assert!(right_command.steering > 0.0);
    }

    #[test]
    fn stanley_has_zero_steering_on_centered_straight_road() {
        let policy = VehicleClassicalBaseline::StanleyRoad(StanleyRoadConfig::default());
        let state = VehicleState::cruising(10.0);
        let command = policy.command(&state, 0.0, 10.0, &chassis()).unwrap();
        assert!(command.steering.abs() < 1e-7);
        assert_eq!(command.throttle, 0.0);
        assert_eq!(command.brake, 0.0);
    }

    #[test]
    fn stanley_steering_includes_heading_and_curvature_feed_forward() {
        let policy = VehicleClassicalBaseline::StanleyRoad(StanleyRoadConfig::default());
        let mut state = VehicleState::cruising(10.0);
        state.heading = -0.1;
        state.curvature_ahead = 0.02;
        let command = policy.command(&state, 0.0, 10.0, &chassis()).unwrap();
        assert!(command.steering > 0.0);
    }

    #[test]
    fn baseline_configuration_digest_is_stable_and_parameter_sensitive() {
        let first = StanleyRoadConfig::default();
        let mut second = first;
        second.cross_track_gain += 0.1;
        assert_eq!(first.digest(), first.digest());
        assert_ne!(first.digest(), second.digest());
        assert_eq!(
            VehicleClassicalBaseline::PdRoadTeacher.controller_digest(),
            Some(pd_road_baseline_digest())
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
