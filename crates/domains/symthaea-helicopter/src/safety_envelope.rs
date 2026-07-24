// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Flight-envelope authority policy for Green, Yellow, and Orange operation.
//!
//! Reduced authority is expressed as explicit speed/attitude/control limits.
//! It never multiplies lift-producing channels toward zero, which would turn a
//! caution state into an unintended descent.

use serde::{Deserialize, Serialize};
use symthaea_core::embodiment::MotorSafetyLevel;

use crate::controller::pd_hover_baseline;
use crate::types::{HelicopterCommand, HelicopterState};

/// Limits for a reduced flight envelope.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FlightEnvelope {
    pub max_cyclic: f32,
    pub max_pedal: f32,
    pub collective_min: f32,
    pub collective_max: f32,
    pub thrust_min: f32,
    pub thrust_max: f32,
    pub tail_rotor_min: f32,
    pub tail_rotor_max: f32,
    pub max_horizontal_speed_mps: f64,
    pub max_tilt_rad: f64,
    pub max_angular_speed_rad_s: f64,
}

impl FlightEnvelope {
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("max_cyclic", self.max_cyclic),
            ("max_pedal", self.max_pedal),
            ("collective_min", self.collective_min),
            ("collective_max", self.collective_max),
            ("thrust_min", self.thrust_min),
            ("thrust_max", self.thrust_max),
            ("tail_rotor_min", self.tail_rotor_min),
            ("tail_rotor_max", self.tail_rotor_max),
        ] {
            if !value.is_finite() {
                return Err(format!("{name} must be finite"));
            }
        }
        if self.max_cyclic <= 0.0 || self.max_cyclic > 1.0 {
            return Err("max_cyclic must be in (0, 1]".to_string());
        }
        if self.max_pedal <= 0.0 || self.max_pedal > 1.0 {
            return Err("max_pedal must be in (0, 1]".to_string());
        }
        if self.collective_min > self.collective_max
            || self.thrust_min > self.thrust_max
            || self.tail_rotor_min > self.tail_rotor_max
        {
            return Err("envelope minimum must not exceed maximum".to_string());
        }
        for (name, value) in [
            ("max_horizontal_speed_mps", self.max_horizontal_speed_mps),
            ("max_tilt_rad", self.max_tilt_rad),
            ("max_angular_speed_rad_s", self.max_angular_speed_rad_s),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!("{name} must be finite and > 0"));
            }
        }
        Ok(())
    }

    fn clamp_command(&self, mut cmd: HelicopterCommand) -> HelicopterCommand {
        cmd.cyclic_lon = cmd.cyclic_lon.clamp(-self.max_cyclic, self.max_cyclic);
        cmd.cyclic_lat = cmd.cyclic_lat.clamp(-self.max_cyclic, self.max_cyclic);
        cmd.pedal = cmd.pedal.clamp(-self.max_pedal, self.max_pedal);
        cmd.collective = cmd
            .collective
            .clamp(self.collective_min, self.collective_max);
        cmd.thrust = cmd.thrust.clamp(self.thrust_min, self.thrust_max);
        cmd.tail_rotor = cmd
            .tail_rotor
            .clamp(self.tail_rotor_min, self.tail_rotor_max);
        cmd.clamped()
    }
}

/// Stateful authority policy. Orange captures a stable altitude target when it
/// is entered rather than continuously moving the target with the aircraft.
#[derive(Debug, Clone)]
pub struct FlightAuthorityPolicy {
    yellow: FlightEnvelope,
    orange: FlightEnvelope,
    orange_hold_altitude_m: Option<f64>,
}

impl Default for FlightAuthorityPolicy {
    fn default() -> Self {
        Self {
            yellow: FlightEnvelope {
                max_cyclic: 0.45,
                max_pedal: 0.45,
                collective_min: 0.15,
                collective_max: 0.60,
                thrust_min: 0.50,
                thrust_max: 0.75,
                tail_rotor_min: 0.35,
                tail_rotor_max: 0.70,
                max_horizontal_speed_mps: 12.0,
                max_tilt_rad: 25.0_f64.to_radians(),
                max_angular_speed_rad_s: 1.5,
            },
            orange: FlightEnvelope {
                max_cyclic: 0.25,
                max_pedal: 0.25,
                collective_min: 0.20,
                collective_max: 0.50,
                thrust_min: 0.55,
                thrust_max: 0.68,
                tail_rotor_min: 0.40,
                tail_rotor_max: 0.62,
                max_horizontal_speed_mps: 3.0,
                max_tilt_rad: 12.0_f64.to_radians(),
                max_angular_speed_rad_s: 0.65,
            },
            orange_hold_altitude_m: None,
        }
    }
}

impl FlightAuthorityPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn apply(
        &mut self,
        level: MotorSafetyLevel,
        state: &HelicopterState,
        requested: HelicopterCommand,
    ) -> HelicopterCommand {
        match level {
            MotorSafetyLevel::Green => {
                self.orange_hold_altitude_m = None;
                requested.clamped()
            }
            MotorSafetyLevel::Yellow => {
                self.orange_hold_altitude_m = None;
                self.apply_envelope(state, requested, self.yellow)
            }
            MotorSafetyLevel::Orange => {
                let hold_altitude = *self.orange_hold_altitude_m.get_or_insert(state.altitude());
                let mut hold = pd_hover_baseline(state, hold_altitude);
                // Arrest translation while preserving enough lift to remain airborne.
                hold.cyclic_lon += (-0.08 * state.linear_velocity[0]) as f32;
                hold.cyclic_lat += (0.08 * state.linear_velocity[1]) as f32;
                self.apply_envelope(state, hold, self.orange)
            }
            // Red is owned by EmergencyLandingController. Do not transform its
            // carefully staged command here.
            MotorSafetyLevel::Red => requested.clamped(),
        }
    }

    fn apply_envelope(
        &self,
        state: &HelicopterState,
        mut cmd: HelicopterCommand,
        envelope: FlightEnvelope,
    ) -> HelicopterCommand {
        let (roll, pitch, _) = state.euler_angles();
        if state.horizontal_speed() > envelope.max_horizontal_speed_mps {
            cmd.cyclic_lon += (-0.10 * state.linear_velocity[0]) as f32;
            cmd.cyclic_lat += (0.10 * state.linear_velocity[1]) as f32;
        }
        if roll.abs() > envelope.max_tilt_rad || pitch.abs() > envelope.max_tilt_rad {
            cmd.cyclic_lon += (-0.8 * pitch - 0.3 * state.angular_velocity[1]) as f32;
            cmd.cyclic_lat += (-0.8 * roll - 0.3 * state.angular_velocity[0]) as f32;
        }
        if state.angular_speed() > envelope.max_angular_speed_rad_s {
            cmd.cyclic_lon += (-0.25 * state.angular_velocity[1]) as f32;
            cmd.cyclic_lat += (-0.25 * state.angular_velocity[0]) as f32;
            cmd.pedal += (-0.20 * state.angular_velocity[2]) as f32;
        }
        envelope.clamp_command(cmd)
    }

    pub fn reset(&mut self) {
        self.orange_hold_altitude_m = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yellow_limits_maneuver_without_removing_hover_lift() {
        let mut policy = FlightAuthorityPolicy::new();
        let state = HelicopterState::hover(20.0);
        let requested = HelicopterCommand {
            collective: 1.0,
            cyclic_lon: 1.0,
            cyclic_lat: -1.0,
            pedal: 1.0,
            thrust: 0.0,
            tail_rotor: 0.0,
        };
        let applied = policy.apply(MotorSafetyLevel::Yellow, &state, requested);
        assert!(applied.cyclic_lon <= 0.45);
        assert!(applied.thrust >= 0.50);
        assert!(applied.tail_rotor >= 0.35);
    }

    #[test]
    fn orange_opposes_translation_and_preserves_lift() {
        let mut policy = FlightAuthorityPolicy::new();
        let mut state = HelicopterState::hover(20.0);
        state.linear_velocity[0] = 5.0;
        let applied = policy.apply(MotorSafetyLevel::Orange, &state, HelicopterCommand::zero());
        assert!(applied.cyclic_lon < 0.0);
        assert!(applied.thrust >= 0.55);
    }

    #[test]
    fn red_command_passes_through_unchanged_except_clamp() {
        let mut policy = FlightAuthorityPolicy::new();
        let state = HelicopterState::hover(20.0);
        let emergency = HelicopterCommand {
            collective: 0.1,
            cyclic_lon: 0.0,
            cyclic_lat: 0.0,
            pedal: 0.0,
            thrust: 0.0,
            tail_rotor: 0.1,
        };
        assert_eq!(
            policy
                .apply(MotorSafetyLevel::Red, &state, emergency)
                .to_ctrl(),
            emergency.to_ctrl()
        );
    }
}
