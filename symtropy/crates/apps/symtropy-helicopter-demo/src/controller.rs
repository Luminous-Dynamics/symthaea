// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Station-hold controller: hovers over a fixed ground point at target altitude.
//!
//! - Collective: PD on altitude error (biased toward HOVER_COLLECTIVE).
//! - Cyclic longitudinal (pitch): tilts the rotor disc to correct X-drift.
//! - Cyclic lateral (roll): tilts the rotor disc to correct Y-drift.
//! - Pedal: damps yaw rate.
//! - Thrust + tail_rotor: stay at HOVER defaults.

use symthaea_helicopter::types::{HelicopterCommand, HelicopterState};

pub struct StationHoldController {
    pub target: [f64; 3],
    pub kp_alt: f64,
    pub kd_alt: f64,
    pub kp_pos: f64,
    pub yaw_rate_kd: f64,
}

impl Default for StationHoldController {
    fn default() -> Self {
        Self {
            target: [0.0, 0.0, 20.0],
            kp_alt: 0.06,
            kd_alt: 0.08,
            kp_pos: 0.04,
            yaw_rate_kd: 0.20,
        }
    }
}

impl StationHoldController {
    pub fn compute(&self, state: &HelicopterState) -> HelicopterCommand {
        // Altitude control → collective
        let dz = self.target[2] - state.position[2];
        let vz = state.linear_velocity[2];
        let collective_delta = self.kp_alt * dz - self.kd_alt * vz;
        let collective =
            (HelicopterCommand::HOVER_COLLECTIVE as f64 + collective_delta).clamp(0.0, 1.0) as f32;

        // Horizontal-position error → desired body tilt
        // (forward tilt to catch up in +x, right roll to catch up in +y)
        let dx = self.target[0] - state.position[0];
        let dy = self.target[1] - state.position[1];
        let cyclic_lon = (self.kp_pos * dx).clamp(-0.8, 0.8) as f32;
        let cyclic_lat = (-self.kp_pos * dy).clamp(-0.8, 0.8) as f32;

        // Also damp body rates to avoid oscillation
        let (_roll, _pitch, _yaw) = state.euler_angles();
        let wz = state.angular_velocity[2];
        let pedal = (-self.yaw_rate_kd * wz).clamp(-0.8, 0.8) as f32;

        HelicopterCommand {
            collective,
            cyclic_lon,
            cyclic_lat,
            pedal,
            thrust: HelicopterCommand::HOVER_THRUST,
            tail_rotor: HelicopterCommand::HOVER_TAIL,
        }
    }
}

/// Scale the command by a consciousness motor gain.
///
/// - Collective is interpolated between HOVER_COLLECTIVE and commanded (so a
///   low gain retreats toward level hover, never dropping to zero collective
///   and killing altitude).
/// - Cyclic+pedal are attenuated uniformly toward zero.
/// - Thrust + tail_rotor are pass-through (rotor RPM must stay at hover;
///   dropping RPM would drop lift capacity mid-gust).
pub fn gain_scale(cmd: HelicopterCommand, gain: f64) -> HelicopterCommand {
    let g = gain.clamp(0.0, 1.0) as f32;
    let hover_coll = HelicopterCommand::HOVER_COLLECTIVE;
    HelicopterCommand {
        collective: hover_coll + g * (cmd.collective - hover_coll),
        cyclic_lon: g * cmd.cyclic_lon,
        cyclic_lat: g * cmd.cyclic_lat,
        pedal: g * cmd.pedal,
        thrust: cmd.thrust,
        tail_rotor: cmd.tail_rotor,
    }
}
