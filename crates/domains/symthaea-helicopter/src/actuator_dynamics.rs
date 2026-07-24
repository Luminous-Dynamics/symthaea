// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Actuator servos and rotor-governor dynamics.
//!
//! Controller outputs are requests, not instantaneous swashplate or governor
//! states. This module applies bounded first-order lag and slew limits before
//! commands reach the rotor/body physics model.

use serde::{Deserialize, Serialize};

use crate::types::HelicopterCommand;

/// Time constants and rate limits for the six command channels.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ActuatorDynamicsConfig {
    /// Collective servo time constant, seconds.
    pub collective_tau_s: f64,
    /// Longitudinal/lateral cyclic servo time constant, seconds.
    pub cyclic_tau_s: f64,
    /// Pedal servo time constant, seconds.
    pub pedal_tau_s: f64,
    /// Main-rotor governor command time constant, seconds.
    pub thrust_tau_s: f64,
    /// Tail-rotor governor command time constant, seconds.
    pub tail_rotor_tau_s: f64,
    /// Maximum normalized collective travel per second.
    pub collective_rate_per_s: f32,
    /// Maximum normalized cyclic travel per second.
    pub cyclic_rate_per_s: f32,
    /// Maximum normalized pedal travel per second.
    pub pedal_rate_per_s: f32,
    /// Maximum normalized main-governor travel per second.
    pub thrust_rate_per_s: f32,
    /// Maximum normalized tail-governor travel per second.
    pub tail_rotor_rate_per_s: f32,
}

impl Default for ActuatorDynamicsConfig {
    fn default() -> Self {
        Self {
            collective_tau_s: 0.09,
            cyclic_tau_s: 0.055,
            pedal_tau_s: 0.07,
            thrust_tau_s: 0.18,
            tail_rotor_tau_s: 0.12,
            collective_rate_per_s: 2.5,
            cyclic_rate_per_s: 4.0,
            pedal_rate_per_s: 3.5,
            thrust_rate_per_s: 1.5,
            tail_rotor_rate_per_s: 2.0,
        }
    }
}

impl ActuatorDynamicsConfig {
    /// Reject values that can create discontinuities or invalid filters.
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("collective_tau_s", self.collective_tau_s),
            ("cyclic_tau_s", self.cyclic_tau_s),
            ("pedal_tau_s", self.pedal_tau_s),
            ("thrust_tau_s", self.thrust_tau_s),
            ("tail_rotor_tau_s", self.tail_rotor_tau_s),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!("{name} must be finite and > 0"));
            }
        }
        for (name, value) in [
            ("collective_rate_per_s", self.collective_rate_per_s),
            ("cyclic_rate_per_s", self.cyclic_rate_per_s),
            ("pedal_rate_per_s", self.pedal_rate_per_s),
            ("thrust_rate_per_s", self.thrust_rate_per_s),
            ("tail_rotor_rate_per_s", self.tail_rotor_rate_per_s),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!("{name} must be finite and > 0"));
            }
        }
        Ok(())
    }
}

/// Stateful actuator model. `applied` is the command seen by physics.
#[derive(Debug, Clone)]
pub struct ActuatorDynamics {
    config: ActuatorDynamicsConfig,
    applied: HelicopterCommand,
}

impl ActuatorDynamics {
    pub fn new() -> Self {
        Self::with_config(ActuatorDynamicsConfig::default())
    }

    pub fn with_config(config: ActuatorDynamicsConfig) -> Self {
        debug_assert!(config.validate().is_ok());
        Self {
            config,
            applied: HelicopterCommand::hover(),
        }
    }

    pub fn config(&self) -> &ActuatorDynamicsConfig {
        &self.config
    }

    pub fn applied_command(&self) -> HelicopterCommand {
        self.applied
    }

    /// Advance actuator states toward the requested command.
    ///
    /// Invalid/non-positive `dt` fails closed by preserving the last applied
    /// command rather than introducing NaNs or an instantaneous jump.
    pub fn step(&mut self, requested: &HelicopterCommand, dt: f64) -> HelicopterCommand {
        if !dt.is_finite() || dt <= 0.0 {
            return self.applied;
        }
        let target = requested.clamped();
        self.applied.collective = filtered_slew(
            self.applied.collective,
            target.collective,
            self.config.collective_tau_s,
            self.config.collective_rate_per_s,
            dt,
        );
        self.applied.cyclic_lon = filtered_slew(
            self.applied.cyclic_lon,
            target.cyclic_lon,
            self.config.cyclic_tau_s,
            self.config.cyclic_rate_per_s,
            dt,
        );
        self.applied.cyclic_lat = filtered_slew(
            self.applied.cyclic_lat,
            target.cyclic_lat,
            self.config.cyclic_tau_s,
            self.config.cyclic_rate_per_s,
            dt,
        );
        self.applied.pedal = filtered_slew(
            self.applied.pedal,
            target.pedal,
            self.config.pedal_tau_s,
            self.config.pedal_rate_per_s,
            dt,
        );
        self.applied.thrust = filtered_slew(
            self.applied.thrust,
            target.thrust,
            self.config.thrust_tau_s,
            self.config.thrust_rate_per_s,
            dt,
        );
        self.applied.tail_rotor = filtered_slew(
            self.applied.tail_rotor,
            target.tail_rotor,
            self.config.tail_rotor_tau_s,
            self.config.tail_rotor_rate_per_s,
            dt,
        );
        self.applied = self.applied.clamped();
        self.applied
    }

    pub fn reset_hover(&mut self) {
        self.applied = HelicopterCommand::hover();
    }

    pub fn reset_grounded(&mut self) {
        self.applied = HelicopterCommand::zero();
    }
}

impl Default for ActuatorDynamics {
    fn default() -> Self {
        Self::new()
    }
}

fn filtered_slew(current: f32, target: f32, tau_s: f64, rate_per_s: f32, dt: f64) -> f32 {
    let alpha = (1.0 - (-dt / tau_s).exp()) as f32;
    let filtered_target = current + alpha * (target - current);
    let max_delta = rate_per_s * dt as f32;
    current + (filtered_target - current).clamp(-max_delta, max_delta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_input_is_not_instantaneous() {
        let mut dynamics = ActuatorDynamics::new();
        let target = HelicopterCommand::zero();
        let applied = dynamics.step(&target, 0.01);
        assert!(applied.thrust > 0.0);
        assert!(applied.collective > 0.0);
        assert!(applied.thrust < HelicopterCommand::HOVER_THRUST);
    }

    #[test]
    fn slew_limit_bounds_one_step_motion() {
        let config = ActuatorDynamicsConfig {
            cyclic_rate_per_s: 1.0,
            ..ActuatorDynamicsConfig::default()
        };
        let mut dynamics = ActuatorDynamics::with_config(config);
        let mut target = HelicopterCommand::hover();
        target.cyclic_lon = 1.0;
        let applied = dynamics.step(&target, 0.01);
        assert!(applied.cyclic_lon <= 0.010_001);
    }

    #[test]
    fn converges_toward_requested_command() {
        let mut dynamics = ActuatorDynamics::new();
        let mut target = HelicopterCommand::hover();
        target.cyclic_lat = -0.7;
        for _ in 0..1000 {
            dynamics.step(&target, 0.01);
        }
        assert!((dynamics.applied_command().cyclic_lat + 0.7).abs() < 1e-4);
    }

    #[test]
    fn invalid_dt_preserves_last_command() {
        let mut dynamics = ActuatorDynamics::new();
        let before = dynamics.applied_command().to_ctrl();
        dynamics.step(&HelicopterCommand::zero(), f64::NAN);
        assert_eq!(dynamics.applied_command().to_ctrl(), before);
    }
}
