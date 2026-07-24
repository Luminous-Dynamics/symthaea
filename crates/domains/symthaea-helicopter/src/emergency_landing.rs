// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Time-based, state-aware emergency landing controller.
//!
//! This is a simulation safety policy, not an aviation-certified procedure.
//! It avoids scheduler-dependent cycle counts and keeps one canonical fallback
//! command generator for embodiment and future benchmark use.

use serde::{Deserialize, Serialize};

use crate::controller::pd_hover_baseline;
use crate::types::{HelicopterCommand, HelicopterState};

/// Emergency fallback phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HelicopterFallbackStage {
    StabilizeHover,
    AutorotationDescent,
    Touchdown,
}

/// Thresholds for phase transitions and flare shaping.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EmergencyLandingConfig {
    /// Minimum time spent attempting stabilized hover before descent escalation.
    pub stabilize_timeout_s: f64,
    /// Descent rate that indicates hover recovery has failed.
    pub failed_hover_descent_mps: f64,
    /// Rotor RPM below which powered hover recovery is not credible.
    pub minimum_powered_hover_rpm: f64,
    /// Below this altitude, skip directly to touchdown/flare control.
    pub direct_touchdown_altitude_m: f64,
    /// Autorotation transitions to flare below this altitude.
    pub flare_altitude_m: f64,
    /// Maximum flare collective request.
    pub flare_collective: f32,
}

impl Default for EmergencyLandingConfig {
    fn default() -> Self {
        Self {
            stabilize_timeout_s: 2.0,
            failed_hover_descent_mps: -2.0,
            minimum_powered_hover_rpm: 2_200.0,
            direct_touchdown_altitude_m: 8.0,
            flare_altitude_m: 5.0,
            flare_collective: 0.55,
        }
    }
}

impl EmergencyLandingConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !self.stabilize_timeout_s.is_finite() || self.stabilize_timeout_s < 0.0 {
            return Err("stabilize_timeout_s must be finite and >= 0".to_string());
        }
        if !self.failed_hover_descent_mps.is_finite() || self.failed_hover_descent_mps >= 0.0 {
            return Err("failed_hover_descent_mps must be finite and < 0".to_string());
        }
        if !self.minimum_powered_hover_rpm.is_finite() || self.minimum_powered_hover_rpm < 0.0 {
            return Err("minimum_powered_hover_rpm must be finite and >= 0".to_string());
        }
        if !self.direct_touchdown_altitude_m.is_finite()
            || self.direct_touchdown_altitude_m < 0.0
            || !self.flare_altitude_m.is_finite()
            || self.flare_altitude_m < 0.0
        {
            return Err("landing altitudes must be finite and >= 0".to_string());
        }
        if self.flare_altitude_m > self.direct_touchdown_altitude_m {
            return Err("flare_altitude_m must not exceed direct_touchdown_altitude_m".to_string());
        }
        if !self.flare_collective.is_finite() || !(0.0..=1.0).contains(&self.flare_collective) {
            return Err("flare_collective must be finite and in [0, 1]".to_string());
        }
        Ok(())
    }
}

/// Stateful emergency command generator.
#[derive(Debug, Clone)]
pub struct EmergencyLandingController {
    config: EmergencyLandingConfig,
    stage: HelicopterFallbackStage,
    elapsed_in_stage_s: f64,
    hold_altitude_m: Option<f64>,
}

impl EmergencyLandingController {
    pub fn new() -> Self {
        Self::with_config(EmergencyLandingConfig::default())
    }

    pub fn with_config(config: EmergencyLandingConfig) -> Self {
        debug_assert!(config.validate().is_ok());
        Self {
            config,
            stage: HelicopterFallbackStage::StabilizeHover,
            elapsed_in_stage_s: 0.0,
            hold_altitude_m: None,
        }
    }

    pub fn stage(&self) -> HelicopterFallbackStage {
        self.stage
    }

    pub fn elapsed_in_stage_s(&self) -> f64 {
        self.elapsed_in_stage_s
    }

    /// Produce the canonical emergency command for this state and timestep.
    pub fn command(&mut self, state: &HelicopterState, dt: f64) -> HelicopterCommand {
        let dt = if dt.is_finite() && dt > 0.0 { dt } else { 0.0 };
        self.elapsed_in_stage_s += dt;
        self.hold_altitude_m.get_or_insert(state.altitude());
        self.update_stage(state);

        match self.stage {
            HelicopterFallbackStage::StabilizeHover => {
                pd_hover_baseline(state, self.hold_altitude_m.unwrap_or(state.altitude()))
            }
            HelicopterFallbackStage::AutorotationDescent => HelicopterCommand {
                collective: 0.10,
                cyclic_lon: (-0.35 * state.angular_velocity[1]) as f32,
                cyclic_lat: (-0.35 * state.angular_velocity[0]) as f32,
                pedal: (-0.25 * state.angular_velocity[2]) as f32,
                thrust: 0.0,
                tail_rotor: 0.1,
            }
            .clamped(),
            HelicopterFallbackStage::Touchdown => {
                let sink_speed = (-state.linear_velocity[2]).max(0.0);
                let flare_fraction = (sink_speed / 5.0).clamp(0.0, 1.0) as f32;
                HelicopterCommand {
                    collective: 0.25 + flare_fraction * (self.config.flare_collective - 0.25),
                    cyclic_lon: (-0.12 - 0.30 * state.angular_velocity[1]) as f32,
                    cyclic_lat: (-0.30 * state.angular_velocity[0]) as f32,
                    pedal: (-0.25 * state.angular_velocity[2]) as f32,
                    thrust: 0.0,
                    tail_rotor: 0.1,
                }
                .clamped()
            }
        }
    }

    fn update_stage(&mut self, state: &HelicopterState) {
        let altitude = state.altitude();
        match self.stage {
            HelicopterFallbackStage::StabilizeHover => {
                if altitude <= self.config.direct_touchdown_altitude_m {
                    self.transition_to(HelicopterFallbackStage::Touchdown);
                } else {
                    let recovery_failed = state.linear_velocity[2]
                        <= self.config.failed_hover_descent_mps
                        || state.main_rotor_rpm < self.config.minimum_powered_hover_rpm;
                    if recovery_failed && self.elapsed_in_stage_s >= self.config.stabilize_timeout_s
                    {
                        self.transition_to(HelicopterFallbackStage::AutorotationDescent);
                    }
                }
            }
            HelicopterFallbackStage::AutorotationDescent => {
                if altitude <= self.config.flare_altitude_m {
                    self.transition_to(HelicopterFallbackStage::Touchdown);
                }
            }
            HelicopterFallbackStage::Touchdown => {}
        }
    }

    fn transition_to(&mut self, stage: HelicopterFallbackStage) {
        if self.stage != stage {
            self.stage = stage;
            self.elapsed_in_stage_s = 0.0;
        }
    }

    pub fn reset(&mut self) {
        self.stage = HelicopterFallbackStage::StabilizeHover;
        self.elapsed_in_stage_s = 0.0;
        self.hold_altitude_m = None;
    }
}

impl Default for EmergencyLandingController {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_hover_does_not_escalate_by_cycle_count() {
        let mut controller = EmergencyLandingController::new();
        let state = HelicopterState::hover(20.0);
        for _ in 0..10_000 {
            controller.command(&state, 0.001);
        }
        assert_eq!(controller.stage(), HelicopterFallbackStage::StabilizeHover);
    }

    #[test]
    fn failed_hover_escalates_after_elapsed_time() {
        let mut controller = EmergencyLandingController::new();
        let mut state = HelicopterState::hover(20.0);
        state.linear_velocity[2] = -3.0;
        for _ in 0..199 {
            controller.command(&state, 0.01);
        }
        assert_eq!(controller.stage(), HelicopterFallbackStage::StabilizeHover);
        controller.command(&state, 0.02);
        assert_eq!(
            controller.stage(),
            HelicopterFallbackStage::AutorotationDescent
        );
    }

    #[test]
    fn low_altitude_goes_directly_to_touchdown() {
        let mut controller = EmergencyLandingController::new();
        let state = HelicopterState::hover(4.0);
        let cmd = controller.command(&state, 0.01);
        assert_eq!(controller.stage(), HelicopterFallbackStage::Touchdown);
        assert_eq!(cmd.thrust, 0.0);
        assert!(cmd.collective >= 0.25);
    }

    #[test]
    fn reset_clears_stage_and_timer() {
        let mut controller = EmergencyLandingController::new();
        let mut state = HelicopterState::hover(20.0);
        state.main_rotor_rpm = 0.0;
        controller.command(&state, 3.0);
        assert_eq!(
            controller.stage(),
            HelicopterFallbackStage::AutorotationDescent
        );
        controller.reset();
        assert_eq!(controller.stage(), HelicopterFallbackStage::StabilizeHover);
        assert_eq!(controller.elapsed_in_stage_s(), 0.0);
    }
}
