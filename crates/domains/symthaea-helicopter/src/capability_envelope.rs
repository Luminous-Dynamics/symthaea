// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Capability-derived degraded flight envelope.
//!
//! Static Green/Yellow/Orange limits remain useful policy boundaries, but they
//! do not reflect the actual authority remaining after a specific combination
//! of rotor-speed and actuator degradations. This module derives maneuver limits
//! directly from the per-axis controllability assessment and exposes the action
//! required when hover or directional control can no longer be guaranteed.

use serde::{Deserialize, Serialize};

use crate::controllability_margin::{ControlAxis, ControllabilityAssessment, ControllabilityState};
use crate::types::HelicopterCommand;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CapabilityEnvelopeConfig {
    pub nominal_horizontal_speed_mps: f64,
    pub nominal_bank_angle_deg: f64,
    pub nominal_pitch_angle_deg: f64,
    pub nominal_climb_rate_mps: f64,
    pub nominal_descent_rate_mps: f64,
    pub nominal_yaw_rate_rad_s: f64,
    pub minimum_hover_authority: f64,
    pub minimum_directional_authority: f64,
    pub immediate_landing_margin_fraction: f64,
}

impl Default for CapabilityEnvelopeConfig {
    fn default() -> Self {
        Self {
            nominal_horizontal_speed_mps: 35.0,
            nominal_bank_angle_deg: 35.0,
            nominal_pitch_angle_deg: 25.0,
            nominal_climb_rate_mps: 4.0,
            nominal_descent_rate_mps: 5.0,
            nominal_yaw_rate_rad_s: 0.8,
            minimum_hover_authority: 0.35,
            minimum_directional_authority: 0.20,
            immediate_landing_margin_fraction: 0.05,
        }
    }
}

impl CapabilityEnvelopeConfig {
    pub fn validate(&self) -> Result<(), CapabilityEnvelopeError> {
        let positive = [
            self.nominal_horizontal_speed_mps,
            self.nominal_bank_angle_deg,
            self.nominal_pitch_angle_deg,
            self.nominal_climb_rate_mps,
            self.nominal_descent_rate_mps,
            self.nominal_yaw_rate_rad_s,
        ];
        if positive
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
            || !self.minimum_hover_authority.is_finite()
            || !(0.0..=1.0).contains(&self.minimum_hover_authority)
            || !self.minimum_directional_authority.is_finite()
            || !(0.0..=1.0).contains(&self.minimum_directional_authority)
            || !self.immediate_landing_margin_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.immediate_landing_margin_fraction)
        {
            return Err(CapabilityEnvelopeError::InvalidConfig);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapabilityEnvelopeAction {
    Continue,
    ReduceEnvelope,
    LandImmediately,
    Autorotate,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CapabilityDerivedEnvelope {
    pub action: CapabilityEnvelopeAction,
    pub max_horizontal_speed_mps: f64,
    pub max_bank_angle_deg: f64,
    pub max_pitch_angle_deg: f64,
    pub max_climb_rate_mps: f64,
    pub max_descent_rate_mps: f64,
    pub max_yaw_rate_rad_s: f64,
    pub max_cyclic_command: f32,
    pub max_pedal_command: f32,
    pub hover_supported: bool,
    pub directional_control_supported: bool,
    pub limiting_axis: ControlAxis,
}

impl CapabilityDerivedEnvelope {
    pub fn clamp_command(&self, mut command: HelicopterCommand) -> HelicopterCommand {
        command.cyclic_lon = command
            .cyclic_lon
            .clamp(-self.max_cyclic_command, self.max_cyclic_command);
        command.cyclic_lat = command
            .cyclic_lat
            .clamp(-self.max_cyclic_command, self.max_cyclic_command);
        command.pedal = command
            .pedal
            .clamp(-self.max_pedal_command, self.max_pedal_command);
        if !self.directional_control_supported {
            command.pedal = 0.0;
        }
        if self.action == CapabilityEnvelopeAction::Autorotate {
            command.thrust = 0.0;
        }
        command.clamped()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilityEnvelopeError {
    InvalidConfig,
    NonFiniteAssessment,
}

#[derive(Debug, Clone)]
pub struct CapabilityEnvelopeDeriver {
    config: CapabilityEnvelopeConfig,
}

impl CapabilityEnvelopeDeriver {
    pub fn new(config: CapabilityEnvelopeConfig) -> Result<Self, CapabilityEnvelopeError> {
        config.validate()?;
        Ok(Self { config })
    }

    pub fn derive(
        &self,
        assessment: &ControllabilityAssessment,
    ) -> Result<CapabilityDerivedEnvelope, CapabilityEnvelopeError> {
        self.config.validate()?;
        if !assessment.minimum_margin_fraction.is_finite()
            || assessment.axes.iter().any(|axis| {
                !axis.retained_authority.is_finite()
                    || !axis.margin_fraction.is_finite()
                    || !axis.available_accel.is_finite()
            })
        {
            return Err(CapabilityEnvelopeError::NonFiniteAssessment);
        }

        let vertical = assessment.axis(ControlAxis::Vertical).retained_authority;
        let roll = assessment.axis(ControlAxis::Roll).retained_authority;
        let pitch = assessment.axis(ControlAxis::Pitch).retained_authority;
        let yaw = assessment.axis(ControlAxis::Yaw).retained_authority;
        let lateral = roll.min(pitch);
        let hover_supported = vertical >= self.config.minimum_hover_authority;
        let directional_control_supported = yaw >= self.config.minimum_directional_authority;

        let action = if vertical <= 1.0e-6 {
            CapabilityEnvelopeAction::Autorotate
        } else if !hover_supported
            || assessment.minimum_margin_fraction <= self.config.immediate_landing_margin_fraction
            || assessment.state == ControllabilityState::Uncontrollable
        {
            CapabilityEnvelopeAction::LandImmediately
        } else if assessment.state == ControllabilityState::Degraded
            || lateral < 0.999
            || yaw < 0.999
            || vertical < 0.999
        {
            CapabilityEnvelopeAction::ReduceEnvelope
        } else {
            CapabilityEnvelopeAction::Continue
        };

        let speed_scale = lateral.sqrt().clamp(0.0, 1.0);
        let max_horizontal_speed_mps = self.config.nominal_horizontal_speed_mps * speed_scale;
        let max_bank_angle_deg = self.config.nominal_bank_angle_deg * roll.clamp(0.0, 1.0);
        let max_pitch_angle_deg = self.config.nominal_pitch_angle_deg * pitch.clamp(0.0, 1.0);
        let max_climb_rate_mps = self.config.nominal_climb_rate_mps * vertical.clamp(0.0, 1.0);
        let max_descent_rate_mps =
            self.config.nominal_descent_rate_mps * (0.35 + 0.65 * vertical.clamp(0.0, 1.0));
        let max_yaw_rate_rad_s = self.config.nominal_yaw_rate_rad_s * yaw.clamp(0.0, 1.0);
        let max_cyclic_command = (0.15 + 0.85 * lateral.clamp(0.0, 1.0)) as f32;
        let max_pedal_command = yaw.clamp(0.0, 1.0) as f32;

        Ok(CapabilityDerivedEnvelope {
            action,
            max_horizontal_speed_mps,
            max_bank_angle_deg,
            max_pitch_angle_deg,
            max_climb_rate_mps,
            max_descent_rate_mps,
            max_yaw_rate_rad_s,
            max_cyclic_command,
            max_pedal_command,
            hover_supported,
            directional_control_supported,
            limiting_axis: assessment.limiting_axis,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control_allocation::{ActuatorHealth, VirtualControlDemand};
    use crate::controllability_margin::ControllabilityMarginEvaluator;
    use crate::types::HelicopterState;

    fn nominal_assessment() -> ControllabilityAssessment {
        ControllabilityMarginEvaluator::default()
            .assess(
                &HelicopterState::hover(20.0),
                ActuatorHealth::default(),
                VirtualControlDemand::default(),
            )
            .unwrap()
    }

    #[test]
    fn nominal_capability_preserves_full_envelope() {
        let envelope = CapabilityEnvelopeDeriver::new(CapabilityEnvelopeConfig::default())
            .unwrap()
            .derive(&nominal_assessment())
            .unwrap();
        assert_eq!(envelope.action, CapabilityEnvelopeAction::Continue);
        assert_eq!(envelope.max_horizontal_speed_mps, 35.0);
        assert!(envelope.hover_supported);
    }

    #[test]
    fn yaw_loss_removes_pedal_authority() {
        let mut health = ActuatorHealth::default();
        health.tail_rotor = 0.0;
        health.pedal = 0.0;
        let assessment = ControllabilityMarginEvaluator::default()
            .assess(
                &HelicopterState::hover(20.0),
                health,
                VirtualControlDemand::default(),
            )
            .unwrap();
        let envelope = CapabilityEnvelopeDeriver::new(CapabilityEnvelopeConfig::default())
            .unwrap()
            .derive(&assessment)
            .unwrap();
        assert!(!envelope.directional_control_supported);
        let command = envelope.clamp_command(HelicopterCommand {
            pedal: 1.0,
            ..HelicopterCommand::hover()
        });
        assert_eq!(command.pedal, 0.0);
    }

    #[test]
    fn total_main_rotor_loss_requires_autorotation() {
        let mut health = ActuatorHealth::default();
        health.main_rotor = 0.0;
        health.collective = 0.0;
        let assessment = ControllabilityMarginEvaluator::default()
            .assess(
                &HelicopterState::hover(20.0),
                health,
                VirtualControlDemand::default(),
            )
            .unwrap();
        let envelope = CapabilityEnvelopeDeriver::new(CapabilityEnvelopeConfig::default())
            .unwrap()
            .derive(&assessment)
            .unwrap();
        assert_eq!(envelope.action, CapabilityEnvelopeAction::Autorotate);
        assert_eq!(
            envelope.clamp_command(HelicopterCommand::hover()).thrust,
            0.0
        );
    }
}
