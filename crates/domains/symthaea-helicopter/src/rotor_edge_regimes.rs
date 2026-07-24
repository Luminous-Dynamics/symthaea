// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reduced-order retreating-blade stall and advancing-tip compressibility gates.
//!
//! The existing rotor model exposes translational lift, vortex-ring state, and
//! autorotation. High forward speed creates a different pair of hazards: loss of
//! retreating-side dynamic pressure and compressibility on the advancing tip.
//! This module makes those margins explicit and applies conservative command
//! limits. It is not a blade-element or aeroelastic certification model.

use serde::{Deserialize, Serialize};

use crate::types::HelicopterCommand;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RotorEdgeRegimeConfig {
    pub rotor_radius_m: f64,
    /// Convert reduced-order governor RPM into physical main-rotor RPM.
    pub physical_rotor_rpm_scale: f64,
    pub maximum_advancing_tip_mach: f64,
    pub caution_advancing_tip_mach: f64,
    pub maximum_advance_ratio: f64,
    pub caution_advance_ratio: f64,
    pub collective_stall_coupling: f64,
    pub minimum_cyclic_limit: f32,
    pub maximum_protected_collective: f32,
}

impl Default for RotorEdgeRegimeConfig {
    fn default() -> Self {
        Self {
            rotor_radius_m: 5.3,
            physical_rotor_rpm_scale: 0.12,
            maximum_advancing_tip_mach: 0.90,
            caution_advancing_tip_mach: 0.82,
            maximum_advance_ratio: 0.42,
            caution_advance_ratio: 0.34,
            collective_stall_coupling: 0.10,
            minimum_cyclic_limit: 0.12,
            maximum_protected_collective: 0.45,
        }
    }
}

impl RotorEdgeRegimeConfig {
    pub fn validate(&self) -> Result<(), RotorEdgeRegimeError> {
        if !self.rotor_radius_m.is_finite()
            || self.rotor_radius_m <= 0.0
            || !self.physical_rotor_rpm_scale.is_finite()
            || !(0.0..=1.0).contains(&self.physical_rotor_rpm_scale)
            || !self.maximum_advancing_tip_mach.is_finite()
            || !self.caution_advancing_tip_mach.is_finite()
            || !(0.0..1.5).contains(&self.maximum_advancing_tip_mach)
            || !(0.0..self.maximum_advancing_tip_mach).contains(&self.caution_advancing_tip_mach)
            || !self.maximum_advance_ratio.is_finite()
            || !self.caution_advance_ratio.is_finite()
            || !(0.0..1.0).contains(&self.maximum_advance_ratio)
            || !(0.0..self.maximum_advance_ratio).contains(&self.caution_advance_ratio)
            || !self.collective_stall_coupling.is_finite()
            || self.collective_stall_coupling < 0.0
            || !self.minimum_cyclic_limit.is_finite()
            || !(0.0..=1.0).contains(&self.minimum_cyclic_limit)
            || !self.maximum_protected_collective.is_finite()
            || !(0.0..=1.0).contains(&self.maximum_protected_collective)
        {
            return Err(RotorEdgeRegimeError::InvalidConfiguration);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RotorEdgeObservation {
    pub main_rotor_rpm: f64,
    pub horizontal_airspeed_mps: f64,
    pub outside_air_temperature_k: f64,
    pub collective_fraction: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RotorEdgeRegime {
    Normal,
    Caution,
    RetreatingBladeStallExposure,
    AdvancingTipCompressibility,
    CombinedLimit,
    RotorSpeedUnavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RotorEdgeAssessment {
    pub regime: RotorEdgeRegime,
    pub tip_speed_mps: f64,
    pub advance_ratio: f64,
    pub effective_stall_ratio: f64,
    pub advancing_tip_mach: f64,
    pub retreating_tip_speed_mps: f64,
    pub cyclic_limit: f32,
    pub collective_limit: f32,
    pub protected_command: HelicopterCommand,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotorEdgeRegimeError {
    InvalidConfiguration,
    NonFiniteObservation,
}

#[derive(Debug, Clone)]
pub struct RotorEdgeRegimeProtector {
    config: RotorEdgeRegimeConfig,
}

impl RotorEdgeRegimeProtector {
    pub fn new(config: RotorEdgeRegimeConfig) -> Result<Self, RotorEdgeRegimeError> {
        config.validate()?;
        Ok(Self { config })
    }

    pub fn assess(
        &self,
        requested: HelicopterCommand,
        observation: RotorEdgeObservation,
    ) -> Result<RotorEdgeAssessment, RotorEdgeRegimeError> {
        self.config.validate()?;
        if [
            observation.main_rotor_rpm,
            observation.horizontal_airspeed_mps,
            observation.outside_air_temperature_k,
            observation.collective_fraction,
        ]
        .iter()
        .any(|value| !value.is_finite())
            || observation.main_rotor_rpm < 0.0
            || observation.horizontal_airspeed_mps < 0.0
            || observation.outside_air_temperature_k <= 0.0
            || !(0.0..=1.0).contains(&observation.collective_fraction)
        {
            return Err(RotorEdgeRegimeError::NonFiniteObservation);
        }

        let physical_rotor_rpm = observation.main_rotor_rpm * self.config.physical_rotor_rpm_scale;
        let omega_rad_s = physical_rotor_rpm * std::f64::consts::TAU / 60.0;
        let tip_speed_mps = omega_rad_s * self.config.rotor_radius_m;
        if tip_speed_mps <= 1.0e-6 {
            return Ok(RotorEdgeAssessment {
                regime: RotorEdgeRegime::RotorSpeedUnavailable,
                tip_speed_mps,
                advance_ratio: f64::INFINITY,
                effective_stall_ratio: f64::INFINITY,
                advancing_tip_mach: 0.0,
                retreating_tip_speed_mps: 0.0,
                cyclic_limit: 0.0,
                collective_limit: 0.0,
                protected_command: HelicopterCommand::zero(),
            });
        }

        let speed_of_sound_mps = (1.4 * 287.05 * observation.outside_air_temperature_k).sqrt();
        let advance_ratio = observation.horizontal_airspeed_mps / tip_speed_mps;
        let effective_stall_ratio =
            advance_ratio + self.config.collective_stall_coupling * observation.collective_fraction;
        let advancing_tip_mach =
            (tip_speed_mps + observation.horizontal_airspeed_mps) / speed_of_sound_mps;
        let retreating_tip_speed_mps =
            (tip_speed_mps - observation.horizontal_airspeed_mps).max(0.0);

        let retreating_limit = effective_stall_ratio >= self.config.maximum_advance_ratio;
        let compressibility_limit = advancing_tip_mach >= self.config.maximum_advancing_tip_mach;
        let caution = effective_stall_ratio >= self.config.caution_advance_ratio
            || advancing_tip_mach >= self.config.caution_advancing_tip_mach;
        let regime = match (retreating_limit, compressibility_limit, caution) {
            (true, true, _) => RotorEdgeRegime::CombinedLimit,
            (true, false, _) => RotorEdgeRegime::RetreatingBladeStallExposure,
            (false, true, _) => RotorEdgeRegime::AdvancingTipCompressibility,
            (false, false, true) => RotorEdgeRegime::Caution,
            _ => RotorEdgeRegime::Normal,
        };

        let stall_margin = normalized_margin(
            effective_stall_ratio,
            self.config.caution_advance_ratio,
            self.config.maximum_advance_ratio,
        );
        let mach_margin = normalized_margin(
            advancing_tip_mach,
            self.config.caution_advancing_tip_mach,
            self.config.maximum_advancing_tip_mach,
        );
        let severity = stall_margin.max(mach_margin);
        let cyclic_limit = if regime == RotorEdgeRegime::Normal {
            1.0
        } else {
            (1.0 - 0.8 * severity).clamp(f64::from(self.config.minimum_cyclic_limit), 1.0) as f32
        };
        let collective_limit = if matches!(
            regime,
            RotorEdgeRegime::RetreatingBladeStallExposure | RotorEdgeRegime::CombinedLimit
        ) {
            self.config.maximum_protected_collective
        } else {
            1.0
        };
        let mut protected_command = requested;
        protected_command.cyclic_lon = protected_command
            .cyclic_lon
            .clamp(-cyclic_limit, cyclic_limit);
        protected_command.cyclic_lat = protected_command
            .cyclic_lat
            .clamp(-cyclic_limit, cyclic_limit);
        protected_command.collective = protected_command.collective.min(collective_limit);

        Ok(RotorEdgeAssessment {
            regime,
            tip_speed_mps,
            advance_ratio,
            effective_stall_ratio,
            advancing_tip_mach,
            retreating_tip_speed_mps,
            cyclic_limit,
            collective_limit,
            protected_command: protected_command.clamped(),
        })
    }
}

fn normalized_margin(value: f64, caution: f64, limit: f64) -> f64 {
    ((value - caution) / (limit - caution)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(rpm: f64, airspeed: f64) -> RotorEdgeObservation {
        RotorEdgeObservation {
            main_rotor_rpm: rpm,
            horizontal_airspeed_mps: airspeed,
            outside_air_temperature_k: 288.15,
            collective_fraction: 0.3,
        }
    }

    #[test]
    fn hover_is_normal() {
        let protector = RotorEdgeRegimeProtector::new(RotorEdgeRegimeConfig::default()).unwrap();
        let result = protector
            .assess(HelicopterCommand::hover(), observation(3_300.0, 0.0))
            .unwrap();
        assert_eq!(result.regime, RotorEdgeRegime::Normal);
    }

    #[test]
    fn high_airspeed_exposes_edge_regime() {
        let protector = RotorEdgeRegimeProtector::new(RotorEdgeRegimeConfig::default()).unwrap();
        let result = protector
            .assess(HelicopterCommand::hover(), observation(3_300.0, 180.0))
            .unwrap();
        assert_ne!(result.regime, RotorEdgeRegime::Normal);
        assert!(result.cyclic_limit < 1.0);
    }

    #[test]
    fn cold_air_increases_tip_mach() {
        let protector = RotorEdgeRegimeProtector::new(RotorEdgeRegimeConfig::default()).unwrap();
        let mut warm = observation(3_300.0, 60.0);
        warm.outside_air_temperature_k = 310.0;
        let mut cold = warm;
        cold.outside_air_temperature_k = 230.0;
        let warm_result = protector.assess(HelicopterCommand::hover(), warm).unwrap();
        let cold_result = protector.assess(HelicopterCommand::hover(), cold).unwrap();
        assert!(cold_result.advancing_tip_mach > warm_result.advancing_tip_mach);
    }

    #[test]
    fn stopped_rotor_disarms_output() {
        let protector = RotorEdgeRegimeProtector::new(RotorEdgeRegimeConfig::default()).unwrap();
        let result = protector
            .assess(HelicopterCommand::hover(), observation(0.0, 0.0))
            .unwrap();
        assert_eq!(result.regime, RotorEdgeRegime::RotorSpeedUnavailable);
        assert_eq!(result.protected_command, HelicopterCommand::zero());
    }
}
