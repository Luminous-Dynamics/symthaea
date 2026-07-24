// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fault-tolerant control reconfiguration policy.
//!
//! Detection and allocation alone do not define how the aircraft changes its
//! control objectives after losing authority. This module converts conservative
//! actuator-health evidence into an explicit reconfiguration mode and applies
//! mode-specific command limits before the ordinary allocator and actuator model.

use serde::{Deserialize, Serialize};

use crate::control_allocation::ActuatorHealth;
use crate::types::HelicopterCommand;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ControlReconfigurationConfig {
    pub healthy_threshold: f64,
    pub degraded_threshold: f64,
    pub lost_threshold: f64,
    pub maximum_degraded_cyclic: f32,
    pub maximum_yaw_limited_pedal: f32,
    pub maximum_emergency_collective: f32,
    pub assertion_updates: u32,
    pub recovery_updates: u32,
}

impl Default for ControlReconfigurationConfig {
    fn default() -> Self {
        Self {
            healthy_threshold: 0.85,
            degraded_threshold: 0.45,
            lost_threshold: 0.10,
            maximum_degraded_cyclic: 0.35,
            maximum_yaw_limited_pedal: 0.10,
            maximum_emergency_collective: 0.35,
            assertion_updates: 2,
            recovery_updates: 5,
        }
    }
}

impl ControlReconfigurationConfig {
    pub fn validate(&self) -> Result<(), ControlReconfigurationError> {
        if !self.healthy_threshold.is_finite()
            || !self.degraded_threshold.is_finite()
            || !self.lost_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.healthy_threshold)
            || !(0.0..=1.0).contains(&self.degraded_threshold)
            || !(0.0..=1.0).contains(&self.lost_threshold)
            || !(self.lost_threshold < self.degraded_threshold
                && self.degraded_threshold < self.healthy_threshold)
            || !self.maximum_degraded_cyclic.is_finite()
            || !(0.0..=1.0).contains(&self.maximum_degraded_cyclic)
            || !self.maximum_yaw_limited_pedal.is_finite()
            || !(0.0..=1.0).contains(&self.maximum_yaw_limited_pedal)
            || !self.maximum_emergency_collective.is_finite()
            || !(0.0..=1.0).contains(&self.maximum_emergency_collective)
            || self.assertion_updates == 0
            || self.recovery_updates == 0
        {
            return Err(ControlReconfigurationError::InvalidConfiguration);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlReconfigurationMode {
    Nominal,
    DegradedEnvelope,
    YawLimited,
    LongitudinalOnly,
    LateralOnly,
    Autorotation,
    LandImmediately,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReconfigurationReason {
    MainRotorAuthorityLost,
    TailRotorAuthorityLost,
    CollectiveAuthorityLost,
    LongitudinalCyclicLost,
    LateralCyclicLost,
    MultipleAxesDegraded,
    CriticalFault,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ControlReconfigurationResult {
    pub mode: ControlReconfigurationMode,
    pub reasons: Vec<ReconfigurationReason>,
    pub protected_command: HelicopterCommand,
    pub translation_allowed: bool,
    pub yaw_tracking_allowed: bool,
    pub assertion_count: u32,
    pub recovery_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlReconfigurationError {
    InvalidConfiguration,
    InvalidActuatorHealth,
}

#[derive(Debug, Clone)]
pub struct ControlReconfigurationManager {
    config: ControlReconfigurationConfig,
    active_mode: ControlReconfigurationMode,
    candidate_mode: ControlReconfigurationMode,
    assertion_count: u32,
    recovery_count: u32,
}

impl ControlReconfigurationManager {
    pub fn new(config: ControlReconfigurationConfig) -> Result<Self, ControlReconfigurationError> {
        config.validate()?;
        Ok(Self {
            config,
            active_mode: ControlReconfigurationMode::Nominal,
            candidate_mode: ControlReconfigurationMode::Nominal,
            assertion_count: 0,
            recovery_count: 0,
        })
    }

    pub fn active_mode(&self) -> ControlReconfigurationMode {
        self.active_mode
    }

    pub fn reconfigure(
        &mut self,
        requested: HelicopterCommand,
        health: ActuatorHealth,
        critical_fault: bool,
    ) -> Result<ControlReconfigurationResult, ControlReconfigurationError> {
        self.config.validate()?;
        health
            .validate()
            .map_err(|_| ControlReconfigurationError::InvalidActuatorHealth)?;
        let (candidate, reasons) = classify(&self.config, health, critical_fault);
        if candidate == self.active_mode {
            self.candidate_mode = candidate;
            self.assertion_count = 0;
            self.recovery_count = 0;
        } else if severity(candidate) > severity(self.active_mode) {
            if candidate != self.candidate_mode {
                self.candidate_mode = candidate;
                self.assertion_count = 0;
            }
            self.assertion_count = self.assertion_count.saturating_add(1);
            self.recovery_count = 0;
            if self.assertion_count >= self.config.assertion_updates {
                self.active_mode = candidate;
                self.assertion_count = 0;
            }
        } else {
            self.recovery_count = self.recovery_count.saturating_add(1);
            self.assertion_count = 0;
            if self.recovery_count >= self.config.recovery_updates {
                self.active_mode = candidate;
                self.candidate_mode = candidate;
                self.recovery_count = 0;
            }
        }

        let protected_command = protect_command(self.config, self.active_mode, requested);
        Ok(ControlReconfigurationResult {
            mode: self.active_mode,
            reasons,
            protected_command,
            translation_allowed: matches!(
                self.active_mode,
                ControlReconfigurationMode::Nominal | ControlReconfigurationMode::DegradedEnvelope
            ),
            yaw_tracking_allowed: !matches!(
                self.active_mode,
                ControlReconfigurationMode::YawLimited
                    | ControlReconfigurationMode::Autorotation
                    | ControlReconfigurationMode::LandImmediately
            ),
            assertion_count: self.assertion_count,
            recovery_count: self.recovery_count,
        })
    }
}

fn classify(
    config: &ControlReconfigurationConfig,
    health: ActuatorHealth,
    critical_fault: bool,
) -> (ControlReconfigurationMode, Vec<ReconfigurationReason>) {
    let mut reasons = Vec::new();
    if critical_fault {
        reasons.push(ReconfigurationReason::CriticalFault);
    }
    if health.main_rotor <= config.lost_threshold {
        reasons.push(ReconfigurationReason::MainRotorAuthorityLost);
        return (ControlReconfigurationMode::Autorotation, reasons);
    }
    if health.collective <= config.lost_threshold {
        reasons.push(ReconfigurationReason::CollectiveAuthorityLost);
        return (ControlReconfigurationMode::LandImmediately, reasons);
    }
    if critical_fault {
        return (ControlReconfigurationMode::LandImmediately, reasons);
    }
    if health.tail_rotor <= config.lost_threshold || health.pedal <= config.lost_threshold {
        reasons.push(ReconfigurationReason::TailRotorAuthorityLost);
        return (ControlReconfigurationMode::YawLimited, reasons);
    }
    let lon_lost = health.cyclic_lon <= config.lost_threshold;
    let lat_lost = health.cyclic_lat <= config.lost_threshold;
    if lon_lost {
        reasons.push(ReconfigurationReason::LongitudinalCyclicLost);
    }
    if lat_lost {
        reasons.push(ReconfigurationReason::LateralCyclicLost);
    }
    if lon_lost && lat_lost {
        reasons.push(ReconfigurationReason::MultipleAxesDegraded);
        return (ControlReconfigurationMode::LandImmediately, reasons);
    }
    if lon_lost {
        return (ControlReconfigurationMode::LateralOnly, reasons);
    }
    if lat_lost {
        return (ControlReconfigurationMode::LongitudinalOnly, reasons);
    }
    let minimum = [
        health.collective,
        health.cyclic_lon,
        health.cyclic_lat,
        health.pedal,
        health.main_rotor,
        health.tail_rotor,
    ]
    .into_iter()
    .fold(1.0_f64, f64::min);
    if minimum < config.healthy_threshold {
        reasons.push(ReconfigurationReason::MultipleAxesDegraded);
        (ControlReconfigurationMode::DegradedEnvelope, reasons)
    } else {
        (ControlReconfigurationMode::Nominal, reasons)
    }
}

fn protect_command(
    config: ControlReconfigurationConfig,
    mode: ControlReconfigurationMode,
    mut command: HelicopterCommand,
) -> HelicopterCommand {
    match mode {
        ControlReconfigurationMode::Nominal => {}
        ControlReconfigurationMode::DegradedEnvelope => {
            command.cyclic_lon = command.cyclic_lon.clamp(
                -config.maximum_degraded_cyclic,
                config.maximum_degraded_cyclic,
            );
            command.cyclic_lat = command.cyclic_lat.clamp(
                -config.maximum_degraded_cyclic,
                config.maximum_degraded_cyclic,
            );
        }
        ControlReconfigurationMode::YawLimited => {
            command.pedal = command.pedal.clamp(
                -config.maximum_yaw_limited_pedal,
                config.maximum_yaw_limited_pedal,
            );
            command.collective = command.collective.min(config.maximum_emergency_collective);
        }
        ControlReconfigurationMode::LongitudinalOnly => {
            command.cyclic_lat = 0.0;
            command.cyclic_lon = command.cyclic_lon.clamp(
                -config.maximum_degraded_cyclic,
                config.maximum_degraded_cyclic,
            );
        }
        ControlReconfigurationMode::LateralOnly => {
            command.cyclic_lon = 0.0;
            command.cyclic_lat = command.cyclic_lat.clamp(
                -config.maximum_degraded_cyclic,
                config.maximum_degraded_cyclic,
            );
        }
        ControlReconfigurationMode::Autorotation => {
            command.thrust = 0.0;
            command.tail_rotor = 0.0;
            command.pedal = 0.0;
            command.collective = command.collective.min(0.0);
            command.cyclic_lon = command.cyclic_lon.clamp(-0.20, 0.20);
            command.cyclic_lat = command.cyclic_lat.clamp(-0.20, 0.20);
        }
        ControlReconfigurationMode::LandImmediately => {
            command.cyclic_lon = command.cyclic_lon.clamp(-0.10, 0.10);
            command.cyclic_lat = command.cyclic_lat.clamp(-0.10, 0.10);
            command.pedal = command.pedal.clamp(-0.10, 0.10);
            command.collective = command.collective.min(config.maximum_emergency_collective);
        }
    }
    command.clamped()
}

const fn severity(mode: ControlReconfigurationMode) -> u8 {
    match mode {
        ControlReconfigurationMode::Nominal => 0,
        ControlReconfigurationMode::DegradedEnvelope => 1,
        ControlReconfigurationMode::YawLimited
        | ControlReconfigurationMode::LongitudinalOnly
        | ControlReconfigurationMode::LateralOnly => 2,
        ControlReconfigurationMode::LandImmediately => 3,
        ControlReconfigurationMode::Autorotation => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn main_rotor_loss_enters_autorotation_after_debounce() {
        let mut manager =
            ControlReconfigurationManager::new(ControlReconfigurationConfig::default()).unwrap();
        let health = ActuatorHealth {
            main_rotor: 0.0,
            ..ActuatorHealth::default()
        };
        manager
            .reconfigure(HelicopterCommand::hover(), health, false)
            .unwrap();
        let result = manager
            .reconfigure(HelicopterCommand::hover(), health, false)
            .unwrap();
        assert_eq!(result.mode, ControlReconfigurationMode::Autorotation);
        assert_eq!(result.protected_command.thrust, 0.0);
    }

    #[test]
    fn tail_loss_limits_yaw_and_collective() {
        let config = ControlReconfigurationConfig::default();
        let mut manager = ControlReconfigurationManager::new(config).unwrap();
        let health = ActuatorHealth {
            tail_rotor: 0.0,
            pedal: 0.0,
            ..ActuatorHealth::default()
        };
        manager
            .reconfigure(HelicopterCommand::hover(), health, false)
            .unwrap();
        let result = manager
            .reconfigure(
                HelicopterCommand {
                    pedal: 1.0,
                    collective: 0.9,
                    ..HelicopterCommand::hover()
                },
                health,
                false,
            )
            .unwrap();
        assert_eq!(result.mode, ControlReconfigurationMode::YawLimited);
        assert!(result.protected_command.pedal <= config.maximum_yaw_limited_pedal);
        assert!(result.protected_command.collective <= config.maximum_emergency_collective);
    }

    #[test]
    fn lateral_loss_zeros_lateral_command() {
        let mut manager =
            ControlReconfigurationManager::new(ControlReconfigurationConfig::default()).unwrap();
        let health = ActuatorHealth {
            cyclic_lat: 0.0,
            ..ActuatorHealth::default()
        };
        manager
            .reconfigure(HelicopterCommand::hover(), health, false)
            .unwrap();
        let result = manager
            .reconfigure(
                HelicopterCommand {
                    cyclic_lat: 0.8,
                    cyclic_lon: 0.2,
                    ..HelicopterCommand::hover()
                },
                health,
                false,
            )
            .unwrap();
        assert_eq!(result.mode, ControlReconfigurationMode::LongitudinalOnly);
        assert_eq!(result.protected_command.cyclic_lat, 0.0);
    }

    #[test]
    fn invalid_health_fails_closed() {
        let mut manager =
            ControlReconfigurationManager::new(ControlReconfigurationConfig::default()).unwrap();
        let health = ActuatorHealth {
            collective: f64::NAN,
            ..ActuatorHealth::default()
        };
        assert_eq!(
            manager.reconfigure(HelicopterCommand::hover(), health, false),
            Err(ControlReconfigurationError::InvalidActuatorHealth)
        );
    }
}
