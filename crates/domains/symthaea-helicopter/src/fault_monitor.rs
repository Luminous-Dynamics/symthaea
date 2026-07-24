// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Debounced onboard fault detection and isolation.
//!
//! The monitor compares commanded/applied actuator state, expected rotor RPM,
//! delivered power, and measured yaw response. It emits explicit fault
//! evidence and a conservative actuator-health estimate for the allocator.
//! Detection is time-ordered and sample-debounced so one transient cannot
//! silently reconfigure the vehicle.

use serde::{Deserialize, Serialize};

use crate::control_allocation::ActuatorHealth;
use crate::types::{HelicopterCommand, HelicopterState};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HelicopterFaultKind {
    MainRotorUnderSpeed,
    TailRotorUnderSpeed,
    CollectiveServoMismatch,
    LongitudinalCyclicMismatch,
    LateralCyclicMismatch,
    PedalMismatch,
    EnginePowerDeficit,
    ExcessiveYawRate,
}

impl HelicopterFaultKind {
    const COUNT: usize = 8;

    const ALL: [Self; Self::COUNT] = [
        Self::MainRotorUnderSpeed,
        Self::TailRotorUnderSpeed,
        Self::CollectiveServoMismatch,
        Self::LongitudinalCyclicMismatch,
        Self::LateralCyclicMismatch,
        Self::PedalMismatch,
        Self::EnginePowerDeficit,
        Self::ExcessiveYawRate,
    ];

    const fn index(self) -> usize {
        match self {
            Self::MainRotorUnderSpeed => 0,
            Self::TailRotorUnderSpeed => 1,
            Self::CollectiveServoMismatch => 2,
            Self::LongitudinalCyclicMismatch => 3,
            Self::LateralCyclicMismatch => 4,
            Self::PedalMismatch => 5,
            Self::EnginePowerDeficit => 6,
            Self::ExcessiveYawRate => 7,
        }
    }

    pub const fn is_critical(self) -> bool {
        matches!(
            self,
            Self::MainRotorUnderSpeed
                | Self::TailRotorUnderSpeed
                | Self::EnginePowerDeficit
                | Self::ExcessiveYawRate
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FaultMonitorConfig {
    pub max_main_rpm_error_fraction: f64,
    pub max_tail_rpm_error_fraction: f64,
    pub max_servo_tracking_error: f64,
    pub minimum_power_delivery_fraction: f64,
    pub excessive_yaw_rate_rad_s: f64,
    pub assertion_samples: u32,
    pub recovery_samples: u32,
}

impl Default for FaultMonitorConfig {
    fn default() -> Self {
        Self {
            max_main_rpm_error_fraction: 0.25,
            max_tail_rpm_error_fraction: 0.35,
            max_servo_tracking_error: 0.15,
            minimum_power_delivery_fraction: 0.70,
            excessive_yaw_rate_rad_s: 1.5,
            assertion_samples: 5,
            recovery_samples: 20,
        }
    }
}

impl FaultMonitorConfig {
    pub fn validate(&self) -> bool {
        [
            self.max_main_rpm_error_fraction,
            self.max_tail_rpm_error_fraction,
            self.max_servo_tracking_error,
            self.minimum_power_delivery_fraction,
        ]
        .iter()
        .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
            && self.excessive_yaw_rate_rad_s.is_finite()
            && self.excessive_yaw_rate_rad_s > 0.0
            && self.assertion_samples > 0
            && self.recovery_samples > 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FlightHealthObservation {
    pub monotonic_time_s: f64,
    pub requested_command: HelicopterCommand,
    pub applied_command: HelicopterCommand,
    pub main_rotor_rpm: f64,
    pub tail_rotor_rpm: f64,
    pub power_delivery_fraction: f64,
    pub yaw_rate_rad_s: f64,
}

impl FlightHealthObservation {
    pub fn from_state(
        monotonic_time_s: f64,
        requested_command: HelicopterCommand,
        applied_command: HelicopterCommand,
        state: &HelicopterState,
        power_delivery_fraction: f64,
    ) -> Self {
        Self {
            monotonic_time_s,
            requested_command,
            applied_command,
            main_rotor_rpm: state.main_rotor_rpm,
            tail_rotor_rpm: state.tail_rotor_rpm,
            power_delivery_fraction,
            yaw_rate_rad_s: state.angular_velocity[2],
        }
    }

    fn is_finite(&self) -> bool {
        self.monotonic_time_s.is_finite()
            && self
                .requested_command
                .to_ctrl()
                .iter()
                .all(|value| value.is_finite())
            && self
                .applied_command
                .to_ctrl()
                .iter()
                .all(|value| value.is_finite())
            && self.main_rotor_rpm.is_finite()
            && self.tail_rotor_rpm.is_finite()
            && self.power_delivery_fraction.is_finite()
            && self.yaw_rate_rad_s.is_finite()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FaultStatus {
    pub kind: HelicopterFaultKind,
    pub active: bool,
    pub assertion_count: u32,
    pub recovery_count: u32,
    pub first_detected_s: Option<f64>,
    pub last_observed_s: Option<f64>,
}

impl FaultStatus {
    const fn new(kind: HelicopterFaultKind) -> Self {
        Self {
            kind,
            active: false,
            assertion_count: 0,
            recovery_count: 0,
            first_detected_s: None,
            last_observed_s: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FaultDiagnosis {
    pub active_faults: Vec<HelicopterFaultKind>,
    pub critical: bool,
    pub diagnosed_health: ActuatorHealth,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultMonitorError {
    InvalidConfiguration,
    NonFiniteObservation,
    TimeWentBackwards,
}

#[derive(Debug, Clone)]
pub struct HelicopterFaultMonitor {
    config: FaultMonitorConfig,
    status: [FaultStatus; HelicopterFaultKind::COUNT],
    last_time_s: Option<f64>,
}

impl Default for HelicopterFaultMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl HelicopterFaultMonitor {
    pub fn new() -> Self {
        Self {
            config: FaultMonitorConfig::default(),
            status: std::array::from_fn(|index| FaultStatus::new(HelicopterFaultKind::ALL[index])),
            last_time_s: None,
        }
    }

    pub fn with_config(config: FaultMonitorConfig) -> Result<Self, FaultMonitorError> {
        if !config.validate() {
            return Err(FaultMonitorError::InvalidConfiguration);
        }
        Ok(Self {
            config,
            ..Self::new()
        })
    }

    pub fn status(&self, kind: HelicopterFaultKind) -> FaultStatus {
        self.status[kind.index()]
    }

    pub fn diagnose(
        &mut self,
        observation: FlightHealthObservation,
    ) -> Result<FaultDiagnosis, FaultMonitorError> {
        if !self.config.validate() {
            return Err(FaultMonitorError::InvalidConfiguration);
        }
        if !observation.is_finite() {
            return Err(FaultMonitorError::NonFiniteObservation);
        }
        if self
            .last_time_s
            .is_some_and(|previous| observation.monotonic_time_s < previous)
        {
            return Err(FaultMonitorError::TimeWentBackwards);
        }
        self.last_time_s = Some(observation.monotonic_time_s);

        let expected_main_rpm = observation.applied_command.thrust as f64 * 5_500.0;
        let expected_tail_rpm = observation.applied_command.tail_rotor as f64 * 4_000.0;
        let main_under_speed = observation.applied_command.thrust > 0.30
            && observation.main_rotor_rpm
                < expected_main_rpm * (1.0 - self.config.max_main_rpm_error_fraction);
        let tail_under_speed = observation.applied_command.tail_rotor > 0.25
            && observation.tail_rotor_rpm
                < expected_tail_rpm * (1.0 - self.config.max_tail_rpm_error_fraction);
        let servo_error = |requested: f32, applied: f32| {
            (requested as f64 - applied as f64).abs() > self.config.max_servo_tracking_error
        };
        let detected = [
            main_under_speed,
            tail_under_speed,
            servo_error(
                observation.requested_command.collective,
                observation.applied_command.collective,
            ),
            servo_error(
                observation.requested_command.cyclic_lon,
                observation.applied_command.cyclic_lon,
            ),
            servo_error(
                observation.requested_command.cyclic_lat,
                observation.applied_command.cyclic_lat,
            ),
            servo_error(
                observation.requested_command.pedal,
                observation.applied_command.pedal,
            ),
            observation.requested_command.thrust > 0.30
                && observation.power_delivery_fraction
                    < self.config.minimum_power_delivery_fraction,
            observation.yaw_rate_rad_s.abs() > self.config.excessive_yaw_rate_rad_s,
        ];

        for (index, is_detected) in detected.into_iter().enumerate() {
            let status = &mut self.status[index];
            status.last_observed_s = Some(observation.monotonic_time_s);
            if is_detected {
                status.assertion_count = status.assertion_count.saturating_add(1);
                status.recovery_count = 0;
                if status.assertion_count >= self.config.assertion_samples {
                    if !status.active {
                        status.first_detected_s = Some(observation.monotonic_time_s);
                    }
                    status.active = true;
                }
            } else {
                status.assertion_count = 0;
                if status.active {
                    status.recovery_count = status.recovery_count.saturating_add(1);
                    if status.recovery_count >= self.config.recovery_samples {
                        status.active = false;
                        status.recovery_count = 0;
                        status.first_detected_s = None;
                    }
                }
            }
        }

        Ok(self.current_diagnosis())
    }

    pub fn current_diagnosis(&self) -> FaultDiagnosis {
        let active_faults: Vec<_> = self
            .status
            .iter()
            .filter(|status| status.active)
            .map(|status| status.kind)
            .collect();
        let mut health = ActuatorHealth::default();
        for kind in &active_faults {
            match kind {
                HelicopterFaultKind::MainRotorUnderSpeed => health.main_rotor = 0.4,
                HelicopterFaultKind::TailRotorUnderSpeed => {
                    health.tail_rotor = 0.0;
                    health.pedal = health.pedal.min(0.5);
                }
                HelicopterFaultKind::CollectiveServoMismatch => health.collective = 0.5,
                HelicopterFaultKind::LongitudinalCyclicMismatch => health.cyclic_lon = 0.5,
                HelicopterFaultKind::LateralCyclicMismatch => health.cyclic_lat = 0.5,
                HelicopterFaultKind::PedalMismatch => health.pedal = 0.5,
                HelicopterFaultKind::EnginePowerDeficit => {
                    health.main_rotor = health.main_rotor.min(0.5)
                }
                HelicopterFaultKind::ExcessiveYawRate => {
                    health.tail_rotor = health.tail_rotor.min(0.25);
                    health.pedal = health.pedal.min(0.25);
                }
            }
        }
        FaultDiagnosis {
            critical: active_faults.iter().any(|kind| kind.is_critical()),
            active_faults,
            diagnosed_health: health,
        }
    }

    pub fn reset(&mut self) {
        self.status =
            std::array::from_fn(|index| FaultStatus::new(HelicopterFaultKind::ALL[index]));
        self.last_time_s = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nominal_observation(time: f64) -> FlightHealthObservation {
        FlightHealthObservation {
            monotonic_time_s: time,
            requested_command: HelicopterCommand::hover(),
            applied_command: HelicopterCommand::hover(),
            main_rotor_rpm: 3_300.0,
            tail_rotor_rpm: 2_000.0,
            power_delivery_fraction: 1.0,
            yaw_rate_rad_s: 0.0,
        }
    }

    #[test]
    fn faults_require_debounced_evidence() {
        let mut monitor = HelicopterFaultMonitor::with_config(FaultMonitorConfig {
            assertion_samples: 3,
            ..FaultMonitorConfig::default()
        })
        .unwrap();
        for sample in 0..2 {
            let mut observation = nominal_observation(sample as f64);
            observation.main_rotor_rpm = 500.0;
            assert!(!monitor.diagnose(observation).unwrap().critical);
        }
        let mut observation = nominal_observation(2.0);
        observation.main_rotor_rpm = 500.0;
        let diagnosis = monitor.diagnose(observation).unwrap();
        assert!(diagnosis.critical);
        assert!(
            diagnosis
                .active_faults
                .contains(&HelicopterFaultKind::MainRotorUnderSpeed)
        );
        assert!(diagnosis.diagnosed_health.main_rotor < 1.0);
    }

    #[test]
    fn active_faults_clear_only_after_recovery_window() {
        let mut monitor = HelicopterFaultMonitor::with_config(FaultMonitorConfig {
            assertion_samples: 1,
            recovery_samples: 2,
            ..FaultMonitorConfig::default()
        })
        .unwrap();
        let mut failed = nominal_observation(0.0);
        failed.tail_rotor_rpm = 100.0;
        assert!(monitor.diagnose(failed).unwrap().critical);
        assert!(monitor.diagnose(nominal_observation(1.0)).unwrap().critical);
        assert!(!monitor.diagnose(nominal_observation(2.0)).unwrap().critical);
    }

    #[test]
    fn time_regression_fails_closed() {
        let mut monitor = HelicopterFaultMonitor::new();
        monitor.diagnose(nominal_observation(2.0)).unwrap();
        assert_eq!(
            monitor.diagnose(nominal_observation(1.0)),
            Err(FaultMonitorError::TimeWentBackwards)
        );
    }
}
