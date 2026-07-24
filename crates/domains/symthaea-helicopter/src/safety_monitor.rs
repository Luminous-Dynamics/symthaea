// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Runtime safety-property and bounded-response monitor.
//!
//! This module checks instantaneous invariants and small temporal obligations:
//! critical faults must lead to an emergency directive within a deadline, and a
//! confirmed landing must lead to disarmed output. The monitor is deliberately
//! simple and deterministic so violations can be replayed from flight evidence.

use serde::{Deserialize, Serialize};

use crate::mission_supervisor::MissionDirective;
use crate::types::{HelicopterCommand, HelicopterState};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SafetyMonitorConfig {
    pub quaternion_norm_tolerance: f64,
    pub below_ground_tolerance_m: f64,
    pub maximum_main_rotor_rpm: f64,
    pub maximum_tail_rotor_rpm: f64,
    pub critical_fault_response_deadline_s: f64,
    pub landing_disarm_deadline_s: f64,
    pub disarmed_thrust_tolerance: f32,
}

impl Default for SafetyMonitorConfig {
    fn default() -> Self {
        Self {
            quaternion_norm_tolerance: 1.0e-3,
            below_ground_tolerance_m: 1.0e-6,
            maximum_main_rotor_rpm: 5_700.0,
            maximum_tail_rotor_rpm: 4_200.0,
            critical_fault_response_deadline_s: 0.25,
            landing_disarm_deadline_s: 0.5,
            disarmed_thrust_tolerance: 0.01,
        }
    }
}

impl SafetyMonitorConfig {
    pub fn validate(&self) -> Result<(), SafetyMonitorError> {
        if !self.quaternion_norm_tolerance.is_finite()
            || self.quaternion_norm_tolerance <= 0.0
            || !self.below_ground_tolerance_m.is_finite()
            || self.below_ground_tolerance_m < 0.0
            || !self.maximum_main_rotor_rpm.is_finite()
            || self.maximum_main_rotor_rpm <= 0.0
            || !self.maximum_tail_rotor_rpm.is_finite()
            || self.maximum_tail_rotor_rpm <= 0.0
            || !self.critical_fault_response_deadline_s.is_finite()
            || self.critical_fault_response_deadline_s < 0.0
            || !self.landing_disarm_deadline_s.is_finite()
            || self.landing_disarm_deadline_s < 0.0
            || !self.disarmed_thrust_tolerance.is_finite()
            || !(0.0..=1.0).contains(&self.disarmed_thrust_tolerance)
        {
            return Err(SafetyMonitorError::InvalidConfiguration);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyMonitorError {
    InvalidConfiguration,
    NonFiniteTime,
    TimeWentBackwards,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SafetyProperty {
    FiniteState,
    UnitQuaternion,
    CommandBounds,
    RotorSpeedBounds,
    NonNegativeAltitude,
    AuthorityFailClosed,
    CriticalFaultResponse,
    LandingDisarm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SafetyAssessmentStatus {
    Satisfied,
    Pending,
    Violated,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SafetyViolation {
    pub property: SafetyProperty,
    pub observed_at_s: f64,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct RuntimeSafetySnapshot {
    pub monotonic_time_s: f64,
    pub state: HelicopterState,
    pub applied_command: HelicopterCommand,
    pub directive: MissionDirective,
    pub authority_valid: bool,
    pub critical_flight_fault: bool,
    pub landed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SafetyAssessment {
    pub status: SafetyAssessmentStatus,
    pub pending_properties: Vec<SafetyProperty>,
    pub violations: Vec<SafetyViolation>,
    pub critical_fault_obligation_age_s: Option<f64>,
    pub landing_disarm_obligation_age_s: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct RuntimeSafetyMonitor {
    config: SafetyMonitorConfig,
    last_time_s: Option<f64>,
    critical_fault_since_s: Option<f64>,
    landed_since_s: Option<f64>,
}

impl RuntimeSafetyMonitor {
    pub fn new(config: SafetyMonitorConfig) -> Result<Self, SafetyMonitorError> {
        config.validate()?;
        Ok(Self {
            config,
            last_time_s: None,
            critical_fault_since_s: None,
            landed_since_s: None,
        })
    }

    pub fn evaluate(
        &mut self,
        snapshot: &RuntimeSafetySnapshot,
    ) -> Result<SafetyAssessment, SafetyMonitorError> {
        self.config.validate()?;
        let now_s = snapshot.monotonic_time_s;
        if !now_s.is_finite() {
            return Err(SafetyMonitorError::NonFiniteTime);
        }
        if self.last_time_s.is_some_and(|last| now_s < last) {
            return Err(SafetyMonitorError::TimeWentBackwards);
        }
        self.last_time_s = Some(now_s);

        let mut violations = Vec::new();
        let mut pending = Vec::new();
        let mut violate = |property, detail: &str| {
            violations.push(SafetyViolation {
                property,
                observed_at_s: now_s,
                detail: detail.to_string(),
            });
        };

        if !snapshot.state.is_finite() {
            violate(
                SafetyProperty::FiniteState,
                "helicopter state contains NaN or infinity",
            );
        }
        let quaternion_norm = snapshot
            .state
            .quaternion
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if !quaternion_norm.is_finite()
            || (quaternion_norm - 1.0).abs() > self.config.quaternion_norm_tolerance
        {
            violate(
                SafetyProperty::UnitQuaternion,
                "attitude quaternion is not normalized",
            );
        }
        if !command_is_bounded(snapshot.applied_command) {
            violate(
                SafetyProperty::CommandBounds,
                "applied command exceeds normalized bounds",
            );
        }
        if snapshot.state.main_rotor_rpm < 0.0
            || snapshot.state.main_rotor_rpm > self.config.maximum_main_rotor_rpm
            || snapshot.state.tail_rotor_rpm < 0.0
            || snapshot.state.tail_rotor_rpm > self.config.maximum_tail_rotor_rpm
        {
            violate(
                SafetyProperty::RotorSpeedBounds,
                "rotor speed is outside monitored bounds",
            );
        }
        if snapshot.state.altitude() < -self.config.below_ground_tolerance_m {
            violate(
                SafetyProperty::NonNegativeAltitude,
                "vehicle state is below the ground plane",
            );
        }
        if !snapshot.authority_valid && snapshot.directive == MissionDirective::Continue {
            violate(
                SafetyProperty::AuthorityFailClosed,
                "mission continued after authority became invalid",
            );
        }

        if snapshot.critical_flight_fault {
            let since = *self.critical_fault_since_s.get_or_insert(now_s);
            let age = now_s - since;
            let responding = matches!(
                snapshot.directive,
                MissionDirective::EmergencyLand
                    | MissionDirective::DivertToLandingZone
                    | MissionDirective::DisarmAfterLanding
            );
            if responding {
                self.critical_fault_since_s = None;
            } else if age > self.config.critical_fault_response_deadline_s {
                violate(
                    SafetyProperty::CriticalFaultResponse,
                    "critical fault did not produce an emergency directive before deadline",
                );
            } else {
                pending.push(SafetyProperty::CriticalFaultResponse);
            }
        } else {
            self.critical_fault_since_s = None;
        }

        if snapshot.landed {
            let since = *self.landed_since_s.get_or_insert(now_s);
            let age = now_s - since;
            let disarmed = snapshot.directive == MissionDirective::DisarmAfterLanding
                && snapshot.applied_command.thrust.abs() <= self.config.disarmed_thrust_tolerance
                && snapshot.applied_command.tail_rotor.abs()
                    <= self.config.disarmed_thrust_tolerance;
            if disarmed {
                self.landed_since_s = None;
            } else if age > self.config.landing_disarm_deadline_s {
                violate(
                    SafetyProperty::LandingDisarm,
                    "landed vehicle retained armed rotor output beyond deadline",
                );
            } else {
                pending.push(SafetyProperty::LandingDisarm);
            }
        } else {
            self.landed_since_s = None;
        }

        let status = if !violations.is_empty() {
            SafetyAssessmentStatus::Violated
        } else if !pending.is_empty() {
            SafetyAssessmentStatus::Pending
        } else {
            SafetyAssessmentStatus::Satisfied
        };
        Ok(SafetyAssessment {
            status,
            pending_properties: pending,
            violations,
            critical_fault_obligation_age_s: self.critical_fault_since_s.map(|since| now_s - since),
            landing_disarm_obligation_age_s: self.landed_since_s.map(|since| now_s - since),
        })
    }

    pub fn reset(&mut self) {
        self.last_time_s = None;
        self.critical_fault_since_s = None;
        self.landed_since_s = None;
    }
}

impl Default for RuntimeSafetyMonitor {
    fn default() -> Self {
        Self::new(SafetyMonitorConfig::default())
            .expect("default safety monitor config must remain valid")
    }
}

fn command_is_bounded(command: HelicopterCommand) -> bool {
    command.collective.is_finite()
        && (-1.0..=1.0).contains(&command.collective)
        && command.cyclic_lon.is_finite()
        && (-1.0..=1.0).contains(&command.cyclic_lon)
        && command.cyclic_lat.is_finite()
        && (-1.0..=1.0).contains(&command.cyclic_lat)
        && command.pedal.is_finite()
        && (-1.0..=1.0).contains(&command.pedal)
        && command.thrust.is_finite()
        && (0.0..=1.0).contains(&command.thrust)
        && command.tail_rotor.is_finite()
        && (0.0..=1.0).contains(&command.tail_rotor)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(time_s: f64) -> RuntimeSafetySnapshot {
        RuntimeSafetySnapshot {
            monotonic_time_s: time_s,
            state: HelicopterState::hover(20.0),
            applied_command: HelicopterCommand::hover(),
            directive: MissionDirective::Continue,
            authority_valid: true,
            critical_flight_fault: false,
            landed: false,
        }
    }

    #[test]
    fn nominal_snapshot_satisfies_all_properties() {
        let assessment = RuntimeSafetyMonitor::default()
            .evaluate(&snapshot(0.0))
            .unwrap();
        assert_eq!(assessment.status, SafetyAssessmentStatus::Satisfied);
        assert!(assessment.violations.is_empty());
    }

    #[test]
    fn critical_fault_creates_then_violates_bounded_obligation() {
        let mut monitor = RuntimeSafetyMonitor::default();
        let mut first = snapshot(1.0);
        first.critical_flight_fault = true;
        assert_eq!(
            monitor.evaluate(&first).unwrap().status,
            SafetyAssessmentStatus::Pending
        );
        let mut late = first.clone();
        late.monotonic_time_s = 1.3;
        let assessment = monitor.evaluate(&late).unwrap();
        assert_eq!(assessment.status, SafetyAssessmentStatus::Violated);
        assert!(
            assessment
                .violations
                .iter()
                .any(|violation| violation.property == SafetyProperty::CriticalFaultResponse)
        );
    }

    #[test]
    fn emergency_directive_discharge_fault_obligation() {
        let mut monitor = RuntimeSafetyMonitor::default();
        let mut fault = snapshot(1.0);
        fault.critical_flight_fault = true;
        monitor.evaluate(&fault).unwrap();
        fault.monotonic_time_s = 1.1;
        fault.directive = MissionDirective::EmergencyLand;
        let assessment = monitor.evaluate(&fault).unwrap();
        assert_eq!(assessment.status, SafetyAssessmentStatus::Satisfied);
        assert_eq!(assessment.critical_fault_obligation_age_s, None);
    }

    #[test]
    fn invalid_authority_cannot_continue() {
        let mut invalid = snapshot(0.0);
        invalid.authority_valid = false;
        let assessment = RuntimeSafetyMonitor::default().evaluate(&invalid).unwrap();
        assert!(
            assessment
                .violations
                .iter()
                .any(|violation| violation.property == SafetyProperty::AuthorityFailClosed)
        );
    }

    #[test]
    fn landing_requires_disarm_within_deadline() {
        let mut monitor = RuntimeSafetyMonitor::default();
        let mut landed = snapshot(2.0);
        landed.landed = true;
        assert_eq!(
            monitor.evaluate(&landed).unwrap().status,
            SafetyAssessmentStatus::Pending
        );
        landed.monotonic_time_s = 2.6;
        assert_eq!(
            monitor.evaluate(&landed).unwrap().status,
            SafetyAssessmentStatus::Violated
        );
    }
}
