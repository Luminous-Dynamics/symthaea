// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent runtime invariants over the final physical command.
//!
//! These checks run after learned control, operator constraints, recovery
//! planning, field envelopes, maintenance derating, and actuator isolation.
//! A violation is therefore evidence of an authority-ordering defect rather
//! than another policy preference. Enforcement is monotonic: it may remove
//! authority, but never add productive motion.

use crate::actuator_isolation::{ActuatorIsolationReport, PhysicalActuator};
use crate::capability_profile::CapabilityDisposition;
use crate::embodiment::MotorSafetyLevel;
use crate::safety::SubterraneanHazard;
use crate::types::{SubterraneanCommand, SubterraneanState};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RuntimeInvariant {
    CommandFiniteAndBounded,
    RedTierRemovesProductiveWork,
    TunnelConflictStopsMotion,
    ReturnReserveRemovesProductiveWork,
    IsolatedActuatorHasNoAuthority,
    HoldDispositionStopsMotion,
    SensorFaultRemovesProductiveWork,
}

impl RuntimeInvariant {
    pub const ALL: [Self; 7] = [
        Self::CommandFiniteAndBounded,
        Self::RedTierRemovesProductiveWork,
        Self::TunnelConflictStopsMotion,
        Self::ReturnReserveRemovesProductiveWork,
        Self::IsolatedActuatorHasNoAuthority,
        Self::HoldDispositionStopsMotion,
        Self::SensorFaultRemovesProductiveWork,
    ];

    pub const fn code(self) -> &'static str {
        match self {
            Self::CommandFiniteAndBounded => "INV-CMD-001",
            Self::RedTierRemovesProductiveWork => "INV-SAF-001",
            Self::TunnelConflictStopsMotion => "INV-COL-001",
            Self::ReturnReserveRemovesProductiveWork => "INV-RET-001",
            Self::IsolatedActuatorHasNoAuthority => "INV-ACT-001",
            Self::HoldDispositionStopsMotion => "INV-HLD-001",
            Self::SensorFaultRemovesProductiveWork => "INV-SEN-001",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct InvariantContext<'a> {
    pub state: &'a SubterraneanState,
    pub safety_level: MotorSafetyLevel,
    pub primary_hazard: SubterraneanHazard,
    pub tunnel_conflict: bool,
    pub return_feasible: bool,
    pub capability_disposition: CapabilityDisposition,
    pub actuator_isolation: ActuatorIsolationReport,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InvariantAssessment {
    pub violations: Vec<RuntimeInvariant>,
    pub command_modified: bool,
    pub total_breaches: u64,
    pub consecutive_breach_frames: u32,
}

impl InvariantAssessment {
    pub fn nominal(total_breaches: u64) -> Self {
        Self {
            violations: Vec::new(),
            command_modified: false,
            total_breaches,
            consecutive_breach_frames: 0,
        }
    }

    pub fn passed(&self) -> bool {
        self.violations.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RuntimeInvariantMonitor {
    total_breaches: u64,
    consecutive_breach_frames: u32,
    last_violations: Vec<RuntimeInvariant>,
}

impl RuntimeInvariantMonitor {
    const EPSILON: f32 = 1e-5;

    fn productive(command: &SubterraneanCommand) -> bool {
        command.cutter_head().abs() > Self::EPSILON || command.auger_feed().abs() > Self::EPSILON
    }

    fn moving(command: &SubterraneanCommand) -> bool {
        command.left_track().abs() > Self::EPSILON
            || command.right_track().abs() > Self::EPSILON
            || Self::productive(command)
    }

    fn command_is_valid(command: &SubterraneanCommand) -> bool {
        command
            .torques
            .iter()
            .all(|value| value.is_finite() && (-1.0..=1.0).contains(value))
            && [
                command.recovery.dewatering_pump,
                command.recovery.sealant_injector,
                command.recovery.relay_deployer,
                command.recovery.roof_support,
            ]
            .into_iter()
            .all(|value| value.is_finite() && (0.0..=1.0).contains(&value))
    }

    fn isolated_actuator_demanded(
        command: &SubterraneanCommand,
        isolation: ActuatorIsolationReport,
    ) -> bool {
        PhysicalActuator::ALL.into_iter().any(|actuator| {
            if !isolation.is_isolated(actuator) {
                return false;
            }
            let demand = match actuator {
                PhysicalActuator::Cutter => command.cutter_head().abs(),
                PhysicalActuator::Auger => command.auger_feed().abs(),
                PhysicalActuator::LeftTrack => command.left_track().abs(),
                PhysicalActuator::RightTrack => command.right_track().abs(),
                PhysicalActuator::Ballast => command.ballast_trim().abs(),
                PhysicalActuator::ThermalPump => command.thermal_pump().abs(),
                PhysicalActuator::DewateringPump => command.recovery.dewatering_pump.abs(),
                PhysicalActuator::SealantInjector => command.recovery.sealant_injector.abs(),
                PhysicalActuator::RelayDeployer => command.recovery.relay_deployer.abs(),
                PhysicalActuator::RoofSupport => command.recovery.roof_support.abs(),
            };
            demand > Self::EPSILON
        })
    }

    fn remove_productive_work(command: &mut SubterraneanCommand) {
        command.set_cutter_head(0.0);
        command.set_auger_feed(0.0);
    }

    fn stop_motion(command: &mut SubterraneanCommand) {
        Self::remove_productive_work(command);
        command.set_left_track(0.0);
        command.set_right_track(0.0);
        command.set_ballast_trim(0.0);
    }

    pub fn enforce(
        &mut self,
        mut command: SubterraneanCommand,
        context: InvariantContext<'_>,
    ) -> (SubterraneanCommand, InvariantAssessment) {
        let mut violations = Vec::new();
        let original = command;

        if !Self::command_is_valid(&command) {
            violations.push(RuntimeInvariant::CommandFiniteAndBounded);
            command.sanitize();
        }
        if context.safety_level == MotorSafetyLevel::Red && Self::productive(&command) {
            violations.push(RuntimeInvariant::RedTierRemovesProductiveWork);
            Self::remove_productive_work(&mut command);
        }
        if context.tunnel_conflict && Self::moving(&command) {
            violations.push(RuntimeInvariant::TunnelConflictStopsMotion);
            Self::stop_motion(&mut command);
        }
        if !context.return_feasible && Self::productive(&command) {
            violations.push(RuntimeInvariant::ReturnReserveRemovesProductiveWork);
            Self::remove_productive_work(&mut command);
        }
        if Self::isolated_actuator_demanded(&command, context.actuator_isolation) {
            violations.push(RuntimeInvariant::IsolatedActuatorHasNoAuthority);
            command = Self::remove_isolated_authority(command, context.actuator_isolation);
        }
        if context.capability_disposition == CapabilityDisposition::HoldForRecovery
            && Self::moving(&command)
        {
            violations.push(RuntimeInvariant::HoldDispositionStopsMotion);
            Self::stop_motion(&mut command);
        }
        if context.primary_hazard == SubterraneanHazard::SensorFault && Self::productive(&command) {
            violations.push(RuntimeInvariant::SensorFaultRemovesProductiveWork);
            Self::remove_productive_work(&mut command);
        }

        let command_modified = command != original;
        if violations.is_empty() {
            self.consecutive_breach_frames = 0;
        } else {
            self.total_breaches = self.total_breaches.saturating_add(1);
            self.consecutive_breach_frames = self.consecutive_breach_frames.saturating_add(1);
        }
        self.last_violations = violations.clone();
        let assessment = InvariantAssessment {
            violations,
            command_modified,
            total_breaches: self.total_breaches,
            consecutive_breach_frames: self.consecutive_breach_frames,
        };
        let _ = context.state;
        (command, assessment)
    }

    fn remove_isolated_authority(
        mut command: SubterraneanCommand,
        isolation: ActuatorIsolationReport,
    ) -> SubterraneanCommand {
        for actuator in PhysicalActuator::ALL {
            if !isolation.is_isolated(actuator) {
                continue;
            }
            match actuator {
                PhysicalActuator::Cutter => command.set_cutter_head(0.0),
                PhysicalActuator::Auger => command.set_auger_feed(0.0),
                PhysicalActuator::LeftTrack => command.set_left_track(0.0),
                PhysicalActuator::RightTrack => command.set_right_track(0.0),
                PhysicalActuator::Ballast => command.set_ballast_trim(0.0),
                PhysicalActuator::ThermalPump => command.set_thermal_pump(0.0),
                PhysicalActuator::DewateringPump => command.recovery.dewatering_pump = 0.0,
                PhysicalActuator::SealantInjector => command.recovery.sealant_injector = 0.0,
                PhysicalActuator::RelayDeployer => command.recovery.relay_deployer = 0.0,
                PhysicalActuator::RoofSupport => command.recovery.roof_support = 0.0,
            }
        }
        command
    }

    pub fn total_breaches(&self) -> u64 {
        self.total_breaches
    }

    pub fn last_violations(&self) -> &[RuntimeInvariant] {
        &self.last_violations
    }

    pub fn reset_runtime(&mut self) {
        self.consecutive_breach_frames = 0;
        self.last_violations.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context<'a>(state: &'a SubterraneanState) -> InvariantContext<'a> {
        InvariantContext {
            state,
            safety_level: MotorSafetyLevel::Green,
            primary_hazard: SubterraneanHazard::None,
            tunnel_conflict: false,
            return_feasible: true,
            capability_disposition: CapabilityDisposition::FullMission,
            actuator_isolation: ActuatorIsolationReport::nominal(),
        }
    }

    #[test]
    fn red_tier_removes_productive_work_without_removing_cooling() {
        let state = SubterraneanState::home();
        let mut command = SubterraneanCommand::zero();
        command.set_cutter_head(0.8);
        command.set_auger_feed(0.7);
        command.set_thermal_pump(1.0);
        let mut input = context(&state);
        input.safety_level = MotorSafetyLevel::Red;
        let (command, assessment) = RuntimeInvariantMonitor::default().enforce(command, input);
        assert_eq!(command.cutter_head(), 0.0);
        assert_eq!(command.auger_feed(), 0.0);
        assert_eq!(command.thermal_pump(), 1.0);
        assert!(
            assessment
                .violations
                .contains(&RuntimeInvariant::RedTierRemovesProductiveWork)
        );
    }

    #[test]
    fn tunnel_conflict_stops_all_motion() {
        let state = SubterraneanState::home();
        let mut command = SubterraneanCommand::zero();
        command.set_left_track(0.5);
        command.set_right_track(0.5);
        let mut input = context(&state);
        input.tunnel_conflict = true;
        let (command, assessment) = RuntimeInvariantMonitor::default().enforce(command, input);
        assert_eq!(command.left_track(), 0.0);
        assert_eq!(command.right_track(), 0.0);
        assert!(!assessment.passed());
    }
}
