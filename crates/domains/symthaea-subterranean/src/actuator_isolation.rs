// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Command/response actuator monitoring and fail-silent isolation.
//!
//! Mechanical maintenance estimates long-term wear. This supervisor addresses a
//! different failure class: a commanded actuator that no longer produces a
//! compatible short-horizon plant response. Persistent mismatches remove only
//! that actuator's authority and remain latched until an explicit reset/service.

use crate::types::{
    CUTTER_TEMP_C, FORWARD_VELOCITY_MPS, PITCH_RAD, RELAY_LINK_QUALITY, ROOF_STABILITY,
    SEAL_INTEGRITY, SPOIL_BUFFER_FILL, SubterraneanCommand, SubterraneanState, WATER_INGRESS_RATIO,
};
use serde::{Deserialize, Serialize};

pub const NUM_MONITORED_ACTUATORS: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhysicalActuator {
    Cutter,
    Auger,
    LeftTrack,
    RightTrack,
    Ballast,
    ThermalPump,
    DewateringPump,
    SealantInjector,
    RelayDeployer,
    RoofSupport,
}

impl PhysicalActuator {
    pub const ALL: [Self; NUM_MONITORED_ACTUATORS] = [
        Self::Cutter,
        Self::Auger,
        Self::LeftTrack,
        Self::RightTrack,
        Self::Ballast,
        Self::ThermalPump,
        Self::DewateringPump,
        Self::SealantInjector,
        Self::RelayDeployer,
        Self::RoofSupport,
    ];

    pub const fn index(self) -> usize {
        match self {
            Self::Cutter => 0,
            Self::Auger => 1,
            Self::LeftTrack => 2,
            Self::RightTrack => 3,
            Self::Ballast => 4,
            Self::ThermalPump => 5,
            Self::DewateringPump => 6,
            Self::SealantInjector => 7,
            Self::RelayDeployer => 8,
            Self::RoofSupport => 9,
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Cutter => "cutter",
            Self::Auger => "auger",
            Self::LeftTrack => "left_track",
            Self::RightTrack => "right_track",
            Self::Ballast => "ballast",
            Self::ThermalPump => "thermal_pump",
            Self::DewateringPump => "dewatering_pump",
            Self::SealantInjector => "sealant_injector",
            Self::RelayDeployer => "relay_deployer",
            Self::RoofSupport => "roof_support",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ActuatorIsolationPolicy {
    pub energized_threshold: f32,
    pub mismatch_penalty: f64,
    pub recovery_rate: f64,
    pub isolation_threshold: f64,
    pub mismatch_streak_limit: u16,
}

impl Default for ActuatorIsolationPolicy {
    fn default() -> Self {
        Self {
            energized_threshold: 0.35,
            mismatch_penalty: 0.08,
            recovery_rate: 0.005,
            isolation_threshold: 0.2,
            mismatch_streak_limit: 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ActuatorIsolationReport {
    pub health: [f64; NUM_MONITORED_ACTUATORS],
    pub isolated: [bool; NUM_MONITORED_ACTUATORS],
    pub mismatch_streaks: [u16; NUM_MONITORED_ACTUATORS],
    pub isolated_count: usize,
    pub mobility_degraded: bool,
    pub cooling_degraded: bool,
    pub recovery_degraded: bool,
}

impl ActuatorIsolationReport {
    pub const fn nominal() -> Self {
        Self {
            health: [1.0; NUM_MONITORED_ACTUATORS],
            isolated: [false; NUM_MONITORED_ACTUATORS],
            mismatch_streaks: [0; NUM_MONITORED_ACTUATORS],
            isolated_count: 0,
            mobility_degraded: false,
            cooling_degraded: false,
            recovery_degraded: false,
        }
    }

    pub fn is_isolated(self, actuator: PhysicalActuator) -> bool {
        self.isolated[actuator.index()]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActuatorIsolationSupervisor {
    policy: ActuatorIsolationPolicy,
    health: [f64; NUM_MONITORED_ACTUATORS],
    mismatch_streaks: [u16; NUM_MONITORED_ACTUATORS],
    isolated: [bool; NUM_MONITORED_ACTUATORS],
    total_isolations: u64,
}

impl ActuatorIsolationSupervisor {
    pub fn new(policy: ActuatorIsolationPolicy) -> Self {
        Self {
            policy,
            health: [1.0; NUM_MONITORED_ACTUATORS],
            mismatch_streaks: [0; NUM_MONITORED_ACTUATORS],
            isolated: [false; NUM_MONITORED_ACTUATORS],
            total_isolations: 0,
        }
    }

    pub fn validate(&self) -> bool {
        self.policy.energized_threshold.is_finite()
            && (0.0..=1.0).contains(&self.policy.energized_threshold)
            && self.policy.mismatch_penalty.is_finite()
            && (0.0..=1.0).contains(&self.policy.mismatch_penalty)
            && self.policy.recovery_rate.is_finite()
            && (0.0..=1.0).contains(&self.policy.recovery_rate)
            && self.policy.isolation_threshold.is_finite()
            && (0.0..=1.0).contains(&self.policy.isolation_threshold)
            && self.policy.mismatch_streak_limit > 0
            && self
                .health
                .iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
    }

    fn demanded(command: &SubterraneanCommand, actuator: PhysicalActuator) -> f32 {
        match actuator {
            PhysicalActuator::Cutter => command.cutter_head().abs(),
            PhysicalActuator::Auger => command.auger_feed().abs(),
            PhysicalActuator::LeftTrack => command.left_track().abs(),
            PhysicalActuator::RightTrack => command.right_track().abs(),
            PhysicalActuator::Ballast => command.ballast_trim().abs(),
            PhysicalActuator::ThermalPump => command.thermal_pump().max(0.0),
            PhysicalActuator::DewateringPump => command.recovery.dewatering_pump,
            PhysicalActuator::SealantInjector => command.recovery.sealant_injector,
            PhysicalActuator::RelayDeployer => command.recovery.relay_deployer,
            PhysicalActuator::RoofSupport => command.recovery.roof_support,
        }
    }

    fn response_is_compatible(
        actuator: PhysicalActuator,
        command: &SubterraneanCommand,
        before: &SubterraneanState,
        after: &SubterraneanState,
    ) -> bool {
        let delta = |channel: usize| after.channels[channel] - before.channels[channel];
        (match actuator {
            PhysicalActuator::Cutter => {
                delta(CUTTER_TEMP_C).abs() > 1e-6 || delta(SPOIL_BUFFER_FILL).abs() > 1e-7
            }
            PhysicalActuator::Auger => delta(SPOIL_BUFFER_FILL).abs() > 1e-7,
            PhysicalActuator::LeftTrack | PhysicalActuator::RightTrack => {
                after.channels[FORWARD_VELOCITY_MPS].abs() > 1e-5
                    || delta(FORWARD_VELOCITY_MPS).abs() > 1e-6
            }
            PhysicalActuator::Ballast => delta(PITCH_RAD).abs() > 1e-7,
            PhysicalActuator::ThermalPump => {
                before.channels[CUTTER_TEMP_C] <= 25.0
                    || after.channels[CUTTER_TEMP_C] <= before.channels[CUTTER_TEMP_C] + 0.02
            }
            PhysicalActuator::DewateringPump => {
                before.channels[WATER_INGRESS_RATIO] <= 0.01
                    || after.channels[WATER_INGRESS_RATIO]
                        <= before.channels[WATER_INGRESS_RATIO] + 0.0005
            }
            PhysicalActuator::SealantInjector => {
                after.channels[SEAL_INTEGRITY] >= before.channels[SEAL_INTEGRITY]
            }
            PhysicalActuator::RelayDeployer => {
                after.channels[RELAY_LINK_QUALITY] >= before.channels[RELAY_LINK_QUALITY]
            }
            PhysicalActuator::RoofSupport => {
                after.channels[ROOF_STABILITY] >= before.channels[ROOF_STABILITY]
            }
        }) || Self::demanded(command, actuator) < 0.01
    }

    pub fn observe(
        &mut self,
        command: &SubterraneanCommand,
        before: &SubterraneanState,
        after: &SubterraneanState,
    ) -> ActuatorIsolationReport {
        for actuator in PhysicalActuator::ALL {
            let index = actuator.index();
            if self.isolated[index] {
                continue;
            }
            if Self::demanded(command, actuator) < self.policy.energized_threshold {
                self.mismatch_streaks[index] = 0;
                continue;
            }
            if Self::response_is_compatible(actuator, command, before, after) {
                self.mismatch_streaks[index] = 0;
                self.health[index] =
                    (self.health[index] + self.policy.recovery_rate).clamp(0.0, 1.0);
            } else {
                self.mismatch_streaks[index] = self.mismatch_streaks[index].saturating_add(1);
                self.health[index] =
                    (self.health[index] - self.policy.mismatch_penalty).clamp(0.0, 1.0);
                if self.mismatch_streaks[index] >= self.policy.mismatch_streak_limit
                    && self.health[index] <= self.policy.isolation_threshold
                {
                    self.isolated[index] = true;
                    self.total_isolations = self.total_isolations.saturating_add(1);
                }
            }
        }
        self.report()
    }

    pub fn constrain(&self, mut command: SubterraneanCommand) -> SubterraneanCommand {
        for actuator in PhysicalActuator::ALL {
            if !self.isolated[actuator.index()] {
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
        command.sanitize();
        command
    }

    pub fn report(&self) -> ActuatorIsolationReport {
        let isolated_count = self.isolated.iter().filter(|value| **value).count();
        ActuatorIsolationReport {
            health: self.health,
            isolated: self.isolated,
            mismatch_streaks: self.mismatch_streaks,
            isolated_count,
            mobility_degraded: self.isolated[PhysicalActuator::LeftTrack.index()]
                || self.isolated[PhysicalActuator::RightTrack.index()],
            cooling_degraded: self.isolated[PhysicalActuator::ThermalPump.index()],
            recovery_degraded: self.isolated[PhysicalActuator::DewateringPump.index()]
                || self.isolated[PhysicalActuator::SealantInjector.index()]
                || self.isolated[PhysicalActuator::RoofSupport.index()],
        }
    }

    pub const fn total_isolations(&self) -> u64 {
        self.total_isolations
    }

    pub fn service(&mut self, actuator: PhysicalActuator) {
        let index = actuator.index();
        self.health[index] = 1.0;
        self.mismatch_streaks[index] = 0;
        self.isolated[index] = false;
    }

    pub fn force_health_for_test(&mut self, actuator: PhysicalActuator, health: f64) {
        let index = actuator.index();
        self.health[index] = if health.is_finite() {
            health.clamp(0.0, 1.0)
        } else {
            0.0
        };
    }
}

impl Default for ActuatorIsolationSupervisor {
    fn default() -> Self {
        Self::new(ActuatorIsolationPolicy::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persistent_track_nonresponse_isolates_only_failed_track() {
        let mut supervisor = ActuatorIsolationSupervisor::new(ActuatorIsolationPolicy {
            mismatch_penalty: 0.25,
            mismatch_streak_limit: 4,
            isolation_threshold: 0.2,
            ..Default::default()
        });
        let mut command = SubterraneanCommand::zero();
        command.set_left_track(1.0);
        let state = SubterraneanState::home();
        for _ in 0..4 {
            supervisor.observe(&command, &state, &state);
        }
        let report = supervisor.report();
        assert!(report.is_isolated(PhysicalActuator::LeftTrack));
        assert!(!report.is_isolated(PhysicalActuator::RightTrack));
        let constrained = supervisor.constrain(command);
        assert_eq!(constrained.left_track(), 0.0);
    }

    #[test]
    fn compatible_cooling_response_preserves_authority() {
        let mut supervisor = ActuatorIsolationSupervisor::default();
        let mut before = SubterraneanState::home();
        before.channels[CUTTER_TEMP_C] = 100.0;
        let mut after = before.clone();
        after.channels[CUTTER_TEMP_C] = 99.0;
        let mut command = SubterraneanCommand::zero();
        command.set_thermal_pump(1.0);
        supervisor.observe(&command, &before, &after);
        assert!(
            !supervisor
                .report()
                .is_isolated(PhysicalActuator::ThermalPump)
        );
    }

    #[test]
    fn service_explicitly_restores_latched_authority() {
        let mut supervisor = ActuatorIsolationSupervisor::default();
        supervisor.force_health_for_test(PhysicalActuator::Cutter, 0.0);
        supervisor.isolated[PhysicalActuator::Cutter.index()] = true;
        supervisor.service(PhysicalActuator::Cutter);
        assert!(!supervisor.report().is_isolated(PhysicalActuator::Cutter));
    }
}
