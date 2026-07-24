// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic mission contingency supervision.
//!
//! The supervisor resolves safety signals with one documented precedence order
//! and records every phase transition. It does not fly the aircraft; it emits
//! directives consumed by guidance, landing-zone selection, and the emergency
//! landing controller.

use serde::{Deserialize, Serialize};

use crate::powertrain::FuelReserveAction;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MissionPhase {
    Dispatch,
    Transit,
    Search,
    Hover,
    Extraction,
    ReturnToBase,
    Divert,
    EmergencyLanding,
    Complete,
    Aborted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MissionDirective {
    Continue,
    HoldPosition,
    ReturnToBase,
    DivertToLandingZone,
    EmergencyLand,
    DisarmAfterLanding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContingencyReason {
    AuthorityExpired,
    NavigationLost,
    FuelReserve,
    TerrainUnsafe,
    WeatherLimit,
    CriticalFlightFault,
    OperatorAbort,
    Landed,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MissionSupervisorConfig {
    /// Time allowed to hold while navigation sources recover.
    pub navigation_loss_grace_s: f64,
    /// Mission wind limit before return/divert action.
    pub maximum_wind_speed_mps: f64,
    /// Bound on retained transition evidence.
    pub max_transitions: usize,
}

impl Default for MissionSupervisorConfig {
    fn default() -> Self {
        Self {
            navigation_loss_grace_s: 2.0,
            maximum_wind_speed_mps: 20.0,
            max_transitions: 128,
        }
    }
}

impl MissionSupervisorConfig {
    pub fn validate(&self) -> bool {
        self.navigation_loss_grace_s.is_finite()
            && self.navigation_loss_grace_s >= 0.0
            && self.maximum_wind_speed_mps.is_finite()
            && self.maximum_wind_speed_mps > 0.0
            && self.max_transitions > 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MissionSafetySnapshot {
    pub authority_valid: bool,
    pub navigation_usable: bool,
    pub fuel_action: FuelReserveAction,
    pub terrain_safe: bool,
    pub wind_speed_mps: f64,
    pub critical_flight_fault: bool,
    pub operator_abort: bool,
    pub landed: bool,
}

impl MissionSafetySnapshot {
    pub fn nominal() -> Self {
        Self {
            authority_valid: true,
            navigation_usable: true,
            fuel_action: FuelReserveAction::ContinueMission,
            terrain_safe: true,
            wind_speed_mps: 0.0,
            critical_flight_fault: false,
            operator_abort: false,
            landed: false,
        }
    }

    fn is_finite(&self) -> bool {
        self.wind_speed_mps.is_finite() && self.wind_speed_mps >= 0.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MissionTransition {
    pub sequence: u64,
    pub from: MissionPhase,
    pub to: MissionPhase,
    pub directive: MissionDirective,
    pub reason: ContingencyReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MissionDecision {
    pub phase: MissionPhase,
    pub directive: MissionDirective,
    pub reason: Option<ContingencyReason>,
    pub monotonic_time_s: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissionSupervisorError {
    InvalidConfiguration,
    NonFiniteSnapshot,
    TimeWentBackwards,
    TransitionCapacityExceeded,
}

#[derive(Debug, Clone)]
pub struct MissionSupervisor {
    config: MissionSupervisorConfig,
    phase: MissionPhase,
    last_time_s: Option<f64>,
    navigation_loss_since_s: Option<f64>,
    transitions: Vec<MissionTransition>,
    next_sequence: u64,
}

impl MissionSupervisor {
    pub fn new(initial_phase: MissionPhase) -> Self {
        Self {
            config: MissionSupervisorConfig::default(),
            phase: initial_phase,
            last_time_s: None,
            navigation_loss_since_s: None,
            transitions: Vec::new(),
            next_sequence: 1,
        }
    }

    pub fn with_config(
        initial_phase: MissionPhase,
        config: MissionSupervisorConfig,
    ) -> Result<Self, MissionSupervisorError> {
        if !config.validate() {
            return Err(MissionSupervisorError::InvalidConfiguration);
        }
        Ok(Self {
            config,
            ..Self::new(initial_phase)
        })
    }

    pub fn phase(&self) -> MissionPhase {
        self.phase
    }

    pub fn transitions(&self) -> &[MissionTransition] {
        &self.transitions
    }

    /// Advance a normal mission phase. Contingency phases cannot be escaped by
    /// a nominal caller; they require reset or a completed landing.
    pub fn set_nominal_phase(&mut self, phase: MissionPhase) -> bool {
        if matches!(
            self.phase,
            MissionPhase::EmergencyLanding | MissionPhase::Complete | MissionPhase::Aborted
        ) || matches!(
            phase,
            MissionPhase::EmergencyLanding | MissionPhase::Complete | MissionPhase::Aborted
        ) {
            return false;
        }
        self.phase = phase;
        true
    }

    pub fn evaluate(
        &mut self,
        monotonic_time_s: f64,
        snapshot: MissionSafetySnapshot,
    ) -> Result<MissionDecision, MissionSupervisorError> {
        if !self.config.validate() {
            return Err(MissionSupervisorError::InvalidConfiguration);
        }
        if !monotonic_time_s.is_finite() || !snapshot.is_finite() {
            return Err(MissionSupervisorError::NonFiniteSnapshot);
        }
        if self
            .last_time_s
            .is_some_and(|previous| monotonic_time_s < previous)
        {
            return Err(MissionSupervisorError::TimeWentBackwards);
        }
        self.last_time_s = Some(monotonic_time_s);

        if snapshot.navigation_usable {
            self.navigation_loss_since_s = None;
        } else if self.navigation_loss_since_s.is_none() {
            self.navigation_loss_since_s = Some(monotonic_time_s);
        }

        // Precedence: landed/abort/critical fault/mandatory fuel landing/
        // terrain/navigation/authority/weather/fuel return/continue.
        let decision = if snapshot.landed {
            (
                MissionPhase::Complete,
                MissionDirective::DisarmAfterLanding,
                Some(ContingencyReason::Landed),
            )
        } else if snapshot.operator_abort {
            (
                MissionPhase::Aborted,
                MissionDirective::EmergencyLand,
                Some(ContingencyReason::OperatorAbort),
            )
        } else if snapshot.critical_flight_fault {
            (
                MissionPhase::EmergencyLanding,
                MissionDirective::EmergencyLand,
                Some(ContingencyReason::CriticalFlightFault),
            )
        } else if snapshot.fuel_action == FuelReserveAction::LandAsSoonAsPracticable {
            (
                MissionPhase::EmergencyLanding,
                MissionDirective::EmergencyLand,
                Some(ContingencyReason::FuelReserve),
            )
        } else if !snapshot.terrain_safe {
            (
                MissionPhase::Divert,
                MissionDirective::DivertToLandingZone,
                Some(ContingencyReason::TerrainUnsafe),
            )
        } else if !snapshot.navigation_usable {
            let lost_for_s =
                monotonic_time_s - self.navigation_loss_since_s.unwrap_or(monotonic_time_s);
            if lost_for_s <= self.config.navigation_loss_grace_s {
                (
                    self.phase,
                    MissionDirective::HoldPosition,
                    Some(ContingencyReason::NavigationLost),
                )
            } else {
                (
                    MissionPhase::EmergencyLanding,
                    MissionDirective::EmergencyLand,
                    Some(ContingencyReason::NavigationLost),
                )
            }
        } else if !snapshot.authority_valid {
            (
                MissionPhase::ReturnToBase,
                MissionDirective::ReturnToBase,
                Some(ContingencyReason::AuthorityExpired),
            )
        } else if snapshot.wind_speed_mps > self.config.maximum_wind_speed_mps {
            (
                MissionPhase::ReturnToBase,
                MissionDirective::ReturnToBase,
                Some(ContingencyReason::WeatherLimit),
            )
        } else if snapshot.fuel_action == FuelReserveAction::ReturnToBase {
            (
                MissionPhase::ReturnToBase,
                MissionDirective::ReturnToBase,
                Some(ContingencyReason::FuelReserve),
            )
        } else {
            (self.phase, MissionDirective::Continue, None)
        };

        if decision.0 != self.phase {
            self.record_transition(
                decision.0,
                decision.1,
                decision.2.expect("phase-changing decision has a reason"),
            )?;
        }
        Ok(MissionDecision {
            phase: self.phase,
            directive: decision.1,
            reason: decision.2,
            monotonic_time_s,
        })
    }

    fn record_transition(
        &mut self,
        next: MissionPhase,
        directive: MissionDirective,
        reason: ContingencyReason,
    ) -> Result<(), MissionSupervisorError> {
        if self.transitions.len() >= self.config.max_transitions {
            return Err(MissionSupervisorError::TransitionCapacityExceeded);
        }
        let previous = self.phase;
        self.phase = next;
        self.transitions.push(MissionTransition {
            sequence: self.next_sequence,
            from: previous,
            to: next,
            directive,
            reason,
        });
        self.next_sequence = self.next_sequence.saturating_add(1);
        Ok(())
    }

    pub fn reset(&mut self, initial_phase: MissionPhase) {
        self.phase = initial_phase;
        self.last_time_s = None;
        self.navigation_loss_since_s = None;
        self.transitions.clear();
        self.next_sequence = 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn critical_fault_overrides_return_fuel_action() {
        let mut supervisor = MissionSupervisor::new(MissionPhase::Search);
        let mut snapshot = MissionSafetySnapshot::nominal();
        snapshot.fuel_action = FuelReserveAction::ReturnToBase;
        snapshot.critical_flight_fault = true;
        let decision = supervisor.evaluate(1.0, snapshot).unwrap();
        assert_eq!(decision.phase, MissionPhase::EmergencyLanding);
        assert_eq!(decision.directive, MissionDirective::EmergencyLand);
        assert_eq!(
            decision.reason,
            Some(ContingencyReason::CriticalFlightFault)
        );
    }

    #[test]
    fn navigation_loss_holds_then_lands() {
        let mut supervisor = MissionSupervisor::with_config(
            MissionPhase::Search,
            MissionSupervisorConfig {
                navigation_loss_grace_s: 2.0,
                ..MissionSupervisorConfig::default()
            },
        )
        .unwrap();
        let mut snapshot = MissionSafetySnapshot::nominal();
        snapshot.navigation_usable = false;
        assert_eq!(
            supervisor.evaluate(1.0, snapshot).unwrap().directive,
            MissionDirective::HoldPosition
        );
        assert_eq!(
            supervisor.evaluate(3.1, snapshot).unwrap().directive,
            MissionDirective::EmergencyLand
        );
    }

    #[test]
    fn expired_authority_returns_while_navigation_is_usable() {
        let mut supervisor = MissionSupervisor::new(MissionPhase::Transit);
        let mut snapshot = MissionSafetySnapshot::nominal();
        snapshot.authority_valid = false;
        let decision = supervisor.evaluate(1.0, snapshot).unwrap();
        assert_eq!(decision.phase, MissionPhase::ReturnToBase);
        assert_eq!(decision.reason, Some(ContingencyReason::AuthorityExpired));
    }

    #[test]
    fn landed_state_is_terminal_and_disarmed() {
        let mut supervisor = MissionSupervisor::new(MissionPhase::EmergencyLanding);
        let mut snapshot = MissionSafetySnapshot::nominal();
        snapshot.landed = true;
        let decision = supervisor.evaluate(1.0, snapshot).unwrap();
        assert_eq!(decision.phase, MissionPhase::Complete);
        assert_eq!(decision.directive, MissionDirective::DisarmAfterLanding);
        assert!(!supervisor.set_nominal_phase(MissionPhase::Search));
    }
}
