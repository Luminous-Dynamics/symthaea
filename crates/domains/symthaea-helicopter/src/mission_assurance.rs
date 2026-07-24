// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent mission-decision assurance and precedence evidence.
//!
//! The mission supervisor chooses a directive. This module independently
//! derives the highest-priority active contingency from a safety snapshot and
//! verifies that the emitted directive/reason cannot contradict that hazard.

use serde::{Deserialize, Serialize};

use crate::mission_supervisor::{
    ContingencyReason, MissionDecision, MissionDirective, MissionSafetySnapshot,
};
use crate::powertrain::FuelReserveAction;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssuranceStatus {
    Verified,
    Rejected,
    Incomplete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MissionInvariantViolation {
    NonFiniteEvidence,
    MissingNavigationLossDuration,
    DirectiveMismatch,
    ReasonMismatch,
    ContinueUnderActiveHazard,
    EmergencyHazardNotEscalated,
    LandedAircraftNotDisarmed,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MissionAssuranceInput {
    pub snapshot: MissionSafetySnapshot,
    /// Required only when navigation is unusable.
    pub navigation_loss_duration_s: Option<f64>,
    pub navigation_loss_grace_s: f64,
    pub maximum_wind_speed_mps: f64,
}

impl MissionAssuranceInput {
    fn validate(&self) -> Result<(), MissionInvariantViolation> {
        if !self.snapshot.wind_speed_mps.is_finite()
            || !self.navigation_loss_grace_s.is_finite()
            || self.navigation_loss_grace_s < 0.0
            || !self.maximum_wind_speed_mps.is_finite()
            || self.maximum_wind_speed_mps <= 0.0
        {
            return Err(MissionInvariantViolation::NonFiniteEvidence);
        }
        if !self.snapshot.navigation_usable {
            let Some(duration) = self.navigation_loss_duration_s else {
                return Err(MissionInvariantViolation::MissingNavigationLossDuration);
            };
            if !duration.is_finite() || duration < 0.0 {
                return Err(MissionInvariantViolation::NonFiniteEvidence);
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpectedContingency {
    pub directive: MissionDirective,
    pub reason: Option<ContingencyReason>,
    /// Lower numeric values have stronger precedence.
    pub priority: u8,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MissionDecisionAudit {
    pub status: AssuranceStatus,
    pub expected: Option<ExpectedContingency>,
    pub observed_directive: MissionDirective,
    pub observed_reason: Option<ContingencyReason>,
    pub violations: Vec<MissionInvariantViolation>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct MissionAssuranceKernel;

impl MissionAssuranceKernel {
    pub fn expected(
        input: MissionAssuranceInput,
    ) -> Result<ExpectedContingency, MissionInvariantViolation> {
        input.validate()?;
        let snapshot = input.snapshot;
        let expected = if snapshot.landed {
            ExpectedContingency {
                directive: MissionDirective::DisarmAfterLanding,
                reason: Some(ContingencyReason::Landed),
                priority: 0,
            }
        } else if snapshot.operator_abort {
            ExpectedContingency {
                directive: MissionDirective::EmergencyLand,
                reason: Some(ContingencyReason::OperatorAbort),
                priority: 1,
            }
        } else if snapshot.critical_flight_fault {
            ExpectedContingency {
                directive: MissionDirective::EmergencyLand,
                reason: Some(ContingencyReason::CriticalFlightFault),
                priority: 2,
            }
        } else if snapshot.fuel_action == FuelReserveAction::LandAsSoonAsPracticable {
            ExpectedContingency {
                directive: MissionDirective::EmergencyLand,
                reason: Some(ContingencyReason::FuelReserve),
                priority: 3,
            }
        } else if !snapshot.terrain_safe {
            ExpectedContingency {
                directive: MissionDirective::DivertToLandingZone,
                reason: Some(ContingencyReason::TerrainUnsafe),
                priority: 4,
            }
        } else if !snapshot.navigation_usable {
            let duration = input.navigation_loss_duration_s.unwrap_or(0.0);
            if duration <= input.navigation_loss_grace_s {
                ExpectedContingency {
                    directive: MissionDirective::HoldPosition,
                    reason: Some(ContingencyReason::NavigationLost),
                    priority: 5,
                }
            } else {
                ExpectedContingency {
                    directive: MissionDirective::EmergencyLand,
                    reason: Some(ContingencyReason::NavigationLost),
                    priority: 5,
                }
            }
        } else if !snapshot.authority_valid {
            ExpectedContingency {
                directive: MissionDirective::ReturnToBase,
                reason: Some(ContingencyReason::AuthorityExpired),
                priority: 6,
            }
        } else if snapshot.wind_speed_mps > input.maximum_wind_speed_mps {
            ExpectedContingency {
                directive: MissionDirective::ReturnToBase,
                reason: Some(ContingencyReason::WeatherLimit),
                priority: 7,
            }
        } else if snapshot.fuel_action == FuelReserveAction::ReturnToBase {
            ExpectedContingency {
                directive: MissionDirective::ReturnToBase,
                reason: Some(ContingencyReason::FuelReserve),
                priority: 8,
            }
        } else {
            ExpectedContingency {
                directive: MissionDirective::Continue,
                reason: None,
                priority: u8::MAX,
            }
        };
        Ok(expected)
    }

    pub fn audit(input: MissionAssuranceInput, decision: MissionDecision) -> MissionDecisionAudit {
        let expected = match Self::expected(input) {
            Ok(expected) => expected,
            Err(violation) => {
                return MissionDecisionAudit {
                    status: AssuranceStatus::Incomplete,
                    expected: None,
                    observed_directive: decision.directive,
                    observed_reason: decision.reason,
                    violations: vec![violation],
                };
            }
        };
        let mut violations = Vec::new();
        if decision.directive != expected.directive {
            violations.push(MissionInvariantViolation::DirectiveMismatch);
        }
        if decision.reason != expected.reason {
            violations.push(MissionInvariantViolation::ReasonMismatch);
        }
        if expected.reason.is_some() && decision.directive == MissionDirective::Continue {
            violations.push(MissionInvariantViolation::ContinueUnderActiveHazard);
        }
        if matches!(
            expected.directive,
            MissionDirective::EmergencyLand | MissionDirective::DisarmAfterLanding
        ) && !matches!(
            decision.directive,
            MissionDirective::EmergencyLand | MissionDirective::DisarmAfterLanding
        ) {
            violations.push(MissionInvariantViolation::EmergencyHazardNotEscalated);
        }
        if input.snapshot.landed && decision.directive != MissionDirective::DisarmAfterLanding {
            violations.push(MissionInvariantViolation::LandedAircraftNotDisarmed);
        }
        MissionDecisionAudit {
            status: if violations.is_empty() {
                AssuranceStatus::Verified
            } else {
                AssuranceStatus::Rejected
            },
            expected: Some(expected),
            observed_directive: decision.directive,
            observed_reason: decision.reason,
            violations,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mission_supervisor::MissionPhase;

    fn input(snapshot: MissionSafetySnapshot) -> MissionAssuranceInput {
        MissionAssuranceInput {
            snapshot,
            navigation_loss_duration_s: Some(0.0),
            navigation_loss_grace_s: 2.0,
            maximum_wind_speed_mps: 20.0,
        }
    }

    fn decision(directive: MissionDirective, reason: Option<ContingencyReason>) -> MissionDecision {
        MissionDecision {
            phase: MissionPhase::Search,
            directive,
            reason,
            monotonic_time_s: 1.0,
        }
    }

    #[test]
    fn critical_fault_cannot_be_masked_by_fuel_return() {
        let mut snapshot = MissionSafetySnapshot::nominal();
        snapshot.critical_flight_fault = true;
        snapshot.fuel_action = FuelReserveAction::ReturnToBase;
        let expected = MissionAssuranceKernel::expected(input(snapshot)).unwrap();
        assert_eq!(expected.directive, MissionDirective::EmergencyLand);
        assert_eq!(
            expected.reason,
            Some(ContingencyReason::CriticalFlightFault)
        );
    }

    #[test]
    fn contradictory_continue_is_rejected() {
        let mut snapshot = MissionSafetySnapshot::nominal();
        snapshot.terrain_safe = false;
        let audit = MissionAssuranceKernel::audit(
            input(snapshot),
            decision(MissionDirective::Continue, None),
        );
        assert_eq!(audit.status, AssuranceStatus::Rejected);
        assert!(
            audit
                .violations
                .contains(&MissionInvariantViolation::ContinueUnderActiveHazard)
        );
    }

    #[test]
    fn navigation_grace_requires_duration_evidence() {
        let mut snapshot = MissionSafetySnapshot::nominal();
        snapshot.navigation_usable = false;
        let mut assurance = input(snapshot);
        assurance.navigation_loss_duration_s = None;
        let audit = MissionAssuranceKernel::audit(
            assurance,
            decision(
                MissionDirective::HoldPosition,
                Some(ContingencyReason::NavigationLost),
            ),
        );
        assert_eq!(audit.status, AssuranceStatus::Incomplete);
    }

    #[test]
    fn matching_emergency_decision_verifies() {
        let mut snapshot = MissionSafetySnapshot::nominal();
        snapshot.operator_abort = true;
        let audit = MissionAssuranceKernel::audit(
            input(snapshot),
            decision(
                MissionDirective::EmergencyLand,
                Some(ContingencyReason::OperatorAbort),
            ),
        );
        assert_eq!(audit.status, AssuranceStatus::Verified);
        assert!(audit.violations.is_empty());
    }
}
