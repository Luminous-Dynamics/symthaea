// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Mission-level intent and deterministic physical/team overrides.
//!
//! Cognitive thought remains expressive, but it is no longer the only intent
//! presented to the controller. Physical hazards remain authoritative; team
//! directives may select yielding, relay maintenance, or an explicitly
//! accepted rescue only while local safety permits.

use crate::occupancy::{ReservationPriority, TunnelDirection};
use crate::safety::{HazardAssessment, SubterraneanHazard};
use crate::team_operations::TeamDirective;
use crate::types::SubterraneanState;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubterraneanMissionIntent {
    Explore,
    ProbeAhead,
    FollowVein,
    ReturnHome,
    EmergencySurface,
    HoldPosition,
    YieldTunnel,
    MaintainRelay,
    AssistPeer,
}

impl SubterraneanMissionIntent {
    pub const COUNT: usize = 9;
    pub const ALL: [Self; Self::COUNT] = [
        Self::Explore,
        Self::ProbeAhead,
        Self::FollowVein,
        Self::ReturnHome,
        Self::EmergencySurface,
        Self::HoldPosition,
        Self::YieldTunnel,
        Self::MaintainRelay,
        Self::AssistPeer,
    ];

    pub const fn index(self) -> usize {
        match self {
            Self::Explore => 0,
            Self::ProbeAhead => 1,
            Self::FollowVein => 2,
            Self::ReturnHome => 3,
            Self::EmergencySurface => 4,
            Self::HoldPosition => 5,
            Self::YieldTunnel => 6,
            Self::MaintainRelay => 7,
            Self::AssistPeer => 8,
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Explore => "explore",
            Self::ProbeAhead => "probe_ahead",
            Self::FollowVein => "follow_vein",
            Self::ReturnHome => "return_home",
            Self::EmergencySurface => "emergency_surface",
            Self::HoldPosition => "hold_position",
            Self::YieldTunnel => "yield_tunnel",
            Self::MaintainRelay => "maintain_relay",
            Self::AssistPeer => "assist_peer",
        }
    }

    pub const fn tunnel_direction(self) -> TunnelDirection {
        match self {
            Self::Explore | Self::ProbeAhead | Self::FollowVein | Self::AssistPeer => {
                TunnelDirection::Outbound
            }
            Self::ReturnHome | Self::EmergencySurface | Self::YieldTunnel => {
                TunnelDirection::Inbound
            }
            Self::HoldPosition | Self::MaintainRelay => TunnelDirection::Holding,
        }
    }

    pub const fn reservation_priority(self) -> ReservationPriority {
        match self {
            Self::EmergencySurface => ReservationPriority::Emergency,
            Self::AssistPeer => ReservationPriority::Rescue,
            Self::ReturnHome | Self::YieldTunnel => ReservationPriority::Return,
            Self::Explore
            | Self::ProbeAhead
            | Self::FollowVein
            | Self::HoldPosition
            | Self::MaintainRelay => ReservationPriority::Routine,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MissionManager {
    requested: SubterraneanMissionIntent,
    effective: SubterraneanMissionIntent,
}

impl MissionManager {
    pub fn new(requested: SubterraneanMissionIntent) -> Self {
        Self {
            requested,
            effective: requested,
        }
    }

    pub fn set_requested(&mut self, requested: SubterraneanMissionIntent) {
        self.requested = requested;
    }

    pub fn requested(&self) -> SubterraneanMissionIntent {
        self.requested
    }

    pub fn effective(&self) -> SubterraneanMissionIntent {
        self.effective
    }

    pub fn update(
        &mut self,
        state: &SubterraneanState,
        hazard: HazardAssessment,
    ) -> SubterraneanMissionIntent {
        self.update_with_team(state, hazard, TeamDirective::None)
    }

    pub fn update_with_team(
        &mut self,
        state: &SubterraneanState,
        hazard: HazardAssessment,
        team_directive: TeamDirective,
    ) -> SubterraneanMissionIntent {
        self.effective = match hazard.primary {
            SubterraneanHazard::GeologicalUncertainty => SubterraneanMissionIntent::ProbeAhead,
            SubterraneanHazard::TunnelConflict => SubterraneanMissionIntent::YieldTunnel,
            SubterraneanHazard::LocalizationLoss
            | SubterraneanHazard::CommunicationsLoss
            | SubterraneanHazard::SensorFault => SubterraneanMissionIntent::HoldPosition,
            SubterraneanHazard::Thermal | SubterraneanHazard::SpoilJam if hazard.severity < 0.9 => {
                SubterraneanMissionIntent::HoldPosition
            }
            SubterraneanHazard::BatteryCritical if state.depth_m() > 20.0 => {
                SubterraneanMissionIntent::ReturnHome
            }
            SubterraneanHazard::ReturnReserve if hazard.severity < 0.9 => {
                SubterraneanMissionIntent::ReturnHome
            }
            SubterraneanHazard::Flood
            | SubterraneanHazard::Gas
            | SubterraneanHazard::RoofInstability
            | SubterraneanHazard::EscapeLoss
            | SubterraneanHazard::BatteryCritical
            | SubterraneanHazard::ReturnReserve => SubterraneanMissionIntent::EmergencySurface,
            SubterraneanHazard::Thermal | SubterraneanHazard::SpoilJam => {
                SubterraneanMissionIntent::ReturnHome
            }
            SubterraneanHazard::None => match team_directive {
                TeamDirective::None => self.requested,
                TeamDirective::YieldTunnel => SubterraneanMissionIntent::YieldTunnel,
                TeamDirective::MaintainRelay => SubterraneanMissionIntent::MaintainRelay,
                TeamDirective::AssistPeer => SubterraneanMissionIntent::AssistPeer,
                TeamDirective::HoldForQuorum => SubterraneanMissionIntent::HoldPosition,
            },
        };
        self.effective
    }

    pub fn reset(&mut self) {
        self.effective = self.requested;
    }
}

impl Default for MissionManager {
    fn default() -> Self {
        Self::new(SubterraneanMissionIntent::Explore)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embodiment::MotorSafetyLevel;

    #[test]
    fn navigation_loss_overrides_exploration_with_hold() {
        let mut manager = MissionManager::new(SubterraneanMissionIntent::Explore);
        let intent = manager.update(
            &SubterraneanState::home(),
            HazardAssessment {
                primary: SubterraneanHazard::LocalizationLoss,
                safety_level: MotorSafetyLevel::Orange,
                severity: 0.8,
            },
        );
        assert_eq!(intent, SubterraneanMissionIntent::HoldPosition);
    }

    #[test]
    fn severe_flood_overrides_requested_vein_following() {
        let mut manager = MissionManager::new(SubterraneanMissionIntent::FollowVein);
        let intent = manager.update(
            &SubterraneanState::home(),
            HazardAssessment {
                primary: SubterraneanHazard::Flood,
                safety_level: MotorSafetyLevel::Red,
                severity: 1.0,
            },
        );
        assert_eq!(intent, SubterraneanMissionIntent::EmergencySurface);
    }

    #[test]
    fn accepted_team_directive_changes_nominal_mission_only_without_local_hazard() {
        let mut manager = MissionManager::new(SubterraneanMissionIntent::Explore);
        let intent = manager.update_with_team(
            &SubterraneanState::home(),
            HazardAssessment::clear(),
            TeamDirective::AssistPeer,
        );
        assert_eq!(intent, SubterraneanMissionIntent::AssistPeer);
        let hazard_intent = manager.update_with_team(
            &SubterraneanState::home(),
            HazardAssessment {
                primary: SubterraneanHazard::Gas,
                safety_level: MotorSafetyLevel::Red,
                severity: 1.0,
            },
            TeamDirective::AssistPeer,
        );
        assert_eq!(hazard_intent, SubterraneanMissionIntent::EmergencySurface);
    }
}
