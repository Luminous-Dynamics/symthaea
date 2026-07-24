// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Explicit rescue request, feasibility, and handoff state machine.
//!
//! A distress heartbeat is not permission to abandon one's own return reserve.
//! Rescue therefore requires a bounded request, a feasible offer, and explicit
//! acceptance before the assisting platform changes mission. Transport-level
//! authentication remains external to this crate.

use crate::path_memory::ReturnPathAssessment;
use crate::relay_mesh::MeshAssessment;
use crate::team::AgentId;
use serde::{Deserialize, Serialize};

pub const RESCUE_BATTERY_RESERVE: f64 = 0.08;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RescueCaseId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RescueCapability {
    Communications,
    Navigation,
    Dewatering,
    RoofSupport,
    Extraction,
    GeneralAssistance,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RescueRequest {
    pub case_id: RescueCaseId,
    pub requester: AgentId,
    pub epoch: u32,
    pub sequence: u64,
    pub issued_step: u64,
    pub expires_step: u64,
    pub depth_m: f64,
    pub battery_ratio: f64,
    pub route_confidence: f64,
    pub hazard_severity: f32,
    pub capability: RescueCapability,
}

impl RescueRequest {
    pub fn is_valid(self) -> bool {
        self.requester != AgentId::SURFACE_CONTROL
            && self.expires_step >= self.issued_step
            && self.depth_m.is_finite()
            && (0.0..=200.0).contains(&self.depth_m)
            && self.battery_ratio.is_finite()
            && (0.0..=1.0).contains(&self.battery_ratio)
            && self.route_confidence.is_finite()
            && (0.0..=1.0).contains(&self.route_confidence)
            && self.hazard_severity.is_finite()
            && (0.0..=1.0).contains(&self.hazard_severity)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RescueFeasibility {
    pub feasible: bool,
    pub estimated_travel_m: f64,
    pub estimated_battery_cost: f64,
    pub projected_battery_margin: f64,
    pub own_return_feasible: bool,
    pub peer_reachable: bool,
    pub request_current: bool,
}

pub fn evaluate_rescue(
    request: RescueRequest,
    current_step: u64,
    local_depth_m: f64,
    local_battery_ratio: f64,
    own_return: ReturnPathAssessment,
    peer_mesh: MeshAssessment,
) -> RescueFeasibility {
    let request_current = request.is_valid() && current_step <= request.expires_step;
    let estimated_travel_m = if local_depth_m.is_finite() {
        (local_depth_m - request.depth_m).abs()
    } else {
        200.0
    };
    let difficulty = 1.0
        + request.hazard_severity as f64 * 0.9
        + (1.0 - request.route_confidence) * 0.6
        + (1.0 - peer_mesh.bottleneck_quality) * 0.3;
    let estimated_battery_cost = (estimated_travel_m * 0.0025 * difficulty + 0.02).clamp(0.0, 1.0);
    let projected_battery_margin =
        local_battery_ratio - estimated_battery_cost - own_return.estimated_battery_required;
    let own_return_feasible = own_return.feasible;
    let peer_reachable = peer_mesh.reachable && peer_mesh.bottleneck_quality >= 0.2;
    RescueFeasibility {
        feasible: request_current
            && own_return_feasible
            && peer_reachable
            && projected_battery_margin >= RESCUE_BATTERY_RESERVE,
        estimated_travel_m,
        estimated_battery_cost,
        projected_battery_margin,
        own_return_feasible,
        peer_reachable,
        request_current,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RescueOffer {
    pub case_id: RescueCaseId,
    pub rescuer: AgentId,
    pub sequence: u64,
    pub offered_step: u64,
    pub feasibility: RescueFeasibility,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RescueHandoffState {
    Idle,
    Requested,
    Offered,
    Accepted,
    Active,
    Completed,
    Aborted,
}

impl RescueHandoffState {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Requested => "requested",
            Self::Offered => "offered",
            Self::Accepted => "accepted",
            Self::Active => "active",
            Self::Completed => "completed",
            Self::Aborted => "aborted",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RescueTransitionError {
    InvalidRequest,
    WrongCase,
    WrongActor,
    InfeasibleOffer,
    InvalidTransition,
    Replay,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RescueHandoff {
    local_agent: AgentId,
    state: RescueHandoffState,
    request: Option<RescueRequest>,
    offer: Option<RescueOffer>,
    last_sequence: u64,
}

impl RescueHandoff {
    pub fn new(local_agent: AgentId) -> Self {
        Self {
            local_agent,
            state: RescueHandoffState::Idle,
            request: None,
            offer: None,
            last_sequence: 0,
        }
    }

    pub fn receive_request(&mut self, request: RescueRequest) -> Result<(), RescueTransitionError> {
        if !request.is_valid() {
            return Err(RescueTransitionError::InvalidRequest);
        }
        if request.sequence <= self.last_sequence {
            return Err(RescueTransitionError::Replay);
        }
        if self
            .request
            .is_some_and(|current| current.case_id != request.case_id)
            && matches!(
                self.state,
                RescueHandoffState::Accepted | RescueHandoffState::Active
            )
        {
            return Err(RescueTransitionError::InvalidTransition);
        }
        self.last_sequence = request.sequence;
        self.request = Some(request);
        self.offer = None;
        self.state = RescueHandoffState::Requested;
        Ok(())
    }

    pub fn offer(
        &mut self,
        sequence: u64,
        offered_step: u64,
        feasibility: RescueFeasibility,
    ) -> Result<RescueOffer, RescueTransitionError> {
        if self.state != RescueHandoffState::Requested {
            return Err(RescueTransitionError::InvalidTransition);
        }
        if sequence <= self.last_sequence {
            return Err(RescueTransitionError::Replay);
        }
        if !feasibility.feasible {
            return Err(RescueTransitionError::InfeasibleOffer);
        }
        let Some(request) = self.request else {
            return Err(RescueTransitionError::InvalidTransition);
        };
        let offer = RescueOffer {
            case_id: request.case_id,
            rescuer: self.local_agent,
            sequence,
            offered_step,
            feasibility,
        };
        self.last_sequence = sequence;
        self.offer = Some(offer);
        self.state = RescueHandoffState::Offered;
        Ok(offer)
    }

    pub fn accept(
        &mut self,
        requester: AgentId,
        case_id: RescueCaseId,
        sequence: u64,
    ) -> Result<(), RescueTransitionError> {
        if self.state != RescueHandoffState::Offered {
            return Err(RescueTransitionError::InvalidTransition);
        }
        let Some(request) = self.request else {
            return Err(RescueTransitionError::InvalidTransition);
        };
        if request.case_id != case_id {
            return Err(RescueTransitionError::WrongCase);
        }
        if request.requester != requester {
            return Err(RescueTransitionError::WrongActor);
        }
        if sequence <= self.last_sequence {
            return Err(RescueTransitionError::Replay);
        }
        self.last_sequence = sequence;
        self.state = RescueHandoffState::Accepted;
        Ok(())
    }

    pub fn begin(&mut self) -> Result<(), RescueTransitionError> {
        if self.state != RescueHandoffState::Accepted {
            return Err(RescueTransitionError::InvalidTransition);
        }
        self.state = RescueHandoffState::Active;
        Ok(())
    }

    pub fn complete(&mut self) -> Result<(), RescueTransitionError> {
        if self.state != RescueHandoffState::Active {
            return Err(RescueTransitionError::InvalidTransition);
        }
        self.state = RescueHandoffState::Completed;
        Ok(())
    }

    pub fn abort(&mut self) -> Result<(), RescueTransitionError> {
        if matches!(
            self.state,
            RescueHandoffState::Idle | RescueHandoffState::Completed | RescueHandoffState::Aborted
        ) {
            return Err(RescueTransitionError::InvalidTransition);
        }
        self.state = RescueHandoffState::Aborted;
        Ok(())
    }

    pub fn state(&self) -> RescueHandoffState {
        self.state
    }

    pub fn request(&self) -> Option<RescueRequest> {
        self.request
    }

    pub fn active_target(&self) -> Option<RescueRequest> {
        if matches!(
            self.state,
            RescueHandoffState::Accepted | RescueHandoffState::Active
        ) {
            self.request
        } else {
            None
        }
    }

    pub fn reset(&mut self) {
        self.state = RescueHandoffState::Idle;
        self.request = None;
        self.offer = None;
        self.last_sequence = 0;
    }
}

impl Default for RescueHandoff {
    fn default() -> Self {
        Self::new(AgentId::new(1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> RescueRequest {
        RescueRequest {
            case_id: RescueCaseId(7),
            requester: AgentId::new(2),
            epoch: 1,
            sequence: 1,
            issued_step: 10,
            expires_step: 100,
            depth_m: 30.0,
            battery_ratio: 0.2,
            route_confidence: 0.7,
            hazard_severity: 0.8,
            capability: RescueCapability::Extraction,
        }
    }

    fn feasible() -> RescueFeasibility {
        RescueFeasibility {
            feasible: true,
            estimated_travel_m: 10.0,
            estimated_battery_cost: 0.05,
            projected_battery_margin: 0.3,
            own_return_feasible: true,
            peer_reachable: true,
            request_current: true,
        }
    }

    #[test]
    fn rescue_requires_offer_and_requester_acceptance() {
        let mut handoff = RescueHandoff::new(AgentId::new(1));
        assert_eq!(handoff.receive_request(request()), Ok(()));
        assert!(handoff.offer(2, 20, feasible()).is_ok());
        assert_eq!(
            handoff.accept(AgentId::new(3), RescueCaseId(7), 3),
            Err(RescueTransitionError::WrongActor)
        );
        assert_eq!(handoff.accept(AgentId::new(2), RescueCaseId(7), 3), Ok(()));
        assert_eq!(handoff.begin(), Ok(()));
        assert_eq!(handoff.state(), RescueHandoffState::Active);
    }

    #[test]
    fn infeasible_offer_cannot_change_mission_authority() {
        let mut handoff = RescueHandoff::new(AgentId::new(1));
        assert_eq!(handoff.receive_request(request()), Ok(()));
        let mut denied = feasible();
        denied.feasible = false;
        assert_eq!(
            handoff.offer(2, 20, denied),
            Err(RescueTransitionError::InfeasibleOffer)
        );
        assert_eq!(handoff.state(), RescueHandoffState::Requested);
    }

    #[test]
    fn feasibility_preserves_own_return_reserve() {
        let mut own_return = ReturnPathAssessment::surface();
        own_return.estimated_battery_required = 0.3;
        own_return.feasible = true;
        let mesh = MeshAssessment {
            reachable: true,
            bottleneck_quality: 0.8,
            hops: 2,
            fresh_links_considered: 2,
            stale_links_ignored: 0,
        };
        let result = evaluate_rescue(request(), 20, 5.0, 0.4, own_return, mesh);
        assert!(!result.feasible);
        assert!(result.projected_battery_margin < RESCUE_BATTERY_RESERVE);
    }
}
