// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Composition root for peer directory, shared map, occupancy, mesh, and rescue.

use crate::occupancy::{
    OccupancyAssessment, ReservationPriority, ReservationRejection, TunnelDirection,
    TunnelOccupancy, TunnelReservation,
};
use crate::peer_trust::{
    PeerAuthenticationAssertion, PeerTrustPolicy, PeerTrustRejection, PeerTrustSupervisor,
};
use crate::relay_mesh::{MeshAssessment, MeshLink, MeshLinkRejection, MeshNodeId, RelayMesh};
use crate::rescue::{
    RescueFeasibility, RescueHandoff, RescueHandoffState, RescueOffer, RescueRequest,
    RescueTransitionError, evaluate_rescue,
};
use crate::shared_map::{
    SharedMapRejection, SharedRouteKnowledge, SharedTunnelMap, SharedTunnelObservation,
};
use crate::team::{AgentId, HeartbeatRejection, TeamDirectory, TeamHeartbeat, TeamStatus};
use crate::team_leadership::{
    ByzantineContainmentAssessment, LeadershipLeaseVote, TeamLeadershipPolicy,
    TeamLeadershipSupervisor, VoteRejection,
};
use crate::{ReturnPathAssessment, SubterraneanState};
use serde::{Deserialize, Serialize};

pub const DISTRIBUTED_RECOVERY_CHECKPOINT_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TeamDirective {
    None,
    YieldTunnel,
    MaintainRelay,
    AssistPeer,
    HoldForQuorum,
}

impl TeamDirective {
    pub const fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::YieldTunnel => "yield_tunnel",
            Self::MaintainRelay => "maintain_relay",
            Self::AssistPeer => "assist_peer",
            Self::HoldForQuorum => "hold_for_quorum",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TeamOperationalAssessment {
    pub status: TeamStatus,
    pub occupancy: OccupancyAssessment,
    pub surface_mesh: MeshAssessment,
    pub shared_route: SharedRouteKnowledge,
    pub directive: TeamDirective,
    pub rescue_state: RescueHandoffState,
    pub distress_target: Option<AgentId>,
    pub byzantine_containment: ByzantineContainmentAssessment,
}

impl TeamOperationalAssessment {
    pub const fn solo() -> Self {
        Self {
            status: TeamStatus::alone(),
            occupancy: OccupancyAssessment::clear(),
            surface_mesh: MeshAssessment::unreachable(),
            shared_route: SharedRouteKnowledge::empty(),
            directive: TeamDirective::None,
            rescue_state: RescueHandoffState::Idle,
            distress_target: None,
            byzantine_containment: ByzantineContainmentAssessment::nominal(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedRecoveryCheckpoint {
    pub schema_version: u16,
    pub peer_trust: PeerTrustSupervisor,
    pub leadership: TeamLeadershipSupervisor,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TeamCoordinator {
    local_agent: AgentId,
    directory: TeamDirectory,
    shared_map: SharedTunnelMap,
    occupancy: TunnelOccupancy,
    relay_mesh: RelayMesh,
    rescue: RescueHandoff,
    peer_trust: PeerTrustSupervisor,
    leadership: TeamLeadershipSupervisor,
    last_assessment: TeamOperationalAssessment,
}

impl TeamCoordinator {
    pub fn new(local_agent: AgentId) -> Self {
        Self::new_with_deployment(local_agent, 0)
    }

    pub fn new_with_deployment(local_agent: AgentId, deployment_id: u64) -> Self {
        Self {
            local_agent,
            directory: TeamDirectory::new(
                local_agent,
                crate::team::DEFAULT_TEAM_CAPACITY,
                crate::team::DEFAULT_PEER_STALE_STEPS,
            ),
            shared_map: SharedTunnelMap::default(),
            occupancy: TunnelOccupancy::new(
                local_agent,
                crate::occupancy::DEFAULT_OCCUPANCY_CAPACITY,
            ),
            relay_mesh: RelayMesh::default(),
            rescue: RescueHandoff::new(local_agent),
            peer_trust: PeerTrustSupervisor::new(deployment_id),
            leadership: TeamLeadershipSupervisor::default(),
            last_assessment: TeamOperationalAssessment::solo(),
        }
    }

    pub fn deployment_id(&self) -> u64 {
        self.peer_trust.deployment_id()
    }

    pub fn ingest_peer_assertion(
        &mut self,
        assertion: PeerAuthenticationAssertion,
    ) -> Result<(), PeerTrustRejection> {
        self.peer_trust.ingest(assertion)
    }

    pub fn ingest_leadership_vote(
        &mut self,
        vote: LeadershipLeaseVote,
    ) -> Result<(), VoteRejection> {
        self.leadership.ingest(vote)
    }

    pub fn recovery_checkpoint(&self) -> DistributedRecoveryCheckpoint {
        DistributedRecoveryCheckpoint {
            schema_version: DISTRIBUTED_RECOVERY_CHECKPOINT_SCHEMA_VERSION,
            peer_trust: self.peer_trust.clone(),
            leadership: self.leadership.clone(),
        }
    }

    pub fn load_recovery_checkpoint(&mut self, checkpoint: &DistributedRecoveryCheckpoint) -> bool {
        if checkpoint.schema_version != DISTRIBUTED_RECOVERY_CHECKPOINT_SCHEMA_VERSION
            || !checkpoint.peer_trust.validate()
            || !checkpoint.leadership.validate()
        {
            return false;
        }
        self.peer_trust = checkpoint.peer_trust.clone();
        self.leadership = checkpoint.leadership.clone();
        true
    }

    pub fn ingest_heartbeat(
        &mut self,
        heartbeat: TeamHeartbeat,
        received_step: u64,
    ) -> Result<(), HeartbeatRejection> {
        self.directory.ingest(heartbeat, received_step)
    }

    pub fn merge_tunnel_observation(
        &mut self,
        observation: SharedTunnelObservation,
    ) -> Result<(), SharedMapRejection> {
        self.shared_map.merge(observation)
    }

    pub fn ingest_reservation(
        &mut self,
        reservation: TunnelReservation,
    ) -> Result<(), ReservationRejection> {
        self.occupancy.ingest(reservation)
    }

    pub fn merge_mesh_link(&mut self, link: MeshLink) -> Result<(), MeshLinkRejection> {
        self.relay_mesh.merge(link)
    }

    pub fn receive_rescue_request(
        &mut self,
        request: RescueRequest,
    ) -> Result<(), RescueTransitionError> {
        self.rescue.receive_request(request)
    }

    pub fn evaluate_pending_rescue(
        &self,
        current_step: u64,
        local_state: &SubterraneanState,
        own_return: ReturnPathAssessment,
    ) -> Option<RescueFeasibility> {
        let request = self.rescue.request()?;
        let peer_mesh = self.relay_mesh.assess(
            MeshNodeId::Agent(self.local_agent),
            MeshNodeId::Agent(request.requester),
            current_step,
        );
        Some(evaluate_rescue(
            request,
            current_step,
            local_state.depth_m(),
            local_state.battery_ratio(),
            own_return,
            peer_mesh,
        ))
    }

    pub fn offer_rescue(
        &mut self,
        sequence: u64,
        offered_step: u64,
        feasibility: RescueFeasibility,
    ) -> Result<RescueOffer, RescueTransitionError> {
        self.rescue.offer(sequence, offered_step, feasibility)
    }

    pub fn accept_rescue(
        &mut self,
        requester: AgentId,
        case_id: crate::rescue::RescueCaseId,
        sequence: u64,
    ) -> Result<(), RescueTransitionError> {
        self.rescue.accept(requester, case_id, sequence)
    }

    pub fn begin_rescue(&mut self) -> Result<(), RescueTransitionError> {
        self.rescue.begin()
    }

    pub fn complete_rescue(&mut self) -> Result<(), RescueTransitionError> {
        self.rescue.complete()
    }

    pub fn abort_rescue(&mut self) -> Result<(), RescueTransitionError> {
        self.rescue.abort()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn assess(
        &mut self,
        current_step: u64,
        local_depth_m: f64,
        direction: TunnelDirection,
        priority: ReservationPriority,
        lookahead_m: f64,
        clearance_m: f64,
        require_hardware_backed_peers: bool,
        leadership_quorum_fraction: f64,
    ) -> TeamOperationalAssessment {
        let byzantine_containment = self.leadership.assess(
            current_step,
            &self.peer_trust,
            PeerTrustPolicy {
                require_hardware_backed: require_hardware_backed_peers,
            },
            TeamLeadershipPolicy {
                quorum_fraction: leadership_quorum_fraction,
            },
        );
        let status = self.directory.status(current_step);
        let occupancy = self.occupancy.assess(
            current_step,
            local_depth_m,
            direction,
            priority,
            lookahead_m,
            clearance_m,
        );
        let surface_mesh = self
            .relay_mesh
            .assess_surface(self.local_agent, current_step);
        let shared_route = self.shared_map.route_knowledge(local_depth_m);
        let distress_target = self
            .directory
            .freshest_distress(current_step)
            .map(|record| record.heartbeat.agent_id);
        let directive = if byzantine_containment.authority
            == crate::team_leadership::ByzantineContainmentAuthority::HoldForQuorum
        {
            TeamDirective::HoldForQuorum
        } else if occupancy.conflict() && occupancy.must_yield {
            TeamDirective::YieldTunnel
        } else if matches!(
            self.rescue.state(),
            RescueHandoffState::Accepted | RescueHandoffState::Active
        ) {
            TeamDirective::AssistPeer
        } else if status.fresh_peers > 0 && !surface_mesh.reachable {
            TeamDirective::MaintainRelay
        } else {
            TeamDirective::None
        };
        self.last_assessment = TeamOperationalAssessment {
            status,
            occupancy,
            surface_mesh,
            shared_route,
            directive,
            rescue_state: self.rescue.state(),
            distress_target,
            byzantine_containment,
        };
        self.last_assessment
    }

    pub fn last_assessment(&self) -> TeamOperationalAssessment {
        self.last_assessment
    }

    pub fn local_agent(&self) -> AgentId {
        self.local_agent
    }

    pub fn local_map_revision(&self) -> u64 {
        self.shared_map.observation_count() as u64
    }

    pub fn highest_peer_revision(&self) -> u64 {
        self.directory
            .peers()
            .map(|record| record.heartbeat.sequence)
            .max()
            .unwrap_or(self.local_map_revision())
    }

    pub fn reset(&mut self) {
        self.directory.clear();
        self.shared_map.clear();
        self.occupancy.clear();
        self.relay_mesh.clear();
        self.rescue.reset();
        self.peer_trust.reset();
        self.leadership.reset();
        self.last_assessment = TeamOperationalAssessment::solo();
    }
}

impl Default for TeamCoordinator {
    fn default() -> Self {
        Self::new(AgentId::new(1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::occupancy::{ReservationPriority, TunnelDirection, TunnelReservation};
    use crate::team::{PeerCondition, TeamRole};
    use crate::team_leadership::ByzantineContainmentAuthority;

    #[test]
    fn lower_priority_local_agent_is_directed_to_yield() {
        let mut coordinator = TeamCoordinator::new(AgentId::new(3));
        let reservation = TunnelReservation {
            agent_id: AgentId::new(2),
            epoch: 1,
            sequence: 1,
            issued_step: 0,
            valid_from_step: 0,
            valid_until_step: 100,
            minimum_depth_m: 9.0,
            maximum_depth_m: 12.0,
            direction: TunnelDirection::Inbound,
            priority: ReservationPriority::Emergency,
        };
        assert_eq!(coordinator.ingest_reservation(reservation), Ok(()));
        let assessment = coordinator.assess(
            10,
            8.0,
            TunnelDirection::Outbound,
            ReservationPriority::Routine,
            5.0,
            1.0,
            false,
            0.5,
        );
        assert_eq!(assessment.directive, TeamDirective::YieldTunnel);
    }

    #[test]
    fn peer_distress_is_visible_but_not_automatically_accepted() {
        let mut coordinator = TeamCoordinator::new(AgentId::new(1));
        let heartbeat = TeamHeartbeat {
            agent_id: AgentId::new(2),
            epoch: 1,
            sequence: 1,
            emitted_step: 10,
            role: TeamRole::Scout,
            condition: PeerCondition::Distress,
            depth_m: 20.0,
            battery_ratio: 0.1,
            route_confidence: 0.5,
            link_quality: 0.8,
            hazard_severity: 0.9,
        };
        assert_eq!(coordinator.ingest_heartbeat(heartbeat, 10), Ok(()));
        let assessment = coordinator.assess(
            12,
            5.0,
            TunnelDirection::Holding,
            ReservationPriority::Routine,
            2.0,
            1.0,
            false,
            0.5,
        );
        assert_eq!(assessment.distress_target, Some(AgentId::new(2)));
        assert_eq!(assessment.directive, TeamDirective::MaintainRelay);
        assert_eq!(assessment.rescue_state, RescueHandoffState::Idle);
    }

    fn trusted_peer_assertion(agent_id: u64, sequence: u64) -> PeerAuthenticationAssertion {
        PeerAuthenticationAssertion {
            schema_version: crate::peer_trust::PEER_TRUST_SCHEMA_VERSION,
            agent_id: AgentId::new(agent_id),
            deployment_id: 7,
            epoch: 1,
            sequence,
            issued_step: 1,
            expires_step: 100,
            authentication_verified: true,
            hardware_backed: true,
        }
    }

    #[test]
    fn conflicting_trusted_leadership_leases_hold_team_motion() {
        let mut coordinator = TeamCoordinator::new_with_deployment(AgentId::new(1), 7);
        assert_eq!(
            coordinator.ingest_peer_assertion(trusted_peer_assertion(2, 1)),
            Ok(())
        );
        assert_eq!(
            coordinator.ingest_peer_assertion(trusted_peer_assertion(3, 1)),
            Ok(())
        );
        assert_eq!(
            coordinator.ingest_leadership_vote(LeadershipLeaseVote {
                schema_version: crate::team_leadership::TEAM_LEADERSHIP_SCHEMA_VERSION,
                reporter: AgentId::new(2),
                leader: AgentId::new(2),
                term: 4,
                membership_digest: 11,
                epoch: 1,
                sequence: 1,
                issued_step: 2,
                expires_step: 100,
            }),
            Ok(())
        );
        assert_eq!(
            coordinator.ingest_leadership_vote(LeadershipLeaseVote {
                schema_version: crate::team_leadership::TEAM_LEADERSHIP_SCHEMA_VERSION,
                reporter: AgentId::new(3),
                leader: AgentId::new(3),
                term: 4,
                membership_digest: 11,
                epoch: 1,
                sequence: 1,
                issued_step: 2,
                expires_step: 100,
            }),
            Ok(())
        );
        let assessment = coordinator.assess(
            3,
            5.0,
            TunnelDirection::Holding,
            ReservationPriority::Routine,
            2.0,
            1.0,
            true,
            0.8,
        );
        assert_eq!(assessment.directive, TeamDirective::HoldForQuorum);
        assert_eq!(
            assessment.byzantine_containment.authority,
            ByzantineContainmentAuthority::HoldForQuorum
        );
    }

    #[test]
    fn split_brain_containment_survives_team_checkpoint_restore() {
        let mut original = TeamCoordinator::new_with_deployment(AgentId::new(1), 7);
        original
            .ingest_peer_assertion(trusted_peer_assertion(2, 1))
            .unwrap();
        original
            .ingest_peer_assertion(trusted_peer_assertion(3, 1))
            .unwrap();
        for (reporter, leader) in [(2, 2), (3, 3)] {
            original
                .ingest_leadership_vote(LeadershipLeaseVote {
                    schema_version: crate::team_leadership::TEAM_LEADERSHIP_SCHEMA_VERSION,
                    reporter: AgentId::new(reporter),
                    leader: AgentId::new(leader),
                    term: 9,
                    membership_digest: 22,
                    epoch: 1,
                    sequence: 1,
                    issued_step: 4,
                    expires_step: 100,
                })
                .unwrap();
        }
        original.assess(
            5,
            5.0,
            TunnelDirection::Holding,
            ReservationPriority::Routine,
            2.0,
            1.0,
            true,
            0.8,
        );
        let checkpoint = original.recovery_checkpoint();
        let mut restored = TeamCoordinator::new_with_deployment(AgentId::new(1), 7);
        assert!(restored.load_recovery_checkpoint(&checkpoint));
        let assessment = restored.assess(
            6,
            5.0,
            TunnelDirection::Holding,
            ReservationPriority::Routine,
            2.0,
            1.0,
            true,
            0.8,
        );
        assert_eq!(assessment.directive, TeamDirective::HoldForQuorum);
    }
}
