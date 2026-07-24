// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic acceptance contracts for multi-agent subterranean operation.

use crate::embodiment::{MotorSafetyLevel, SubterraneanEmbodiment};
use crate::geology::GeotechnicalProfile;
use crate::occupancy::{ReservationPriority, TunnelDirection, TunnelReservation};
use crate::relay_mesh::{MeshLink, MeshNodeId, RelayMesh};
use crate::rescue::{RescueCapability, RescueCaseId, RescueRequest, evaluate_rescue};
use crate::shared_map::{SharedTunnelMap, SharedTunnelObservation};
use crate::team::{AgentId, PeerCondition, TeamHeartbeat, TeamRole};
use crate::team_operations::{TeamCoordinator, TeamDirective};
use crate::types::{SubterraneanConfig, SubterraneanState};
use serde::{Deserialize, Serialize};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TeamOperationalContract {
    SharedMapConvergence,
    TunnelConflictArrest,
    StaleMeshIsolation,
    RescueReserveProtection,
    DistressRequiresAcceptance,
}

impl TeamOperationalContract {
    pub const ALL: [Self; 5] = [
        Self::SharedMapConvergence,
        Self::TunnelConflictArrest,
        Self::StaleMeshIsolation,
        Self::RescueReserveProtection,
        Self::DistressRequiresAcceptance,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::SharedMapConvergence => "shared_map_convergence",
            Self::TunnelConflictArrest => "tunnel_conflict_arrest",
            Self::StaleMeshIsolation => "stale_mesh_isolation",
            Self::RescueReserveProtection => "rescue_reserve_protection",
            Self::DistressRequiresAcceptance => "distress_requires_acceptance",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TeamOperationalGateFailure {
    pub contract: TeamOperationalContract,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TeamOperationalValidationReport {
    pub evaluated: usize,
    pub passed: usize,
    pub failures: Vec<TeamOperationalGateFailure>,
}

impl TeamOperationalValidationReport {
    pub fn all_passed(&self) -> bool {
        self.failures.is_empty() && self.evaluated == self.passed
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TeamOperationalValidator;

impl TeamOperationalValidator {
    pub fn validate_all(self) -> TeamOperationalValidationReport {
        let mut failures = Vec::new();
        for contract in TeamOperationalContract::ALL {
            if let Err(reason) = self.validate(contract) {
                failures.push(TeamOperationalGateFailure { contract, reason });
            }
        }
        TeamOperationalValidationReport {
            evaluated: TeamOperationalContract::ALL.len(),
            passed: TeamOperationalContract::ALL.len() - failures.len(),
            failures,
        }
    }

    pub fn validate(self, contract: TeamOperationalContract) -> Result<(), String> {
        match contract {
            TeamOperationalContract::SharedMapConvergence => self.shared_map_convergence(),
            TeamOperationalContract::TunnelConflictArrest => self.tunnel_conflict_arrest(),
            TeamOperationalContract::StaleMeshIsolation => self.stale_mesh_isolation(),
            TeamOperationalContract::RescueReserveProtection => self.rescue_reserve_protection(),
            TeamOperationalContract::DistressRequiresAcceptance => {
                self.distress_requires_acceptance()
            }
        }
    }

    fn observation(source: u64, water: f64, roof: f64) -> SharedTunnelObservation {
        SharedTunnelObservation {
            source: AgentId::new(source),
            epoch: 1,
            sequence: 1,
            observed_step: 10,
            bin_index: 4,
            minimum_depth_m: 4.0,
            maximum_depth_m: 5.0,
            roof_stability: roof,
            water_ingress: water,
            slurry_load: 0.1,
            localization_confidence: 0.9,
            survey_confidence: 0.8,
            roof_supported: false,
        }
    }

    fn shared_map_convergence(self) -> Result<(), String> {
        let wet = Self::observation(2, 0.8, 0.9);
        let weak = Self::observation(3, 0.1, 0.2);
        let mut left = SharedTunnelMap::default();
        let mut right = SharedTunnelMap::default();
        left.merge(wet).map_err(|error| format!("{error:?}"))?;
        left.merge(weak).map_err(|error| format!("{error:?}"))?;
        right.merge(weak).map_err(|error| format!("{error:?}"))?;
        right.merge(wet).map_err(|error| format!("{error:?}"))?;
        if left.aggregate_bin(4) != right.aggregate_bin(4) {
            return Err("arrival order changed the conservative aggregate".to_string());
        }
        Ok(())
    }

    fn tunnel_conflict_arrest(self) -> Result<(), String> {
        let genesis = GenesisSeed::from_phrase("team-validation-conflict");
        let mut embodiment = SubterraneanEmbodiment::with_config_geology_and_agent(
            &genesis,
            SubterraneanConfig::default(),
            GeotechnicalProfile::default(),
            AgentId::new(3),
        );
        embodiment
            .ingest_tunnel_reservation(TunnelReservation {
                agent_id: AgentId::new(2),
                epoch: 1,
                sequence: 1,
                issued_step: 0,
                valid_from_step: 0,
                valid_until_step: 100,
                minimum_depth_m: 0.0,
                maximum_depth_m: 5.0,
                direction: TunnelDirection::Inbound,
                priority: ReservationPriority::Emergency,
            })
            .map_err(|error| format!("reservation rejected: {error:?}"))?;
        let thought = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 9);
        let result = embodiment.step(&thought, 0.005, 0.9);
        if result.safety_level != MotorSafetyLevel::Red
            || embodiment.last_command().cutter_head() != 0.0
            || embodiment.last_command().left_track() != 0.0
        {
            return Err("conflict did not arrest cutter and track motion".to_string());
        }
        Ok(())
    }

    fn stale_mesh_isolation(self) -> Result<(), String> {
        let agent = AgentId::new(2);
        let mut mesh = RelayMesh::new(4, 5);
        mesh.merge(MeshLink {
            first: MeshNodeId::Agent(agent),
            second: MeshNodeId::Surface,
            epoch: 1,
            sequence: 1,
            observed_step: 10,
            quality: 0.9,
            capacity_ratio: 1.0,
        })
        .map_err(|error| format!("mesh link rejected: {error:?}"))?;
        let assessment = mesh.assess_surface(agent, 16);
        if assessment.reachable || assessment.stale_links_ignored != 1 {
            return Err("stale link remained authoritative".to_string());
        }
        Ok(())
    }

    fn rescue_reserve_protection(self) -> Result<(), String> {
        let request = RescueRequest {
            case_id: RescueCaseId(1),
            requester: AgentId::new(2),
            epoch: 1,
            sequence: 1,
            issued_step: 0,
            expires_step: 100,
            depth_m: 40.0,
            battery_ratio: 0.1,
            route_confidence: 0.5,
            hazard_severity: 0.9,
            capability: RescueCapability::Extraction,
        };
        let mut own_return = crate::ReturnPathAssessment::surface();
        own_return.estimated_battery_required = 0.25;
        let mesh = crate::MeshAssessment {
            reachable: true,
            bottleneck_quality: 0.8,
            hops: 2,
            fresh_links_considered: 2,
            stale_links_ignored: 0,
        };
        let result = evaluate_rescue(request, 10, 5.0, 0.4, own_return, mesh);
        if result.feasible {
            return Err("rescue consumed protected return reserve".to_string());
        }
        Ok(())
    }

    fn distress_requires_acceptance(self) -> Result<(), String> {
        let mut coordinator = TeamCoordinator::new(AgentId::new(1));
        coordinator
            .ingest_heartbeat(
                TeamHeartbeat {
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
                },
                10,
            )
            .map_err(|error| format!("heartbeat rejected: {error:?}"))?;
        let assessment = coordinator.assess(
            12,
            SubterraneanState::home().depth_m(),
            TunnelDirection::Holding,
            ReservationPriority::Routine,
            2.0,
            1.0,
        );
        if assessment.directive == TeamDirective::AssistPeer {
            return Err("distress heartbeat bypassed explicit rescue acceptance".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_team_contracts_pass() {
        let report = TeamOperationalValidator.validate_all();
        assert!(report.all_passed(), "failures: {:?}", report.failures);
    }
}
