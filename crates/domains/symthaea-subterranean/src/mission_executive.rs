// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Long-horizon mission executive.
//!
//! The executive selects *what* work is currently admissible. It does not own
//! physical safety or direct actuator authority. Every selected work order is
//! still constrained by the hazard supervisor, recovery planner, team
//! right-of-way, maintenance derating, and the plant model.

use crate::embodiment::MotorSafetyLevel;
use crate::logistics::{AdmissionRefusal, LogisticsLedger, LogisticsPlanner, WorkAdmission};
use crate::maintenance::{MaintenanceAssessment, MaintenanceMonitor};
use crate::mission::SubterraneanMissionIntent;
use crate::safety::{HazardAssessment, SubterraneanHazard};
use crate::simulator::RecoveryResources;
use crate::team_operations::TeamDirective;
use crate::tunnel_graph::{
    BoundedTunnelGraph, RouteCostPolicy, TunnelEdge, TunnelGraphError, TunnelNode, TunnelNodeId,
    TunnelRoute,
};
use crate::types::{SubterraneanCommand, SubterraneanState};
use crate::work_orders::{
    SchedulerSnapshot, WorkOrder, WorkOrderError, WorkOrderId, WorkPreemptionReason, WorkScheduler,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutiveAbortReason {
    RouteUnavailable,
    BatteryReserve,
    ResourceUnavailable,
    MaintenanceCritical,
    Immobilized,
    PhysicalHazard,
    TeamRightOfWay,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutiveDirective {
    Idle,
    Execute(WorkOrderId),
    ReturnToBase(ExecutiveAbortReason),
    HoldPosition(ExecutiveAbortReason),
    SafetyPreempted,
}

impl ExecutiveDirective {
    pub const fn mission_override(self) -> Option<SubterraneanMissionIntent> {
        match self {
            Self::Idle | Self::SafetyPreempted => None,
            Self::Execute(_) => None,
            Self::ReturnToBase(_) => Some(SubterraneanMissionIntent::ReturnHome),
            Self::HoldPosition(_) => Some(SubterraneanMissionIntent::HoldPosition),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutiveAssessment {
    pub directive: ExecutiveDirective,
    pub selected_work: Option<WorkOrderId>,
    pub work_mission: Option<SubterraneanMissionIntent>,
    pub scheduler: SchedulerSnapshot,
    pub admission: Option<WorkAdmission>,
    pub outbound_route: Option<TunnelRoute>,
    pub return_route: Option<TunnelRoute>,
    pub maintenance: MaintenanceAssessment,
    pub current_node: TunnelNodeId,
    pub surface_node: TunnelNodeId,
}

impl ExecutiveAssessment {
    pub fn idle(current_node: TunnelNodeId, surface_node: TunnelNodeId) -> Self {
        Self {
            directive: ExecutiveDirective::Idle,
            selected_work: None,
            work_mission: None,
            scheduler: SchedulerSnapshot {
                queued: 0,
                completed: 0,
                failed: 0,
                active: None,
                last_preemption: None,
            },
            admission: None,
            outbound_route: None,
            return_route: None,
            maintenance: MaintenanceAssessment::nominal(),
            current_node,
            surface_node,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissionExecutiveError {
    Graph(TunnelGraphError),
    Work(WorkOrderError),
    UnknownCurrentNode,
    UnknownSurfaceNode,
}

impl From<TunnelGraphError> for MissionExecutiveError {
    fn from(value: TunnelGraphError) -> Self {
        Self::Graph(value)
    }
}

impl From<WorkOrderError> for MissionExecutiveError {
    fn from(value: WorkOrderError) -> Self {
        Self::Work(value)
    }
}

pub const MISSION_CHECKPOINT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissionExecutiveCheckpoint {
    pub schema_version: u32,
    pub graph: BoundedTunnelGraph,
    pub scheduler: WorkScheduler,
    pub logistics: LogisticsLedger,
    pub maintenance: MaintenanceMonitor,
    pub current_node: TunnelNodeId,
    pub surface_node: TunnelNodeId,
    pub route_policy: RouteCostPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissionCheckpointError {
    UnsupportedSchema { found: u32, expected: u32 },
    InvalidGraph(TunnelGraphError),
    InvalidScheduler(WorkOrderError),
    InvalidLogistics,
    InvalidMaintenance,
    InvalidCurrentNode,
    InvalidSurfaceNode,
    InvalidRoutePolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissionExecutive {
    graph: BoundedTunnelGraph,
    scheduler: WorkScheduler,
    logistics: LogisticsLedger,
    maintenance: MaintenanceMonitor,
    current_node: TunnelNodeId,
    surface_node: TunnelNodeId,
    route_policy: RouteCostPolicy,
    #[serde(skip, default = "LogisticsPlanner::default")]
    logistics_planner: LogisticsPlanner,
    #[serde(skip)]
    last: Option<ExecutiveAssessment>,
}

impl MissionExecutive {
    pub fn new(surface: TunnelNode) -> Result<Self, MissionExecutiveError> {
        let mut graph = BoundedTunnelGraph::new();
        graph.add_node(surface)?;
        Ok(Self {
            graph,
            scheduler: WorkScheduler::new(),
            logistics: LogisticsLedger::new(),
            maintenance: MaintenanceMonitor::new(),
            current_node: surface.id,
            surface_node: surface.id,
            route_policy: RouteCostPolicy::default(),
            logistics_planner: LogisticsPlanner::default(),
            last: None,
        })
    }

    pub fn graph(&self) -> &BoundedTunnelGraph {
        &self.graph
    }

    pub fn scheduler(&self) -> &WorkScheduler {
        &self.scheduler
    }

    pub fn logistics(&self) -> LogisticsLedger {
        self.logistics
    }

    pub fn maintenance(&self) -> &MaintenanceMonitor {
        &self.maintenance
    }

    pub fn add_tunnel_node(&mut self, node: TunnelNode) -> Result<(), MissionExecutiveError> {
        self.graph.add_node(node)?;
        Ok(())
    }

    pub fn upsert_tunnel_edge(&mut self, edge: TunnelEdge) -> Result<(), MissionExecutiveError> {
        self.graph.upsert_edge(edge)?;
        Ok(())
    }

    pub fn submit_work(&mut self, order: WorkOrder) -> Result<(), MissionExecutiveError> {
        if self.graph.node(order.target).is_none() {
            return Err(MissionExecutiveError::Graph(
                TunnelGraphError::MissingEndpoint,
            ));
        }
        self.scheduler.submit(order)?;
        Ok(())
    }

    pub fn set_current_node(&mut self, node: TunnelNodeId) -> Result<(), MissionExecutiveError> {
        if self.graph.node(node).is_none() {
            return Err(MissionExecutiveError::UnknownCurrentNode);
        }
        self.current_node = node;
        Ok(())
    }

    fn preempt_if_active(&mut self, reason: WorkPreemptionReason) {
        if self.scheduler.active_order().is_some() {
            let _ = self.scheduler.preempt(reason);
        }
    }

    fn refusal_reason(refusal: AdmissionRefusal) -> ExecutiveAbortReason {
        match refusal {
            AdmissionRefusal::BatteryReserve => ExecutiveAbortReason::BatteryReserve,
            AdmissionRefusal::NoOutboundRoute | AdmissionRefusal::NoReturnRoute => {
                ExecutiveAbortReason::RouteUnavailable
            }
            AdmissionRefusal::InvalidEstimate
            | AdmissionRefusal::Sealant
            | AdmissionRefusal::Relay
            | AdmissionRefusal::RoofSupport
            | AdmissionRefusal::SampleCapacity
            | AdmissionRefusal::SpoilCapacity
            | AdmissionRefusal::CoolantUnavailable => ExecutiveAbortReason::ResourceUnavailable,
        }
    }

    pub fn assess(
        &mut self,
        step: u64,
        state: &SubterraneanState,
        hazard: HazardAssessment,
        team_directive: TeamDirective,
        recovery: RecoveryResources,
    ) -> ExecutiveAssessment {
        let maintenance = self.maintenance.assessment();
        if maintenance.mission_abort_required {
            self.preempt_if_active(WorkPreemptionReason::Maintenance);
            let reason = if maintenance.mobility_available {
                ExecutiveAbortReason::MaintenanceCritical
            } else {
                ExecutiveAbortReason::Immobilized
            };
            let directive = if maintenance.mobility_available && maintenance.cooling_available {
                ExecutiveDirective::ReturnToBase(reason)
            } else {
                ExecutiveDirective::HoldPosition(reason)
            };
            let assessment = ExecutiveAssessment {
                directive,
                selected_work: None,
                work_mission: None,
                scheduler: self.scheduler.snapshot(),
                admission: None,
                outbound_route: None,
                return_route: None,
                maintenance,
                current_node: self.current_node,
                surface_node: self.surface_node,
            };
            self.last = Some(assessment.clone());
            return assessment;
        }

        if hazard.primary != SubterraneanHazard::None
            || hazard.safety_level != MotorSafetyLevel::Green
        {
            self.preempt_if_active(WorkPreemptionReason::PhysicalHazard);
            let assessment = ExecutiveAssessment {
                directive: ExecutiveDirective::SafetyPreempted,
                selected_work: None,
                work_mission: None,
                scheduler: self.scheduler.snapshot(),
                admission: None,
                outbound_route: None,
                return_route: None,
                maintenance,
                current_node: self.current_node,
                surface_node: self.surface_node,
            };
            self.last = Some(assessment.clone());
            return assessment;
        }

        if team_directive == TeamDirective::YieldTunnel {
            self.preempt_if_active(WorkPreemptionReason::TeamRightOfWay);
            let assessment = ExecutiveAssessment {
                directive: ExecutiveDirective::HoldPosition(ExecutiveAbortReason::TeamRightOfWay),
                selected_work: None,
                work_mission: None,
                scheduler: self.scheduler.snapshot(),
                admission: None,
                outbound_route: None,
                return_route: None,
                maintenance,
                current_node: self.current_node,
                surface_node: self.surface_node,
            };
            self.last = Some(assessment.clone());
            return assessment;
        }

        let selected = self.scheduler.select_next(step);
        let Some(selected_id) = selected else {
            let assessment = ExecutiveAssessment {
                directive: ExecutiveDirective::Idle,
                selected_work: None,
                work_mission: None,
                scheduler: self.scheduler.snapshot(),
                admission: None,
                outbound_route: None,
                return_route: None,
                maintenance,
                current_node: self.current_node,
                surface_node: self.surface_node,
            };
            self.last = Some(assessment.clone());
            return assessment;
        };
        let Some(order) = self.scheduler.order(selected_id).cloned() else {
            self.preempt_if_active(WorkPreemptionReason::ResourceLimit);
            let assessment = ExecutiveAssessment {
                directive: ExecutiveDirective::HoldPosition(
                    ExecutiveAbortReason::ResourceUnavailable,
                ),
                selected_work: None,
                work_mission: None,
                scheduler: self.scheduler.snapshot(),
                admission: None,
                outbound_route: None,
                return_route: None,
                maintenance,
                current_node: self.current_node,
                surface_node: self.surface_node,
            };
            self.last = Some(assessment.clone());
            return assessment;
        };

        let outbound = self
            .graph
            .route(self.current_node, order.target, self.route_policy)
            .ok();
        let return_route = self
            .graph
            .route(order.target, self.surface_node, self.route_policy)
            .ok();
        let admission = self.logistics_planner.assess(
            &order,
            outbound.as_ref(),
            return_route.as_ref(),
            state,
            recovery,
            self.logistics,
        );
        let (directive, work_mission) = if admission.admitted {
            (
                ExecutiveDirective::Execute(selected_id),
                Some(order.kind.mission_intent()),
            )
        } else {
            self.logistics.record_refusal();
            self.preempt_if_active(WorkPreemptionReason::ResourceLimit);
            let reason = admission
                .refusal
                .map(Self::refusal_reason)
                .unwrap_or(ExecutiveAbortReason::ResourceUnavailable);
            let directive = if state.depth_m() > 0.5 {
                ExecutiveDirective::ReturnToBase(reason)
            } else {
                ExecutiveDirective::HoldPosition(reason)
            };
            (directive, None)
        };
        let assessment = ExecutiveAssessment {
            directive,
            selected_work: Some(selected_id),
            work_mission,
            scheduler: self.scheduler.snapshot(),
            admission: Some(admission),
            outbound_route: outbound,
            return_route,
            maintenance,
            current_node: self.current_node,
            surface_node: self.surface_node,
        };
        self.last = Some(assessment.clone());
        assessment
    }

    pub fn observe_post_step(
        &mut self,
        command: &SubterraneanCommand,
        state: &SubterraneanState,
        dt: f64,
        safe_to_progress: bool,
    ) {
        self.maintenance.observe(command, state, dt);
        if state.depth_m() <= 0.1 {
            self.current_node = self.surface_node;
            self.logistics.unload_at_surface();
        }
        if safe_to_progress && self.scheduler.active_order().is_some() {
            let completed_order = self
                .scheduler
                .active_order()
                .filter(|order| order.completed_steps.saturating_add(1) >= order.estimated_steps)
                .cloned();
            if self.scheduler.advance_active(1).is_ok() {
                if let Some(order) = completed_order {
                    self.logistics.apply_completion(order.resources);
                    self.current_node = order.target;
                }
            }
        }
    }

    pub fn last_assessment(&self) -> Option<&ExecutiveAssessment> {
        self.last.as_ref()
    }

    pub fn checkpoint(&self) -> MissionExecutiveCheckpoint {
        MissionExecutiveCheckpoint {
            schema_version: MISSION_CHECKPOINT_SCHEMA_VERSION,
            graph: self.graph.clone(),
            scheduler: self.scheduler.clone(),
            logistics: self.logistics,
            maintenance: self.maintenance.clone(),
            current_node: self.current_node,
            surface_node: self.surface_node,
            route_policy: self.route_policy,
        }
    }

    pub fn load_checkpoint(
        &mut self,
        checkpoint: &MissionExecutiveCheckpoint,
    ) -> Result<(), MissionCheckpointError> {
        if checkpoint.schema_version != MISSION_CHECKPOINT_SCHEMA_VERSION {
            return Err(MissionCheckpointError::UnsupportedSchema {
                found: checkpoint.schema_version,
                expected: MISSION_CHECKPOINT_SCHEMA_VERSION,
            });
        }
        checkpoint
            .graph
            .validate()
            .map_err(MissionCheckpointError::InvalidGraph)?;
        checkpoint
            .scheduler
            .validate()
            .map_err(MissionCheckpointError::InvalidScheduler)?;
        if !checkpoint.logistics.validate() {
            return Err(MissionCheckpointError::InvalidLogistics);
        }
        if !checkpoint.maintenance.validate() {
            return Err(MissionCheckpointError::InvalidMaintenance);
        }
        if checkpoint.graph.node(checkpoint.current_node).is_none() {
            return Err(MissionCheckpointError::InvalidCurrentNode);
        }
        if checkpoint.graph.node(checkpoint.surface_node).is_none() {
            return Err(MissionCheckpointError::InvalidSurfaceNode);
        }
        checkpoint
            .route_policy
            .validate()
            .map_err(|_| MissionCheckpointError::InvalidRoutePolicy)?;
        self.graph = checkpoint.graph.clone();
        self.scheduler = checkpoint.scheduler.clone();
        self.logistics = checkpoint.logistics;
        self.maintenance = checkpoint.maintenance.clone();
        self.current_node = checkpoint.current_node;
        self.surface_node = checkpoint.surface_node;
        self.route_policy = checkpoint.route_policy;
        self.logistics_planner = LogisticsPlanner::default();
        self.last = None;
        Ok(())
    }

    pub fn reset_runtime(&mut self) {
        self.scheduler.reset_runtime();
        self.maintenance.reset_runtime();
        self.current_node = self.surface_node;
        self.last = None;
    }
}

impl Default for MissionExecutive {
    fn default() -> Self {
        let surface = TunnelNode {
            id: TunnelNodeId(0),
            kind: crate::tunnel_graph::TunnelNodeKind::Surface,
            depth_m: 0.0,
            survey_confidence: 1.0,
        };
        let mut graph = BoundedTunnelGraph::new();
        let _ = graph.add_node(surface);
        Self {
            graph,
            scheduler: WorkScheduler::new(),
            logistics: LogisticsLedger::new(),
            maintenance: MaintenanceMonitor::new(),
            current_node: surface.id,
            surface_node: surface.id,
            route_policy: RouteCostPolicy::default(),
            logistics_planner: LogisticsPlanner::default(),
            last: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tunnel_graph::TunnelNodeKind;
    use crate::work_orders::{WorkKind, WorkPriority, WorkResourceEstimate, WorkStatus};

    fn node(id: u32, kind: TunnelNodeKind, depth_m: f64) -> TunnelNode {
        TunnelNode {
            id: TunnelNodeId(id),
            kind,
            depth_m,
            survey_confidence: 0.95,
        }
    }

    fn edge(from: u32, to: u32) -> TunnelEdge {
        TunnelEdge {
            from: TunnelNodeId(from),
            to: TunnelNodeId(to),
            length_m: 10.0,
            energy_per_m: 0.001,
            obstruction_risk: 0.05,
            water_risk: 0.05,
            roof_risk: 0.05,
            confidence: 0.95,
            traversable: true,
            bidirectional: true,
            revision: 1,
        }
    }

    fn work(id: u64, target: u32, battery: f64) -> WorkOrder {
        WorkOrder {
            id: WorkOrderId(id),
            kind: WorkKind::Bore,
            target: TunnelNodeId(target),
            priority: WorkPriority::Routine,
            prerequisites: [None; 4],
            estimated_steps: 2,
            deadline_step: None,
            resources: WorkResourceEstimate {
                battery_fraction: battery,
                sealant_fraction: 0.0,
                relay_units: 0,
                roof_support_units: 0,
                sample_capacity: 0.0,
                spoil_capacity: 0.05,
            },
            status: WorkStatus::Pending,
            completed_steps: 0,
        }
    }

    fn executive() -> MissionExecutive {
        let mut executive =
            MissionExecutive::new(node(0, TunnelNodeKind::Surface, 0.0)).expect("surface");
        executive
            .add_tunnel_node(node(1, TunnelNodeKind::Workface, 10.0))
            .expect("workface");
        executive.upsert_tunnel_edge(edge(0, 1)).expect("edge");
        executive
    }

    #[test]
    fn admitted_work_becomes_long_horizon_mission() {
        let mut executive = executive();
        executive.submit_work(work(1, 1, 0.02)).expect("work");
        let assessment = executive.assess(
            0,
            &SubterraneanState::home(),
            HazardAssessment::clear(),
            TeamDirective::None,
            RecoveryResources::full(),
        );
        assert_eq!(
            assessment.directive,
            ExecutiveDirective::Execute(WorkOrderId(1))
        );
        assert_eq!(
            assessment.work_mission,
            Some(SubterraneanMissionIntent::FollowVein)
        );
    }

    #[test]
    fn underfunded_work_is_preempted_before_motion() {
        let mut executive = executive();
        executive.submit_work(work(1, 1, 0.9)).expect("work");
        let mut state = SubterraneanState::home();
        state.channels[crate::types::BATTERY_RATIO] = 0.3;
        state.channels[crate::types::DEPTH_M] = 10.0;
        let assessment = executive.assess(
            0,
            &state,
            HazardAssessment::clear(),
            TeamDirective::None,
            RecoveryResources::full(),
        );
        assert!(matches!(
            assessment.directive,
            ExecutiveDirective::ReturnToBase(_)
        ));
        assert!(executive.scheduler().active_order().is_none());
    }

    #[test]
    fn physical_hazard_suspends_active_work() {
        let mut executive = executive();
        executive.submit_work(work(1, 1, 0.02)).expect("work");
        let _ = executive.assess(
            0,
            &SubterraneanState::home(),
            HazardAssessment::clear(),
            TeamDirective::None,
            RecoveryResources::full(),
        );
        let hazard = HazardAssessment {
            primary: SubterraneanHazard::Gas,
            safety_level: MotorSafetyLevel::Red,
            severity: 1.0,
        };
        let assessment = executive.assess(
            1,
            &SubterraneanState::home(),
            hazard,
            TeamDirective::None,
            RecoveryResources::full(),
        );
        assert_eq!(assessment.directive, ExecutiveDirective::SafetyPreempted);
        assert!(executive.scheduler().active_order().is_none());
    }

    #[test]
    fn mission_checkpoint_round_trip_preserves_active_work_and_health() {
        let mut source = executive();
        source.submit_work(work(1, 1, 0.02)).expect("work");
        let _ = source.assess(
            0,
            &SubterraneanState::home(),
            HazardAssessment::clear(),
            TeamDirective::None,
            RecoveryResources::full(),
        );
        source
            .maintenance
            .set_health_for_test(crate::maintenance::ComponentKind::Cutter, 0.6);
        let checkpoint = source.checkpoint();
        let mut restored = MissionExecutive::default();
        restored.load_checkpoint(&checkpoint).expect("checkpoint");
        assert_eq!(
            restored.scheduler().active_order().map(|order| order.id),
            Some(WorkOrderId(1))
        );
        assert_eq!(
            restored
                .maintenance()
                .health(crate::maintenance::ComponentKind::Cutter),
            0.6
        );
        assert_eq!(restored.graph().nodes().len(), source.graph().nodes().len());
    }
}
