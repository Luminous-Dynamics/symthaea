// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic acceptance contracts for mission-scale autonomy.
//!
//! These contracts target cross-module claims that unit tests alone can miss:
//! safe route choice, resource admission, mid-work abort, mechanical authority,
//! and restart continuity.

use crate::embodiment::MotorSafetyLevel;
use crate::maintenance::{ComponentKind, MaintenanceMonitor};
use crate::mission_executive::{ExecutiveAbortReason, ExecutiveDirective, MissionExecutive};
use crate::safety::{HazardAssessment, SubterraneanHazard};
use crate::simulator::RecoveryResources;
use crate::team_operations::TeamDirective;
use crate::tunnel_graph::{RouteCostPolicy, TunnelEdge, TunnelNode, TunnelNodeId, TunnelNodeKind};
use crate::types::{BATTERY_RATIO, SubterraneanCommand, SubterraneanState};
use crate::work_orders::{
    WorkKind, WorkOrder, WorkOrderId, WorkPriority, WorkResourceEstimate, WorkStatus,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LongHorizonContract {
    SaferRoutePreferred,
    UnderfundedWorkRejected,
    MidWorkReserveAbort,
    HardwareFailureRemovesAuthority,
    OperationalCheckpointContinuity,
}

impl LongHorizonContract {
    pub const ALL: [Self; 5] = [
        Self::SaferRoutePreferred,
        Self::UnderfundedWorkRejected,
        Self::MidWorkReserveAbort,
        Self::HardwareFailureRemovesAuthority,
        Self::OperationalCheckpointContinuity,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::SaferRoutePreferred => "safer_route_preferred",
            Self::UnderfundedWorkRejected => "underfunded_work_rejected",
            Self::MidWorkReserveAbort => "mid_work_reserve_abort",
            Self::HardwareFailureRemovesAuthority => "hardware_failure_removes_authority",
            Self::OperationalCheckpointContinuity => "operational_checkpoint_continuity",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LongHorizonGateFailure {
    pub contract: LongHorizonContract,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LongHorizonValidationReport {
    pub passed: bool,
    pub evaluated_contracts: usize,
    pub failures: Vec<LongHorizonGateFailure>,
}

pub struct LongHorizonValidator;

impl LongHorizonValidator {
    fn node(id: u32, kind: TunnelNodeKind, depth_m: f64) -> TunnelNode {
        TunnelNode {
            id: TunnelNodeId(id),
            kind,
            depth_m,
            survey_confidence: 0.95,
        }
    }

    fn edge(from: u32, to: u32, length_m: f64, risk: f64) -> TunnelEdge {
        TunnelEdge {
            from: TunnelNodeId(from),
            to: TunnelNodeId(to),
            length_m,
            energy_per_m: 0.001,
            obstruction_risk: risk,
            water_risk: risk * 0.5,
            roof_risk: risk * 0.5,
            confidence: 0.95,
            traversable: true,
            bidirectional: true,
            revision: 1,
        }
    }

    fn work(id: u64, target: u32, battery_fraction: f64) -> WorkOrder {
        WorkOrder {
            id: WorkOrderId(id),
            kind: WorkKind::Bore,
            target: TunnelNodeId(target),
            priority: WorkPriority::Important,
            prerequisites: [None; 4],
            estimated_steps: 20,
            deadline_step: None,
            resources: WorkResourceEstimate {
                battery_fraction,
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

    fn reference_executive() -> MissionExecutive {
        let mut executive =
            MissionExecutive::new(Self::node(0, TunnelNodeKind::Surface, 0.0)).unwrap_or_default();
        let additions = [
            Self::node(1, TunnelNodeKind::Junction, 10.0),
            Self::node(2, TunnelNodeKind::Junction, 12.0),
            Self::node(3, TunnelNodeKind::Workface, 20.0),
        ];
        for node in additions {
            let _ = executive.add_tunnel_node(node);
        }
        for edge in [
            Self::edge(0, 1, 10.0, 0.9),
            Self::edge(1, 3, 10.0, 0.9),
            Self::edge(0, 2, 16.0, 0.05),
            Self::edge(2, 3, 16.0, 0.05),
        ] {
            let _ = executive.upsert_tunnel_edge(edge);
        }
        executive
    }

    fn evaluate(contract: LongHorizonContract) -> Result<(), String> {
        match contract {
            LongHorizonContract::SaferRoutePreferred => {
                let executive = Self::reference_executive();
                let route = executive
                    .graph()
                    .route(TunnelNodeId(0), TunnelNodeId(3), RouteCostPolicy::default())
                    .map_err(|error| format!("route failed: {error:?}"))?;
                if route.nodes != vec![TunnelNodeId(0), TunnelNodeId(2), TunnelNodeId(3)] {
                    return Err(format!("selected unexpected route: {:?}", route.nodes));
                }
                if route.maximum_risk >= 0.1 {
                    return Err("selected route exceeded preregistered risk".to_string());
                }
                Ok(())
            }
            LongHorizonContract::UnderfundedWorkRejected => {
                let mut executive = Self::reference_executive();
                executive
                    .submit_work(Self::work(1, 3, 0.8))
                    .map_err(|error| format!("submit failed: {error:?}"))?;
                let mut state = SubterraneanState::home();
                state.channels[BATTERY_RATIO] = 0.35;
                state.channels[crate::types::DEPTH_M] = 5.0;
                let assessment = executive.assess(
                    0,
                    &state,
                    HazardAssessment::clear(),
                    TeamDirective::None,
                    RecoveryResources::full(),
                );
                if !matches!(
                    assessment.directive,
                    ExecutiveDirective::ReturnToBase(ExecutiveAbortReason::BatteryReserve)
                ) {
                    return Err(format!(
                        "underfunded work was not rejected: {:?}",
                        assessment.directive
                    ));
                }
                Ok(())
            }
            LongHorizonContract::MidWorkReserveAbort => {
                let mut executive = Self::reference_executive();
                executive
                    .submit_work(Self::work(1, 3, 0.03))
                    .map_err(|error| format!("submit failed: {error:?}"))?;
                let nominal = executive.assess(
                    0,
                    &SubterraneanState::home(),
                    HazardAssessment::clear(),
                    TeamDirective::None,
                    RecoveryResources::full(),
                );
                if !matches!(nominal.directive, ExecutiveDirective::Execute(_)) {
                    return Err("reference work was not initially admitted".to_string());
                }
                let mut depleted = SubterraneanState::home();
                depleted.channels[BATTERY_RATIO] = 0.18;
                depleted.channels[crate::types::DEPTH_M] = 10.0;
                let abort = executive.assess(
                    10,
                    &depleted,
                    HazardAssessment::clear(),
                    TeamDirective::None,
                    RecoveryResources::full(),
                );
                if !matches!(abort.directive, ExecutiveDirective::ReturnToBase(_)) {
                    return Err(format!(
                        "reserve loss did not abort active work: {:?}",
                        abort.directive
                    ));
                }
                if executive.scheduler().active_order().is_some() {
                    return Err("aborted work retained active authority".to_string());
                }
                Ok(())
            }
            LongHorizonContract::HardwareFailureRemovesAuthority => {
                let mut maintenance = MaintenanceMonitor::new();
                maintenance.set_health_for_test(ComponentKind::Cutter, 0.01);
                let mut command = SubterraneanCommand::zero();
                command.set_cutter_head(1.0);
                command.set_left_track(-0.5);
                command.set_right_track(-0.5);
                let derated = maintenance.derate_command(command);
                if derated.cutter_head() != 0.0 {
                    return Err("failed cutter retained actuator authority".to_string());
                }
                if derated.left_track() >= -0.1 || derated.right_track() >= -0.1 {
                    return Err("unrelated healthy return mobility was removed".to_string());
                }
                Ok(())
            }
            LongHorizonContract::OperationalCheckpointContinuity => {
                let mut executive = Self::reference_executive();
                executive
                    .submit_work(Self::work(1, 3, 0.03))
                    .map_err(|error| format!("submit failed: {error:?}"))?;
                let _ = executive.assess(
                    0,
                    &SubterraneanState::home(),
                    HazardAssessment::clear(),
                    TeamDirective::None,
                    RecoveryResources::full(),
                );
                let checkpoint = executive.checkpoint();
                let mut restored = MissionExecutive::default();
                restored
                    .load_checkpoint(&checkpoint)
                    .map_err(|error| format!("restore failed: {error:?}"))?;
                if restored.scheduler().active_order().map(|order| order.id) != Some(WorkOrderId(1))
                {
                    return Err("active work identity was not restored".to_string());
                }
                if restored.graph().edges().len() != executive.graph().edges().len() {
                    return Err("tunnel topology changed across checkpoint".to_string());
                }
                Ok(())
            }
        }
    }

    pub fn evaluate_reference() -> LongHorizonValidationReport {
        let mut failures = Vec::new();
        for contract in LongHorizonContract::ALL {
            if let Err(detail) = Self::evaluate(contract) {
                failures.push(LongHorizonGateFailure { contract, detail });
            }
        }
        LongHorizonValidationReport {
            passed: failures.is_empty(),
            evaluated_contracts: LongHorizonContract::ALL.len(),
            failures,
        }
    }

    pub fn hazard_preemption_fixture() -> HazardAssessment {
        HazardAssessment {
            primary: SubterraneanHazard::Gas,
            safety_level: MotorSafetyLevel::Red,
            severity: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_long_horizon_contracts_pass() {
        let report = LongHorizonValidator::evaluate_reference();
        assert!(report.passed, "{:?}", report.failures);
        assert_eq!(report.evaluated_contracts, LongHorizonContract::ALL.len());
    }

    #[test]
    fn hazard_fixture_is_fail_closed() {
        let hazard = LongHorizonValidator::hazard_preemption_fixture();
        assert_eq!(hazard.primary, SubterraneanHazard::Gas);
        assert_eq!(hazard.safety_level, MotorSafetyLevel::Red);
    }
}
