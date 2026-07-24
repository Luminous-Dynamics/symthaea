// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic release contracts for resource-conflict assurance.

use crate::embodiment::SubterraneanEmbodiment;
use crate::genesis::GenesisSeed;
use crate::objective_budget::{
    ConflictObjective, ObjectiveBudget, ObjectiveDemand, ResourceVector,
};
use crate::resource_conflict::{ConflictAuthority, ResourceConflictSupervisor};
use crate::types::SubterraneanCommand;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceConflictContract {
    ProtectedObjectivesFirst,
    ProtectedReservePreserved,
    DeterministicTieBreak,
    StarvationCannotAddMotion,
    FairnessThrottleIsMonotonic,
    SameFrameCommandRestriction,
    CheckpointStateValid,
}

impl ResourceConflictContract {
    pub const ALL: [Self; 7] = [
        Self::ProtectedObjectivesFirst,
        Self::ProtectedReservePreserved,
        Self::DeterministicTieBreak,
        Self::StarvationCannotAddMotion,
        Self::FairnessThrottleIsMonotonic,
        Self::SameFrameCommandRestriction,
        Self::CheckpointStateValid,
    ];
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceConflictGateFailure {
    pub contract: ResourceConflictContract,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceConflictValidationReport {
    pub passed: Vec<ResourceConflictContract>,
    pub failures: Vec<ResourceConflictGateFailure>,
}

impl ResourceConflictValidationReport {
    pub fn passes(&self) -> bool {
        self.failures.is_empty()
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ResourceConflictValidator;

impl ResourceConflictValidator {
    pub fn validate(self) -> ResourceConflictValidationReport {
        let mut passed = Vec::new();
        let mut failures = Vec::new();
        for contract in ResourceConflictContract::ALL {
            match evaluate(contract) {
                Ok(()) => passed.push(contract),
                Err(detail) => failures.push(ResourceConflictGateFailure { contract, detail }),
            }
        }
        ResourceConflictValidationReport { passed, failures }
    }
}

fn demand(
    objective: ConflictObjective,
    urgency: f32,
    resources: ResourceVector,
    stakeholder: Option<u64>,
) -> ObjectiveDemand {
    ObjectiveDemand {
        objective,
        active: true,
        urgency,
        demand: resources,
        deadline_step: None,
        stakeholder,
    }
}

fn small() -> ResourceVector {
    ResourceVector {
        battery: 0.2,
        thermal: 0.1,
        time: 0.1,
        recovery: 0.0,
    }
}

fn evaluate(contract: ResourceConflictContract) -> Result<(), String> {
    match contract {
        ResourceConflictContract::ProtectedObjectivesFirst => {
            let mut budget = ObjectiveBudget::new(ResourceVector::unit(), ResourceVector::zero());
            let _ = budget.push(demand(
                ConflictObjective::MissionWork,
                1.0,
                ResourceVector {
                    battery: 0.8,
                    thermal: 0.8,
                    time: 0.8,
                    recovery: 0.2,
                },
                Some(1),
            ));
            let _ = budget.push(demand(
                ConflictObjective::ReturnReserve,
                0.5,
                ResourceVector {
                    battery: 0.4,
                    thermal: 0.1,
                    time: 0.2,
                    recovery: 0.0,
                },
                None,
            ));
            let assessment = ResourceConflictSupervisor::default().assess(1, &budget);
            (assessment
                .selected_objectives
                .contains(&ConflictObjective::ReturnReserve)
                && assessment.authority == ConflictAuthority::ReturnOnly)
                .then_some(())
                .ok_or_else(|| "mission urgency displaced protected return reserve".to_string())
        }
        ResourceConflictContract::ProtectedReservePreserved => {
            let mut budget = ObjectiveBudget::new(
                ResourceVector::unit(),
                ResourceVector {
                    battery: 0.4,
                    thermal: 0.3,
                    time: 0.2,
                    recovery: 0.2,
                },
            );
            let _ = budget.push(demand(
                ConflictObjective::MissionWork,
                1.0,
                ResourceVector {
                    battery: 0.8,
                    thermal: 0.8,
                    time: 0.8,
                    recovery: 0.1,
                },
                Some(1),
            ));
            let assessment = ResourceConflictSupervisor::default().assess(1, &budget);
            (!assessment
                .selected_objectives
                .contains(&ConflictObjective::MissionWork)
                && assessment.authority == ConflictAuthority::Throttled)
                .then_some(())
                .ok_or_else(|| "discretionary work consumed protected reserve".to_string())
        }
        ResourceConflictContract::DeterministicTieBreak => {
            let build = || {
                let mut budget = ObjectiveBudget::new(
                    ResourceVector {
                        battery: 0.3,
                        thermal: 0.3,
                        time: 0.3,
                        recovery: 0.3,
                    },
                    ResourceVector::zero(),
                );
                let _ = budget.push(demand(
                    ConflictObjective::Communications,
                    0.5,
                    small(),
                    Some(2),
                ));
                let _ = budget.push(demand(
                    ConflictObjective::MissionWork,
                    0.5,
                    small(),
                    Some(1),
                ));
                budget
            };
            let left = ResourceConflictSupervisor::default().assess(1, &build());
            let right = ResourceConflictSupervisor::default().assess(1, &build());
            (left.selected_objectives == right.selected_objectives
                && left.deferred_objectives == right.deferred_objectives)
                .then_some(())
                .ok_or_else(|| "equal-score arbitration was not deterministic".to_string())
        }
        ResourceConflictContract::StarvationCannotAddMotion => {
            let mut supervisor = ResourceConflictSupervisor::default();
            let mut budget = ObjectiveBudget::new(
                ResourceVector {
                    battery: 0.1,
                    thermal: 0.1,
                    time: 0.1,
                    recovery: 0.1,
                },
                ResourceVector::zero(),
            );
            let _ = budget.push(demand(
                ConflictObjective::PeerAssistance,
                0.7,
                ResourceVector {
                    battery: 0.9,
                    thermal: 0.2,
                    time: 0.2,
                    recovery: 0.0,
                },
                Some(9),
            ));
            for step in 0..1_100 {
                supervisor.assess(step, &budget);
            }
            let command = supervisor.constrain_command(SubterraneanCommand::zero());
            (command == SubterraneanCommand::zero()
                && supervisor.last().authority >= ConflictAuthority::Throttled)
                .then_some(())
                .ok_or_else(|| "starvation created actuator authority".to_string())
        }
        ResourceConflictContract::FairnessThrottleIsMonotonic => {
            let mut supervisor = ResourceConflictSupervisor::default();
            let mut budget = ObjectiveBudget::new(
                ResourceVector {
                    battery: 0.35,
                    thermal: 0.35,
                    time: 0.35,
                    recovery: 0.35,
                },
                ResourceVector::zero(),
            );
            let _ = budget.push(demand(
                ConflictObjective::MissionWork,
                1.0,
                small(),
                Some(1),
            ));
            let _ = budget.push(demand(
                ConflictObjective::PeerAssistance,
                0.1,
                ResourceVector {
                    battery: 0.3,
                    thermal: 0.3,
                    time: 0.3,
                    recovery: 0.0,
                },
                Some(2),
            ));
            for step in 0..5 {
                supervisor.assess(step, &budget);
            }
            let mut command = SubterraneanCommand::zero();
            command.set_cutter_head(1.0);
            let constrained = supervisor.constrain_command(command);
            (constrained.cutter_head() <= command.cutter_head()
                && constrained.control_effort() <= command.control_effort())
                .then_some(())
                .ok_or_else(|| "fairness handling increased command authority".to_string())
        }
        ResourceConflictContract::SameFrameCommandRestriction => {
            let mut budget = ObjectiveBudget::default();
            let _ = budget.push(demand(
                ConflictObjective::ReturnReserve,
                1.0,
                small(),
                None,
            ));
            let mut supervisor = ResourceConflictSupervisor::default();
            supervisor.assess(1, &budget);
            let mut command = SubterraneanCommand::zero();
            command.set_cutter_head(1.0);
            command.set_left_track(0.8);
            command.recovery.dewatering_pump = 1.0;
            let constrained = supervisor.constrain_command(command);
            (constrained.cutter_head() == 0.0
                && constrained.left_track() == 0.0
                && constrained.recovery.dewatering_pump == 1.0)
                .then_some(())
                .ok_or_else(|| "return-only restriction failed in same frame".to_string())
        }
        ResourceConflictContract::CheckpointStateValid => {
            let genesis = GenesisSeed::from_phrase("resource-conflict-checkpoint-contract");
            let original = SubterraneanEmbodiment::new(&genesis);
            let mut checkpoint = original.operational_checkpoint();
            let mut budget = ObjectiveBudget::new(
                ResourceVector {
                    battery: 0.2,
                    thermal: 0.2,
                    time: 0.2,
                    recovery: 0.2,
                },
                ResourceVector::zero(),
            );
            let _ = budget.push(demand(
                ConflictObjective::PeerAssistance,
                0.9,
                ResourceVector {
                    battery: 0.8,
                    thermal: 0.2,
                    time: 0.3,
                    recovery: 0.1,
                },
                Some(77),
            ));
            for step in 0..16 {
                checkpoint.resource_conflict.assess(step, &budget);
            }
            let expected = checkpoint.resource_conflict.last().clone();
            let mut restored = SubterraneanEmbodiment::new(&genesis);
            restored
                .load_operational_checkpoint(&checkpoint)
                .map_err(|error| format!("resource-conflict checkpoint rejected: {error:?}"))?;
            (restored.resource_conflict().validate()
                && restored.resource_conflict_assessment() == &expected)
                .then_some(())
                .ok_or_else(|| {
                    "resource conflict debt fairness or authority did not survive restart"
                        .to_string()
                })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_resource_conflict_release_contracts_pass() {
        let report = ResourceConflictValidator.validate();
        assert!(report.passes(), "{report:#?}");
    }
}
