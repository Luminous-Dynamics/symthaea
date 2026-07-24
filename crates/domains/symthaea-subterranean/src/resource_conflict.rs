// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Safety-monotonic arbitration for simultaneously active objectives.
//!
//! The arbiter never invents actuator authority. It admits objectives against
//! explicit resource budgets, tracks starvation and service equity, and can
//! only throttle work, require return, or hold for accountable review.

use crate::fairness_ledger::{FairnessAssessment, FairnessLedger, StakeholderId};
use crate::objective_budget::{
    ConflictObjective, ObjectiveBudget, ObjectiveClass, ObjectiveDemand, ResourceVector,
};
use crate::objective_starvation::{
    ObjectiveStarvationMonitor, StarvationAssessment, StarvationDisposition,
};
use crate::types::SubterraneanCommand;
use serde::{Deserialize, Serialize};

pub const RESOURCE_CONFLICT_SCHEMA_VERSION: u16 = 1;
pub const MAX_CONFLICT_REASONS: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ConflictAuthority {
    Nominal,
    Throttled,
    ReturnOnly,
    HoldForReview,
}

impl ConflictAuthority {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Nominal => "nominal",
            Self::Throttled => "throttled",
            Self::ReturnOnly => "return_only",
            Self::HoldForReview => "hold_for_review",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictDisposition {
    Feasible,
    SoftResourceConflict,
    ProtectedResourceConflict,
    StarvationEscalation,
    FairnessEscalation,
    InvalidInput,
}

impl ConflictDisposition {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Feasible => "feasible",
            Self::SoftResourceConflict => "soft_resource_conflict",
            Self::ProtectedResourceConflict => "protected_resource_conflict",
            Self::StarvationEscalation => "starvation_escalation",
            Self::FairnessEscalation => "fairness_escalation",
            Self::InvalidInput => "invalid_input",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceConflictAssessment {
    pub authority: ConflictAuthority,
    pub disposition: ConflictDisposition,
    pub selected_objectives: Vec<ConflictObjective>,
    pub deferred_objectives: Vec<ConflictObjective>,
    pub protected_deferred: Vec<ConflictObjective>,
    pub consumed: ResourceVector,
    pub remaining: ResourceVector,
    pub maximum_capacity_fraction: f32,
    pub starvation: StarvationAssessment,
    pub fairness: FairnessAssessment,
    pub reasons: Vec<String>,
}

impl ResourceConflictAssessment {
    pub fn nominal() -> Self {
        Self {
            authority: ConflictAuthority::Nominal,
            disposition: ConflictDisposition::Feasible,
            selected_objectives: Vec::new(),
            deferred_objectives: Vec::new(),
            protected_deferred: Vec::new(),
            consumed: ResourceVector::zero(),
            remaining: ResourceVector::unit(),
            maximum_capacity_fraction: 0.0,
            starvation: StarvationAssessment::nominal(),
            fairness: FairnessAssessment::nominal(),
            reasons: Vec::new(),
        }
    }

    pub fn productive_work_allowed(&self) -> bool {
        self.authority == ConflictAuthority::Nominal
    }

    pub fn motion_allowed(&self) -> bool {
        self.authority != ConflictAuthority::HoldForReview
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConflictSupervisor {
    schema_version: u16,
    starvation: ObjectiveStarvationMonitor,
    fairness: FairnessLedger,
    last: ResourceConflictAssessment,
}

impl Default for ResourceConflictSupervisor {
    fn default() -> Self {
        Self {
            schema_version: RESOURCE_CONFLICT_SCHEMA_VERSION,
            starvation: ObjectiveStarvationMonitor::default(),
            fairness: FairnessLedger::default(),
            last: ResourceConflictAssessment::nominal(),
        }
    }
}

impl ResourceConflictSupervisor {
    pub fn validate(&self) -> bool {
        self.schema_version == RESOURCE_CONFLICT_SCHEMA_VERSION
            && self.starvation.validate()
            && self.fairness.validate()
            && self.last.selected_objectives.len()
                <= crate::objective_budget::MAX_OBJECTIVE_DEMANDS
            && self.last.deferred_objectives.len()
                <= crate::objective_budget::MAX_OBJECTIVE_DEMANDS
            && self.last.protected_deferred.len()
                <= crate::objective_budget::MAX_OBJECTIVE_DEMANDS
            && self.last.reasons.len() <= MAX_CONFLICT_REASONS
            && self.last.consumed.validate()
            && self.last.remaining.validate()
            && self.last.maximum_capacity_fraction.is_finite()
            && (0.0..=1.0).contains(&self.last.maximum_capacity_fraction)
    }

    pub fn assess(
        &mut self,
        current_step: u64,
        budget: &ObjectiveBudget,
    ) -> ResourceConflictAssessment {
        if !budget.validate(current_step) {
            self.last = ResourceConflictAssessment {
                authority: ConflictAuthority::HoldForReview,
                disposition: ConflictDisposition::InvalidInput,
                reasons: vec!["invalid_objective_budget".to_string()],
                ..ResourceConflictAssessment::nominal()
            };
            return self.last.clone();
        }

        let mut active: Vec<ObjectiveDemand> = budget.active().collect();
        active.sort_by(|left, right| {
            right
                .objective
                .class()
                .cmp(&left.objective.class())
                .then_with(|| {
                    let left_score = left.urgency + self.starvation.debt_score(left.objective);
                    let right_score = right.urgency + self.starvation.debt_score(right.objective);
                    right_score.total_cmp(&left_score)
                })
                .then_with(|| left.objective.index().cmp(&right.objective.index()))
        });

        let mut remaining = budget.capacity;
        let mut consumed = ResourceVector::zero();
        let mut selected = Vec::new();
        let mut deferred = Vec::new();
        let mut protected_deferred = Vec::new();

        for demand in active
            .iter()
            .copied()
            .filter(|demand| demand.objective.class() == ObjectiveClass::Protected)
        {
            if demand.demand.fits_within(remaining) {
                selected.push(demand.objective);
                consumed = consumed.saturating_add(demand.demand);
                remaining = remaining.headroom_after(demand.demand);
            } else {
                deferred.push(demand.objective);
                protected_deferred.push(demand.objective);
            }
        }

        let mut discretionary_remaining = remaining.headroom_after(budget.protected_reserve);
        for demand in active
            .iter()
            .copied()
            .filter(|demand| demand.objective.class() != ObjectiveClass::Protected)
        {
            if demand.demand.fits_within(discretionary_remaining) {
                selected.push(demand.objective);
                consumed = consumed.saturating_add(demand.demand);
                remaining = remaining.headroom_after(demand.demand);
                discretionary_remaining = discretionary_remaining.headroom_after(demand.demand);
            } else {
                deferred.push(demand.objective);
            }
        }

        let active_objectives: Vec<ConflictObjective> =
            active.iter().map(|demand| demand.objective).collect();
        let starvation = self.starvation.observe(&active_objectives, &selected);
        for demand in active.iter().copied() {
            if let Some(stakeholder) = demand.stakeholder {
                let served = if selected.contains(&demand.objective) {
                    1.0
                } else {
                    0.0
                };
                let _ = self.fairness.record(
                    current_step,
                    StakeholderId(stakeholder),
                    demand.objective,
                    1.0,
                    served,
                );
            }
        }
        let fairness = self.fairness.assess();

        let mut authority = ConflictAuthority::Nominal;
        let mut disposition = ConflictDisposition::Feasible;
        let mut reasons = Vec::new();

        if !protected_deferred.is_empty()
            || starvation.disposition == StarvationDisposition::ProtectedObjectiveDeferred
        {
            authority = ConflictAuthority::HoldForReview;
            disposition = ConflictDisposition::ProtectedResourceConflict;
            push_reason(&mut reasons, "protected_objective_unfunded".into());
        } else if selected.contains(&ConflictObjective::PhysicalSafety) {
            authority = ConflictAuthority::HoldForReview;
            disposition = ConflictDisposition::ProtectedResourceConflict;
            push_reason(&mut reasons, "physical_safety_owns_resources".into());
        } else if selected.contains(&ConflictObjective::ReturnReserve) {
            authority = ConflictAuthority::ReturnOnly;
            disposition = ConflictDisposition::ProtectedResourceConflict;
            push_reason(&mut reasons, "return_reserve_owns_remaining_capacity".into());
        } else if selected.contains(&ConflictObjective::EnvironmentalContainment) {
            authority = ConflictAuthority::Throttled;
            disposition = ConflictDisposition::ProtectedResourceConflict;
            push_reason(&mut reasons, "environmental_containment_preempts_work".into());
        }

        if !deferred.is_empty() && authority < ConflictAuthority::Throttled {
            authority = ConflictAuthority::Throttled;
            disposition = ConflictDisposition::SoftResourceConflict;
            push_reason(&mut reasons, "active_objective_demand_exceeds_capacity".into());
        }
        if matches!(
            starvation.disposition,
            StarvationDisposition::Warning | StarvationDisposition::Critical
        ) && authority < ConflictAuthority::Throttled
        {
            authority = ConflictAuthority::Throttled;
            disposition = ConflictDisposition::StarvationEscalation;
            push_reason(&mut reasons, "objective_starvation_requires_replanning".into());
        }
        if !fairness.underserved.is_empty()
            && selected.contains(&ConflictObjective::MissionWork)
            && authority < ConflictAuthority::Throttled
        {
            authority = ConflictAuthority::Throttled;
            disposition = ConflictDisposition::FairnessEscalation;
            push_reason(&mut reasons, "mission_work_monopolizes_shared_capacity".into());
        }
        if active
            .iter()
            .any(|demand| demand.objective == ConflictObjective::AssetIntegrity && demand.urgency >= 0.8)
            && authority < ConflictAuthority::ReturnOnly
        {
            authority = ConflictAuthority::ReturnOnly;
            push_reason(&mut reasons, "critical_asset_integrity_service_due".into());
        }

        let maximum_capacity_fraction = consumed.maximum_fraction_of(budget.capacity);
        self.last = ResourceConflictAssessment {
            authority,
            disposition,
            selected_objectives: selected,
            deferred_objectives: deferred,
            protected_deferred,
            consumed,
            remaining,
            maximum_capacity_fraction,
            starvation,
            fairness,
            reasons,
        };
        self.last.clone()
    }

    pub fn constrain_command(&self, mut command: SubterraneanCommand) -> SubterraneanCommand {
        match self.last.authority {
            ConflictAuthority::Nominal => {}
            ConflictAuthority::Throttled => {
                command.set_cutter_head(command.cutter_head().clamp(0.0, 0.35));
                command.set_auger_feed(command.auger_feed().clamp(0.0, 0.30));
                command.set_left_track(command.left_track().clamp(-0.5, 0.5));
                command.set_right_track(command.right_track().clamp(-0.5, 0.5));
                command.set_ballast_trim(command.ballast_trim().clamp(-0.25, 0.25));
            }
            ConflictAuthority::ReturnOnly => {
                command.set_cutter_head(0.0);
                command.set_auger_feed(0.0);
                command.set_left_track(command.left_track().min(0.0));
                command.set_right_track(command.right_track().min(0.0));
                command.set_ballast_trim(0.0);
            }
            ConflictAuthority::HoldForReview => {
                command.set_cutter_head(0.0);
                command.set_auger_feed(0.0);
                command.set_left_track(0.0);
                command.set_right_track(0.0);
                command.set_ballast_trim(0.0);
            }
        }
        command.sanitize();
        command
    }

    pub fn last(&self) -> &ResourceConflictAssessment {
        &self.last
    }

    pub fn fairness(&self) -> &FairnessLedger {
        &self.fairness
    }

    pub fn starvation(&self) -> &ObjectiveStarvationMonitor {
        &self.starvation
    }
}

fn push_reason(reasons: &mut Vec<String>, reason: String) {
    if reasons.len() < MAX_CONFLICT_REASONS && !reasons.contains(&reason) {
        reasons.push(reason);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn demand(
        objective: ConflictObjective,
        urgency: f32,
        battery: f32,
    ) -> ObjectiveDemand {
        ObjectiveDemand {
            objective,
            active: true,
            urgency,
            demand: ResourceVector {
                battery,
                thermal: 0.1,
                time: 0.1,
                recovery: 0.0,
            },
            deadline_step: None,
            stakeholder: None,
        }
    }

    #[test]
    fn return_reserve_preempts_productive_work() {
        let mut budget = ObjectiveBudget::default();
        assert!(budget.push(demand(ConflictObjective::ReturnReserve, 1.0, 0.3)));
        assert!(budget.push(demand(ConflictObjective::MissionWork, 0.5, 0.3)));
        let mut supervisor = ResourceConflictSupervisor::default();
        let assessment = supervisor.assess(1, &budget);
        assert_eq!(assessment.authority, ConflictAuthority::ReturnOnly);
        let mut command = SubterraneanCommand::zero();
        command.set_cutter_head(1.0);
        command.set_left_track(0.7);
        let command = supervisor.constrain_command(command);
        assert_eq!(command.cutter_head(), 0.0);
        assert_eq!(command.left_track(), 0.0);
    }

    #[test]
    fn protected_demand_that_cannot_fit_forces_hold() {
        let mut budget = ObjectiveBudget::new(
            ResourceVector {
                battery: 0.1,
                thermal: 0.1,
                time: 0.1,
                recovery: 0.1,
            },
            ResourceVector::zero(),
        );
        assert!(budget.push(demand(ConflictObjective::PhysicalSafety, 1.0, 0.9)));
        let assessment = ResourceConflictSupervisor::default().assess(1, &budget);
        assert_eq!(assessment.authority, ConflictAuthority::HoldForReview);
        assert_eq!(assessment.protected_deferred.len(), 1);
    }

    #[test]
    fn soft_conflict_throttles_but_does_not_select_motion() {
        let mut budget = ObjectiveBudget::new(
            ResourceVector {
                battery: 0.4,
                thermal: 0.4,
                time: 0.4,
                recovery: 0.4,
            },
            ResourceVector::zero(),
        );
        assert!(budget.push(demand(ConflictObjective::MissionWork, 0.5, 0.3)));
        assert!(budget.push(demand(ConflictObjective::Communications, 0.4, 0.3)));
        let assessment = ResourceConflictSupervisor::default().assess(1, &budget);
        assert_eq!(assessment.authority, ConflictAuthority::Throttled);
        assert_eq!(assessment.disposition, ConflictDisposition::SoftResourceConflict);
    }
}
