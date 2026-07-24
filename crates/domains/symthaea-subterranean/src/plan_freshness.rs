// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Revision-bound plan freshness and invalidation.

use serde::{Deserialize, Serialize};

pub const PLAN_FRESHNESS_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct RuntimeRevisions {
    pub state: u64,
    pub hazard: u64,
    pub topology: u64,
    pub calibration: u64,
    pub mission: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanBasis {
    pub plan_id: u64,
    pub created_step: u64,
    pub expires_step: u64,
    pub revisions: RuntimeRevisions,
    pub permits_productive_work: bool,
}

impl PlanBasis {
    pub fn validate(self) -> bool {
        self.plan_id > 0 && self.created_step <= self.expires_step
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlanInvalidationReason {
    None,
    Malformed,
    Expired,
    FuturePlan,
    StateChanged,
    HazardChanged,
    TopologyChanged,
    CalibrationChanged,
    MissionChanged,
}

impl PlanInvalidationReason {
    pub const fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Malformed => "malformed",
            Self::Expired => "expired",
            Self::FuturePlan => "future_plan",
            Self::StateChanged => "state_changed",
            Self::HazardChanged => "hazard_changed",
            Self::TopologyChanged => "topology_changed",
            Self::CalibrationChanged => "calibration_changed",
            Self::MissionChanged => "mission_changed",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanFreshnessAssessment {
    pub current: bool,
    pub work_authorized: bool,
    pub reason: PlanInvalidationReason,
    pub age_steps: u64,
}

impl PlanFreshnessAssessment {
    pub const fn nominal() -> Self {
        Self {
            current: true,
            work_authorized: true,
            reason: PlanInvalidationReason::None,
            age_steps: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanFreshnessSupervisor {
    schema_version: u16,
    current: Option<PlanBasis>,
    invalidations: u64,
    last: PlanFreshnessAssessment,
}

impl Default for PlanFreshnessSupervisor {
    fn default() -> Self {
        Self {
            schema_version: PLAN_FRESHNESS_SCHEMA_VERSION,
            current: None,
            invalidations: 0,
            last: PlanFreshnessAssessment::nominal(),
        }
    }
}

impl PlanFreshnessSupervisor {
    pub fn validate(&self) -> bool {
        self.schema_version == PLAN_FRESHNESS_SCHEMA_VERSION
            && self.current.is_none_or(PlanBasis::validate)
    }

    pub fn install(&mut self, plan: PlanBasis) -> bool {
        if !plan.validate() {
            return false;
        }
        self.current = Some(plan);
        true
    }

    pub fn clear(&mut self) {
        self.current = None;
    }

    pub fn assess(
        &mut self,
        current_step: u64,
        revisions: RuntimeRevisions,
    ) -> PlanFreshnessAssessment {
        let Some(plan) = self.current else {
            self.last = PlanFreshnessAssessment {
                current: false,
                work_authorized: false,
                reason: PlanInvalidationReason::Malformed,
                age_steps: 0,
            };
            return self.last;
        };
        let reason = if !plan.validate() {
            PlanInvalidationReason::Malformed
        } else if current_step < plan.created_step {
            PlanInvalidationReason::FuturePlan
        } else if current_step > plan.expires_step {
            PlanInvalidationReason::Expired
        } else if revisions.hazard != plan.revisions.hazard {
            PlanInvalidationReason::HazardChanged
        } else if revisions.topology != plan.revisions.topology {
            PlanInvalidationReason::TopologyChanged
        } else if revisions.calibration != plan.revisions.calibration {
            PlanInvalidationReason::CalibrationChanged
        } else if revisions.mission != plan.revisions.mission {
            PlanInvalidationReason::MissionChanged
        } else if revisions.state != plan.revisions.state {
            PlanInvalidationReason::StateChanged
        } else {
            PlanInvalidationReason::None
        };
        let current = reason == PlanInvalidationReason::None;
        if !current && self.last.current {
            self.invalidations = self.invalidations.saturating_add(1);
        }
        self.last = PlanFreshnessAssessment {
            current,
            work_authorized: current && plan.permits_productive_work,
            reason,
            age_steps: current_step.saturating_sub(plan.created_step),
        };
        self.last
    }

    pub const fn last(&self) -> PlanFreshnessAssessment {
        self.last
    }

    pub const fn invalidations(&self) -> u64 {
        self.invalidations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn any_safety_relevant_revision_invalidates_work_plan() {
        let revisions = RuntimeRevisions {
            state: 1,
            hazard: 1,
            topology: 1,
            calibration: 1,
            mission: 1,
        };
        let mut supervisor = PlanFreshnessSupervisor::default();
        supervisor.install(PlanBasis {
            plan_id: 7,
            created_step: 1,
            expires_step: 10,
            revisions,
            permits_productive_work: true,
        });
        assert!(supervisor.assess(1, revisions).work_authorized);
        let assessment = supervisor.assess(
            2,
            RuntimeRevisions {
                hazard: 2,
                ..revisions
            },
        );
        assert_eq!(assessment.reason, PlanInvalidationReason::HazardChanged);
        assert!(!assessment.work_authorized);
    }
}
