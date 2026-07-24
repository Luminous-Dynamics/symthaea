// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded starvation accounting for competing operational objectives.
//!
//! Starvation debt can reduce discretionary authority, but it never outranks
//! physical safety, protected return reserve, environmental containment, or a
//! terminal lifecycle state.

use crate::objective_budget::{ConflictObjective, NUM_CONFLICT_OBJECTIVES};
use serde::{Deserialize, Serialize};

pub const OBJECTIVE_STARVATION_SCHEMA_VERSION: u16 = 1;
pub const DEFAULT_STARVATION_WARNING_STEPS: u32 = 200;
pub const DEFAULT_STARVATION_CRITICAL_STEPS: u32 = 1_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StarvationDisposition {
    Nominal,
    Warning,
    Critical,
    ProtectedObjectiveDeferred,
}

impl StarvationDisposition {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Nominal => "nominal",
            Self::Warning => "warning",
            Self::Critical => "critical",
            Self::ProtectedObjectiveDeferred => "protected_objective_deferred",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObjectiveStarvationRecord {
    pub objective: ConflictObjective,
    pub consecutive_deferred_steps: u32,
    pub total_requested_steps: u64,
    pub total_served_steps: u64,
    pub total_deferred_steps: u64,
}

impl ObjectiveStarvationRecord {
    pub const fn new(objective: ConflictObjective) -> Self {
        Self {
            objective,
            consecutive_deferred_steps: 0,
            total_requested_steps: 0,
            total_served_steps: 0,
            total_deferred_steps: 0,
        }
    }

    pub fn service_ratio(self) -> f32 {
        if self.total_requested_steps == 0 {
            1.0
        } else {
            self.total_served_steps as f32 / self.total_requested_steps as f32
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StarvationAssessment {
    pub disposition: StarvationDisposition,
    pub worst_objective: Option<ConflictObjective>,
    pub maximum_consecutive_deferred_steps: u32,
    pub warning_objectives: Vec<ConflictObjective>,
    pub critical_objectives: Vec<ConflictObjective>,
}

impl StarvationAssessment {
    pub fn nominal() -> Self {
        Self {
            disposition: StarvationDisposition::Nominal,
            worst_objective: None,
            maximum_consecutive_deferred_steps: 0,
            warning_objectives: Vec::new(),
            critical_objectives: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveStarvationMonitor {
    schema_version: u16,
    warning_steps: u32,
    critical_steps: u32,
    records: [ObjectiveStarvationRecord; NUM_CONFLICT_OBJECTIVES],
    last: StarvationAssessment,
}

impl ObjectiveStarvationMonitor {
    pub fn new(warning_steps: u32, critical_steps: u32) -> Self {
        let warning_steps = warning_steps.max(1);
        let critical_steps = critical_steps.max(warning_steps.saturating_add(1));
        Self {
            schema_version: OBJECTIVE_STARVATION_SCHEMA_VERSION,
            warning_steps,
            critical_steps,
            records: ConflictObjective::ALL.map(ObjectiveStarvationRecord::new),
            last: StarvationAssessment::nominal(),
        }
    }

    pub fn validate(&self) -> bool {
        self.schema_version == OBJECTIVE_STARVATION_SCHEMA_VERSION
            && self.warning_steps > 0
            && self.critical_steps > self.warning_steps
            && self
                .records
                .iter()
                .enumerate()
                .all(|(index, record)| record.objective.index() == index)
            && self.last.warning_objectives.len() <= NUM_CONFLICT_OBJECTIVES
            && self.last.critical_objectives.len() <= NUM_CONFLICT_OBJECTIVES
    }

    pub fn observe(
        &mut self,
        active: &[ConflictObjective],
        served: &[ConflictObjective],
    ) -> StarvationAssessment {
        for objective in ConflictObjective::ALL {
            let record = &mut self.records[objective.index()];
            let requested = active.contains(&objective);
            let selected = served.contains(&objective);
            if requested {
                record.total_requested_steps = record.total_requested_steps.saturating_add(1);
                if selected {
                    record.total_served_steps = record.total_served_steps.saturating_add(1);
                    record.consecutive_deferred_steps = 0;
                } else {
                    record.total_deferred_steps = record.total_deferred_steps.saturating_add(1);
                    record.consecutive_deferred_steps =
                        record.consecutive_deferred_steps.saturating_add(1);
                }
            } else {
                record.consecutive_deferred_steps = 0;
            }
        }

        let mut warning_objectives = Vec::new();
        let mut critical_objectives = Vec::new();
        let mut worst_objective = None;
        let mut maximum = 0u32;
        let mut protected_deferred = false;
        for record in self.records.iter().copied() {
            if record.consecutive_deferred_steps > maximum {
                maximum = record.consecutive_deferred_steps;
                worst_objective = Some(record.objective);
            }
            if record.consecutive_deferred_steps >= self.warning_steps {
                warning_objectives.push(record.objective);
            }
            if record.consecutive_deferred_steps >= self.critical_steps {
                critical_objectives.push(record.objective);
            }
            if record.objective.class() == crate::objective_budget::ObjectiveClass::Protected
                && record.consecutive_deferred_steps > 0
            {
                protected_deferred = true;
            }
        }
        let disposition = if protected_deferred {
            StarvationDisposition::ProtectedObjectiveDeferred
        } else if !critical_objectives.is_empty() {
            StarvationDisposition::Critical
        } else if !warning_objectives.is_empty() {
            StarvationDisposition::Warning
        } else {
            StarvationDisposition::Nominal
        };
        self.last = StarvationAssessment {
            disposition,
            worst_objective,
            maximum_consecutive_deferred_steps: maximum,
            warning_objectives,
            critical_objectives,
        };
        self.last.clone()
    }

    pub fn debt_score(&self, objective: ConflictObjective) -> f32 {
        let record = self.records[objective.index()];
        (record.consecutive_deferred_steps as f32 / self.critical_steps as f32).clamp(0.0, 1.0)
    }

    pub fn record(&self, objective: ConflictObjective) -> ObjectiveStarvationRecord {
        self.records[objective.index()]
    }

    pub fn last(&self) -> &StarvationAssessment {
        &self.last
    }
}

impl Default for ObjectiveStarvationMonitor {
    fn default() -> Self {
        Self::new(
            DEFAULT_STARVATION_WARNING_STEPS,
            DEFAULT_STARVATION_CRITICAL_STEPS,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protected_objective_deferral_is_immediately_visible() {
        let mut monitor = ObjectiveStarvationMonitor::new(2, 4);
        let assessment = monitor.observe(&[ConflictObjective::ReturnReserve], &[]);
        assert_eq!(
            assessment.disposition,
            StarvationDisposition::ProtectedObjectiveDeferred
        );
    }

    #[test]
    fn soft_objective_debt_resets_after_service() {
        let mut monitor = ObjectiveStarvationMonitor::new(2, 4);
        for _ in 0..3 {
            monitor.observe(&[ConflictObjective::MissionWork], &[]);
        }
        assert_eq!(monitor.last().disposition, StarvationDisposition::Warning);
        monitor.observe(
            &[ConflictObjective::MissionWork],
            &[ConflictObjective::MissionWork],
        );
        assert_eq!(monitor.debt_score(ConflictObjective::MissionWork), 0.0);
    }
}
