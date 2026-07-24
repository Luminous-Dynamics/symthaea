// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic shedding of discretionary objectives during liveness failure.

use crate::objective_budget::{ConflictObjective, ObjectiveClass};
use serde::{Deserialize, Serialize};

pub const MISSION_SHEDDING_SCHEMA_VERSION: u16 = 1;
pub const MAX_SHED_OBJECTIVES: usize = 16;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MissionSheddingPlan {
    pub schema_version: u16,
    pub retained: Vec<ConflictObjective>,
    pub shed: Vec<ConflictObjective>,
    pub protected_preserved: bool,
}

impl MissionSheddingPlan {
    pub fn derive(active: &[ConflictObjective], maximum_discretionary: usize) -> Self {
        let mut protected = Vec::new();
        let mut discretionary = Vec::new();
        for objective in active.iter().copied() {
            if objective.class() == ObjectiveClass::Protected {
                protected.push(objective);
            } else {
                discretionary.push(objective);
            }
        }
        protected.sort_by_key(|objective| objective.index());
        discretionary.sort_by_key(|objective| objective.index());
        let keep = maximum_discretionary.min(discretionary.len());
        let mut retained = protected.clone();
        retained.extend(discretionary.iter().copied().take(keep));
        let shed = discretionary.into_iter().skip(keep).collect();
        Self {
            schema_version: MISSION_SHEDDING_SCHEMA_VERSION,
            retained,
            shed,
            protected_preserved: true,
        }
    }

    pub fn validate(&self) -> bool {
        self.schema_version == MISSION_SHEDDING_SCHEMA_VERSION
            && self.retained.len() <= MAX_SHED_OBJECTIVES
            && self.shed.len() <= MAX_SHED_OBJECTIVES
            && self.protected_preserved
            && self
                .shed
                .iter()
                .all(|objective| objective.class() != ObjectiveClass::Protected)
            && self
                .retained
                .iter()
                .all(|objective| !self.shed.contains(objective))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protected_objectives_are_never_shed() {
        let plan = MissionSheddingPlan::derive(
            &[
                ConflictObjective::PhysicalSafety,
                ConflictObjective::ReturnReserve,
                ConflictObjective::MissionWork,
                ConflictObjective::Communications,
            ],
            1,
        );
        assert!(plan.retained.contains(&ConflictObjective::PhysicalSafety));
        assert!(plan.retained.contains(&ConflictObjective::ReturnReserve));
        assert!(!plan.shed.contains(&ConflictObjective::PhysicalSafety));
        assert!(plan.validate());
    }
}
