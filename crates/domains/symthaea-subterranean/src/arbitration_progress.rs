// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded progress observations for multi-objective arbitration.

use crate::objective_budget::ConflictObjective;
use serde::{Deserialize, Serialize};

pub const ARBITRATION_PROGRESS_SCHEMA_VERSION: u16 = 1;
pub const MAX_PROGRESS_OBJECTIVES: usize = 16;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArbitrationProgressFrame {
    pub schema_version: u16,
    pub step: u64,
    pub battery_ratio: f64,
    pub return_margin: f64,
    pub hazard_severity: f32,
    pub selected: Vec<ConflictObjective>,
    pub completed_work_orders: usize,
    pub restoration_progress: f32,
}

impl ArbitrationProgressFrame {
    pub fn validate(&self) -> bool {
        self.schema_version == ARBITRATION_PROGRESS_SCHEMA_VERSION
            && self.battery_ratio.is_finite()
            && (0.0..=1.0).contains(&self.battery_ratio)
            && self.return_margin.is_finite()
            && (-1.0..=1.0).contains(&self.return_margin)
            && self.hazard_severity.is_finite()
            && (0.0..=1.0).contains(&self.hazard_severity)
            && self.selected.len() <= MAX_PROGRESS_OBJECTIVES
            && self.restoration_progress.is_finite()
            && (0.0..=1.0).contains(&self.restoration_progress)
    }

    pub fn materially_progressed_from(&self, previous: &Self) -> bool {
        self.completed_work_orders > previous.completed_work_orders
            || self.restoration_progress > previous.restoration_progress + 0.01
            || self.hazard_severity + 0.05 < previous.hazard_severity
            || self.return_margin > previous.return_margin + 0.02
            || self.selected != previous.selected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn material_progress_requires_more_than_numeric_noise() {
        let a = ArbitrationProgressFrame {
            schema_version: ARBITRATION_PROGRESS_SCHEMA_VERSION,
            step: 1,
            battery_ratio: 0.8,
            return_margin: 0.3,
            hazard_severity: 0.4,
            selected: vec![ConflictObjective::MissionWork],
            completed_work_orders: 0,
            restoration_progress: 0.2,
        };
        let mut b = a.clone();
        b.step = 2;
        b.return_margin += 0.005;
        assert!(!b.materially_progressed_from(&a));
        b.return_margin += 0.03;
        assert!(b.materially_progressed_from(&a));
    }
}
