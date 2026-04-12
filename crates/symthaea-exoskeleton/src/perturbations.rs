// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExoskeletonPerturbation {
    Trip { force_n: f64, at_step: usize },
    HumanFatigue { factor: f64, at_step: usize },
    TerrainStep { height_m: f64, at_step: usize },
    LoadChange { added_mass_kg: f64, at_step: usize },
    ActuatorDegradation { joint: usize, factor: f64, at_step: usize },
    BatteryLow { soc: f64, at_step: usize },
}

#[derive(Debug, Clone, Default)]
pub struct PerturbationSchedule { pub perturbations: Vec<ExoskeletonPerturbation> }

impl PerturbationSchedule {
    pub fn gait_robustness() -> Self {
        Self { perturbations: vec![
            ExoskeletonPerturbation::Trip { force_n: 200.0, at_step: 300 },
            ExoskeletonPerturbation::HumanFatigue { factor: 0.5, at_step: 600 },
            ExoskeletonPerturbation::LoadChange { added_mass_kg: 15.0, at_step: 1200 },
        ]}
    }
}
