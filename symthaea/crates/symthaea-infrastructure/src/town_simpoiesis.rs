// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Town Sympoiesis Orchestrator — Unified metabolism for circular settlements.

use crate::types::{InfrastructureCommand, InfrastructureState};
use serde::{Deserialize, Serialize};
use symthaea_engineering::{EngineeringManager, InfrastructureNode};
use symthaea_silicon::PowerDistributionLogic;

/// The primary metabolism of a circular town.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TownSympoiesis {
    pub name: String,
    pub power_grid: PowerDistributionLogic,
    pub infrastructure: InfrastructureNode,
    pub water_clarity: f32,
    pub nutrient_advection: f32,
}

impl TownSympoiesis {
    /// Create a new town sympoiesis loop.
    pub fn new(name: &str, manager: &mut EngineeringManager) -> Self {
        let infra = manager.design_infrastructure(name, 10.0);
        let grid = PowerDistributionLogic {
            grid_frequency_hz: 60.0,
            renewable_ratio: 0.8,
            active_loads_mw: 10.0,
            battery_reserve_mwh: 100.0,
            min_critical_mw: 2.0, // Core life support threshold
        };

        Self {
            name: name.into(),
            power_grid: grid,
            infrastructure: infra,
            water_clarity: 0.95,
            nutrient_advection: 0.5,
        }
    }

    /// Execute a metabolic step: synchronize silicon brain with physical infrastructure.
    pub fn step(&mut self, demand_mw: f32, available_mw: f32) -> f32 {
        // 1. Silicon Brain optimizes the grid
        let surprise = self.power_grid.optimize_routing(demand_mw, available_mw);

        // 2. Physical Metabolism: Adjust fluid states based on power stability
        if surprise < 0.2 {

            self.water_clarity = (self.water_clarity + 0.01).min(1.0);
            self.nutrient_advection = (self.nutrient_advection + 0.02).min(1.0);
        } else {
            self.water_clarity = (self.water_clarity - 0.05).max(0.0);
        }

        surprise
    }
}
