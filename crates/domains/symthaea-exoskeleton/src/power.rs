// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Protected multi-bus exosuit power model.
//!
//! Survival, mobility, and mission loads are intentionally isolated. Mobility
//! and mission loads cannot draw the protected survival reserve through this
//! model. Numbers are simulation inputs, not flight battery specifications.

use serde::{Deserialize, Serialize};

use crate::space_exosuit::ExosuitEvidenceLevel;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PowerBusKind {
    Survival,
    Mobility,
    Mission,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EnergyBus {
    pub capacity_wh: f64,
    pub energy_wh: f64,
    pub reserve_floor_wh: f64,
    pub max_discharge_w: f64,
    pub max_charge_w: f64,
    pub temperature_k: f64,
}

impl EnergyBus {
    pub fn is_valid(&self) -> bool {
        self.capacity_wh.is_finite()
            && self.capacity_wh > 0.0
            && self.energy_wh.is_finite()
            && (0.0..=self.capacity_wh).contains(&self.energy_wh)
            && self.reserve_floor_wh.is_finite()
            && (0.0..=self.capacity_wh).contains(&self.reserve_floor_wh)
            && self.max_discharge_w.is_finite()
            && self.max_discharge_w >= 0.0
            && self.max_charge_w.is_finite()
            && self.max_charge_w >= 0.0
            && self.temperature_k.is_finite()
            && self.temperature_k > 0.0
    }

    pub fn state_of_charge(&self) -> f64 {
        self.energy_wh / self.capacity_wh
    }

    fn deliver(&mut self, requested_w: f64, dt_s: f64, protect_reserve: bool) -> f64 {
        if requested_w <= 0.0 || dt_s <= 0.0 {
            return 0.0;
        }
        let reserve = if protect_reserve { self.reserve_floor_wh } else { 0.0 };
        let available_wh = (self.energy_wh - reserve).max(0.0);
        let available_w = available_wh * 3600.0 / dt_s;
        let delivered_w = requested_w.min(self.max_discharge_w).min(available_w);
        self.energy_wh -= delivered_w * dt_s / 3600.0;
        delivered_w
    }

    fn charge(&mut self, offered_w: f64, dt_s: f64) -> f64 {
        if offered_w <= 0.0 || dt_s <= 0.0 {
            return 0.0;
        }
        let headroom_wh = (self.capacity_wh - self.energy_wh).max(0.0);
        let headroom_w = headroom_wh * 3600.0 / dt_s;
        let accepted_w = offered_w.min(self.max_charge_w).min(headroom_w);
        self.energy_wh += accepted_w * dt_s / 3600.0;
        accepted_w
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MultiBusPowerConfig {
    pub survival: EnergyBus,
    pub mobility: EnergyBus,
    pub mission: EnergyBus,
    /// Maximum regenerative power accepted from exoskeleton joints.
    pub max_regen_w: f64,
    /// Maximum charger/rover input available to the complete suit.
    pub max_external_charge_w: f64,
    pub evidence: ExosuitEvidenceLevel,
}

impl MultiBusPowerConfig {
    pub fn simulation_reference() -> Self {
        Self {
            survival: EnergyBus {
                capacity_wh: 1_000.0,
                energy_wh: 1_000.0,
                reserve_floor_wh: 250.0,
                max_discharge_w: 500.0,
                max_charge_w: 400.0,
                temperature_k: 298.0,
            },
            mobility: EnergyBus {
                capacity_wh: 700.0,
                energy_wh: 700.0,
                reserve_floor_wh: 35.0,
                max_discharge_w: 1_200.0,
                max_charge_w: 500.0,
                temperature_k: 298.0,
            },
            mission: EnergyBus {
                capacity_wh: 300.0,
                energy_wh: 300.0,
                reserve_floor_wh: 30.0,
                max_discharge_w: 300.0,
                max_charge_w: 200.0,
                temperature_k: 298.0,
            },
            max_regen_w: 300.0,
            max_external_charge_w: 1_000.0,
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.survival.is_valid()
            && self.mobility.is_valid()
            && self.mission.is_valid()
            && self.max_regen_w.is_finite()
            && self.max_regen_w >= 0.0
            && self.max_external_charge_w.is_finite()
            && self.max_external_charge_w >= 0.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PowerRequest {
    pub survival_w: f64,
    pub mobility_w: f64,
    pub mission_w: f64,
    pub regenerative_w: f64,
    pub external_charger_w: f64,
    pub dt_s: f64,
}

impl PowerRequest {
    pub fn is_valid(&self) -> bool {
        [
            self.survival_w,
            self.mobility_w,
            self.mission_w,
            self.regenerative_w,
            self.external_charger_w,
            self.dt_s,
        ]
        .into_iter()
        .all(f64::is_finite)
            && self.survival_w >= 0.0
            && self.mobility_w >= 0.0
            && self.mission_w >= 0.0
            && self.regenerative_w >= 0.0
            && self.external_charger_w >= 0.0
            && self.dt_s > 0.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PowerAllocation {
    pub survival_delivered_w: f64,
    pub mobility_delivered_w: f64,
    pub mission_delivered_w: f64,
    pub regen_accepted_w: f64,
    pub external_charge_accepted_w: f64,
    pub survival_satisfied: bool,
    pub mobility_satisfied: bool,
    pub mission_satisfied: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PowerError {
    InvalidConfig,
    InvalidRequest,
}

#[derive(Debug, Clone)]
pub struct MultiBusPowerSystem {
    config: MultiBusPowerConfig,
}

impl MultiBusPowerSystem {
    pub fn new(config: MultiBusPowerConfig) -> Result<Self, PowerError> {
        if !config.is_valid() {
            return Err(PowerError::InvalidConfig);
        }
        Ok(Self { config })
    }

    pub fn simulation_reference() -> Self {
        Self::new(MultiBusPowerConfig::simulation_reference())
            .expect("reference power config must be valid")
    }

    pub fn config(&self) -> &MultiBusPowerConfig {
        &self.config
    }

    pub fn config_mut_for_fault_injection(&mut self) -> &mut MultiBusPowerConfig {
        &mut self.config
    }

    pub fn step(&mut self, request: PowerRequest) -> Result<PowerAllocation, PowerError> {
        if !self.config.is_valid() {
            return Err(PowerError::InvalidConfig);
        }
        if !request.is_valid() {
            return Err(PowerError::InvalidRequest);
        }

        // Survival gets first use of the survival bus; other classes never
        // draw it. The reserve is protected from routine operation.
        let survival = self
            .config
            .survival
            .deliver(request.survival_w, request.dt_s, true);
        let mobility = self
            .config
            .mobility
            .deliver(request.mobility_w, request.dt_s, true);
        let mission = self
            .config
            .mission
            .deliver(request.mission_w, request.dt_s, true);

        let regen_offer = request.regenerative_w.min(self.config.max_regen_w);
        let regen = self.config.mobility.charge(regen_offer, request.dt_s);

        // External charging is priority ordered: survival -> mobility -> mission.
        let mut external = request
            .external_charger_w
            .min(self.config.max_external_charge_w);
        let survival_charge = self.config.survival.charge(external, request.dt_s);
        external -= survival_charge;
        let mobility_charge = self.config.mobility.charge(external, request.dt_s);
        external -= mobility_charge;
        let mission_charge = self.config.mission.charge(external, request.dt_s);

        Ok(PowerAllocation {
            survival_delivered_w: survival,
            mobility_delivered_w: mobility,
            mission_delivered_w: mission,
            regen_accepted_w: regen,
            external_charge_accepted_w: survival_charge + mobility_charge + mission_charge,
            survival_satisfied: survival + 1e-9 >= request.survival_w,
            mobility_satisfied: mobility + 1e-9 >= request.mobility_w,
            mission_satisfied: mission + 1e-9 >= request.mission_w,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mobility_cannot_consume_survival_reserve() {
        let mut power = MultiBusPowerSystem::simulation_reference();
        let survival_before = power.config().survival.energy_wh;
        let _ = power
            .step(PowerRequest {
                survival_w: 0.0,
                mobility_w: 1_000.0,
                mission_w: 0.0,
                regenerative_w: 0.0,
                external_charger_w: 0.0,
                dt_s: 60.0,
            })
            .unwrap();
        assert_eq!(power.config().survival.energy_wh, survival_before);
    }

    #[test]
    fn external_charger_prioritizes_survival_bus() {
        let mut power = MultiBusPowerSystem::simulation_reference();
        power.config_mut_for_fault_injection().survival.energy_wh = 500.0;
        power.config_mut_for_fault_injection().mobility.energy_wh = 500.0;
        let before_survival = power.config().survival.energy_wh;
        let before_mobility = power.config().mobility.energy_wh;
        let allocation = power
            .step(PowerRequest {
                survival_w: 0.0,
                mobility_w: 0.0,
                mission_w: 0.0,
                regenerative_w: 0.0,
                external_charger_w: 300.0,
                dt_s: 60.0,
            })
            .unwrap();
        assert!(allocation.external_charge_accepted_w > 0.0);
        assert!(power.config().survival.energy_wh > before_survival);
        assert_eq!(power.config().mobility.energy_wh, before_mobility);
    }

    #[test]
    fn regenerative_energy_returns_only_to_mobility_bus() {
        let mut power = MultiBusPowerSystem::simulation_reference();
        power.config_mut_for_fault_injection().mobility.energy_wh = 500.0;
        let survival_before = power.config().survival.energy_wh;
        let mobility_before = power.config().mobility.energy_wh;
        let allocation = power
            .step(PowerRequest {
                survival_w: 0.0,
                mobility_w: 0.0,
                mission_w: 0.0,
                regenerative_w: 200.0,
                external_charger_w: 0.0,
                dt_s: 60.0,
            })
            .unwrap();
        assert!(allocation.regen_accepted_w > 0.0);
        assert!(power.config().mobility.energy_wh > mobility_before);
        assert_eq!(power.config().survival.energy_wh, survival_before);
    }
}
