// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Portable Life Support System reference twin for EVA trade studies.
//!
//! This is a simulation-only digital twin. It never commands primary oxygen,
//! pressure, ventilation, or thermal hardware. The architecture mirrors the
//! broad NASA PLSS decomposition: oxygen, ventilation/CO2+humidity removal,
//! and thermal control, with independent primary/secondary survival paths.

use serde::{Deserialize, Serialize};

use crate::metabolism::MetabolicEstimate;
use crate::space_exosuit::ExosuitEvidenceLevel;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlssPathState {
    Available,
    Degraded,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PlssConfig {
    /// Usable primary oxygen inventory, standard litres.
    pub primary_o2_l: f64,
    /// Independent contingency oxygen inventory, standard litres.
    pub secondary_o2_l: f64,
    /// Ventilation-loop flow reference, L/min.
    pub ventilation_flow_l_min: f64,
    /// Fraction of incoming CO2 removed per reference pass [0,1].
    pub co2_removal_efficiency: f64,
    /// Fraction of humidity load removed per reference pass [0,1].
    pub humidity_removal_efficiency: f64,
    /// Primary thermal-loop heat rejection capacity, W.
    pub primary_heat_rejection_w: f64,
    /// Secondary/emergency thermal-loop heat rejection capacity, W.
    pub secondary_heat_rejection_w: f64,
    /// Effective thermal capacitance of wearer+suit model, J/K.
    pub thermal_capacitance_j_k: f64,
    pub evidence: ExosuitEvidenceLevel,
}

impl PlssConfig {
    pub fn simulation_reference() -> Self {
        Self {
            primary_o2_l: 2_000.0,
            secondary_o2_l: 400.0,
            ventilation_flow_l_min: 170.0,
            co2_removal_efficiency: 0.98,
            humidity_removal_efficiency: 0.95,
            primary_heat_rejection_w: 900.0,
            secondary_heat_rejection_w: 250.0,
            thermal_capacitance_j_k: 300_000.0,
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.primary_o2_l.is_finite()
            && self.primary_o2_l >= 0.0
            && self.secondary_o2_l.is_finite()
            && self.secondary_o2_l >= 0.0
            && self.ventilation_flow_l_min.is_finite()
            && self.ventilation_flow_l_min > 0.0
            && self.co2_removal_efficiency.is_finite()
            && (0.0..=1.0).contains(&self.co2_removal_efficiency)
            && self.humidity_removal_efficiency.is_finite()
            && (0.0..=1.0).contains(&self.humidity_removal_efficiency)
            && self.primary_heat_rejection_w.is_finite()
            && self.primary_heat_rejection_w >= 0.0
            && self.secondary_heat_rejection_w.is_finite()
            && self.secondary_heat_rejection_w >= 0.0
            && self.thermal_capacitance_j_k.is_finite()
            && self.thermal_capacitance_j_k > 0.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PlssState {
    pub primary_o2_remaining_l: f64,
    pub secondary_o2_remaining_l: f64,
    /// Simplified suit-loop CO2 burden, litres-equivalent.
    pub co2_burden_l: f64,
    /// Simplified normalized humidity burden [0,+inf).
    pub humidity_burden: f64,
    pub thermal_store_k: f64,
    pub primary_oxygen: PlssPathState,
    pub secondary_oxygen: PlssPathState,
    pub ventilation: PlssPathState,
    pub primary_thermal: PlssPathState,
    pub secondary_thermal: PlssPathState,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PlssStepInput {
    pub metabolism: MetabolicEstimate,
    /// Additional electrical/actuator waste heat entering the suit loop, W.
    pub equipment_heat_w: f64,
    /// Relative humidity production proxy, normalized units/min.
    pub humidity_generation_per_min: f64,
    pub dt_s: f64,
}

impl PlssStepInput {
    pub fn is_valid(&self) -> bool {
        self.metabolism.metabolic_power_w.is_finite()
            && self.metabolism.oxygen_l_min.is_finite()
            && self.metabolism.co2_l_min.is_finite()
            && self.metabolism.metabolic_heat_w.is_finite()
            && self.equipment_heat_w.is_finite()
            && self.equipment_heat_w >= 0.0
            && self.humidity_generation_per_min.is_finite()
            && self.humidity_generation_per_min >= 0.0
            && self.dt_s.is_finite()
            && self.dt_s > 0.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlssStepError {
    InvalidConfig,
    InvalidInput,
    NoOxygenPath,
    VentilationFailed,
    NoThermalPath,
}

#[derive(Debug, Clone)]
pub struct PlssReferenceTwin {
    config: PlssConfig,
    state: PlssState,
}

impl PlssReferenceTwin {
    pub fn new(config: PlssConfig) -> Result<Self, PlssStepError> {
        if !config.is_valid() {
            return Err(PlssStepError::InvalidConfig);
        }
        Ok(Self {
            state: PlssState {
                primary_o2_remaining_l: config.primary_o2_l,
                secondary_o2_remaining_l: config.secondary_o2_l,
                co2_burden_l: 0.0,
                humidity_burden: 0.0,
                thermal_store_k: 0.0,
                primary_oxygen: PlssPathState::Available,
                secondary_oxygen: PlssPathState::Available,
                ventilation: PlssPathState::Available,
                primary_thermal: PlssPathState::Available,
                secondary_thermal: PlssPathState::Available,
            },
            config,
        })
    }

    pub fn simulation_reference() -> Self {
        Self::new(PlssConfig::simulation_reference()).expect("reference PLSS config must be valid")
    }

    pub fn state(&self) -> &PlssState {
        &self.state
    }

    pub fn state_mut_for_fault_injection(&mut self) -> &mut PlssState {
        &mut self.state
    }

    pub fn config(&self) -> &PlssConfig {
        &self.config
    }

    pub fn step(&mut self, input: PlssStepInput) -> Result<(), PlssStepError> {
        if !self.config.is_valid() {
            return Err(PlssStepError::InvalidConfig);
        }
        if !input.is_valid() {
            return Err(PlssStepError::InvalidInput);
        }
        if self.state.ventilation == PlssPathState::Failed {
            return Err(PlssStepError::VentilationFailed);
        }

        let minutes = input.dt_s / 60.0;
        let oxygen_needed_l = input.metabolism.oxygen_l_min * minutes;
        self.consume_oxygen(oxygen_needed_l)?;

        let ventilation_factor = match self.state.ventilation {
            PlssPathState::Available => 1.0,
            PlssPathState::Degraded => 0.6,
            PlssPathState::Failed => 0.0,
        };
        let co2_added_l = input.metabolism.co2_l_min * minutes;
        let removable_co2_l = (self.config.ventilation_flow_l_min * minutes)
            * self.config.co2_removal_efficiency
            * ventilation_factor;
        self.state.co2_burden_l =
            (self.state.co2_burden_l + co2_added_l - removable_co2_l).max(0.0);

        let humidity_added = input.humidity_generation_per_min * minutes;
        let humidity_removed = self.config.humidity_removal_efficiency
            * ventilation_factor
            * minutes;
        self.state.humidity_burden =
            (self.state.humidity_burden + humidity_added - humidity_removed).max(0.0);

        let heat_in_w = input.metabolism.metabolic_heat_w + input.equipment_heat_w;
        let heat_rejection_w = self.available_heat_rejection_w()?;
        let net_heat_j = (heat_in_w - heat_rejection_w) * input.dt_s;
        self.state.thermal_store_k += net_heat_j / self.config.thermal_capacitance_j_k;

        Ok(())
    }

    fn consume_oxygen(&mut self, mut needed_l: f64) -> Result<(), PlssStepError> {
        if self.state.primary_oxygen != PlssPathState::Failed {
            let take = needed_l.min(self.state.primary_o2_remaining_l);
            self.state.primary_o2_remaining_l -= take;
            needed_l -= take;
        }
        if needed_l > 0.0 && self.state.secondary_oxygen != PlssPathState::Failed {
            let take = needed_l.min(self.state.secondary_o2_remaining_l);
            self.state.secondary_o2_remaining_l -= take;
            needed_l -= take;
        }
        if needed_l > 1e-9 {
            Err(PlssStepError::NoOxygenPath)
        } else {
            Ok(())
        }
    }

    fn available_heat_rejection_w(&self) -> Result<f64, PlssStepError> {
        let primary = match self.state.primary_thermal {
            PlssPathState::Available => self.config.primary_heat_rejection_w,
            PlssPathState::Degraded => 0.5 * self.config.primary_heat_rejection_w,
            PlssPathState::Failed => 0.0,
        };
        let secondary = match self.state.secondary_thermal {
            PlssPathState::Available => self.config.secondary_heat_rejection_w,
            PlssPathState::Degraded => 0.5 * self.config.secondary_heat_rejection_w,
            PlssPathState::Failed => 0.0,
        };
        let total = primary + secondary;
        if total <= 0.0 {
            Err(PlssStepError::NoThermalPath)
        } else {
            Ok(total)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metabolism::{HumanMetabolicModel, HumanWorkload};

    fn metabolism() -> MetabolicEstimate {
        HumanMetabolicModel::reference()
            .estimate(HumanWorkload {
                positive_mechanical_power_w: 100.0,
                negative_mechanical_power_w: 0.0,
            })
            .unwrap()
    }

    #[test]
    fn nominal_step_consumes_oxygen() {
        let mut twin = PlssReferenceTwin::simulation_reference();
        let before = twin.state().primary_o2_remaining_l;
        twin.step(PlssStepInput {
            metabolism: metabolism(),
            equipment_heat_w: 50.0,
            humidity_generation_per_min: 0.2,
            dt_s: 60.0,
        })
        .unwrap();
        assert!(twin.state().primary_o2_remaining_l < before);
    }

    #[test]
    fn secondary_oxygen_carries_primary_failure() {
        let mut twin = PlssReferenceTwin::simulation_reference();
        twin.state_mut_for_fault_injection().primary_oxygen = PlssPathState::Failed;
        let before = twin.state().secondary_o2_remaining_l;
        twin.step(PlssStepInput {
            metabolism: metabolism(),
            equipment_heat_w: 0.0,
            humidity_generation_per_min: 0.0,
            dt_s: 30.0,
        })
        .unwrap();
        assert!(twin.state().secondary_o2_remaining_l < before);
    }

    #[test]
    fn losing_all_thermal_paths_fails_closed() {
        let mut twin = PlssReferenceTwin::simulation_reference();
        twin.state_mut_for_fault_injection().primary_thermal = PlssPathState::Failed;
        twin.state_mut_for_fault_injection().secondary_thermal = PlssPathState::Failed;
        let result = twin.step(PlssStepInput {
            metabolism: metabolism(),
            equipment_heat_w: 0.0,
            humidity_generation_per_min: 0.0,
            dt_s: 1.0,
        });
        assert_eq!(result, Err(PlssStepError::NoThermalPath));
    }
}
