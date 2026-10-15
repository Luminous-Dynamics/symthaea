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
    pub primary_o2_l: f64,
    pub secondary_o2_l: f64,
    pub ventilation_flow_l_min: f64,
    /// Maximum CO2 capture rate in L CO2-equivalent/min at declared reference conditions.
    /// This is separate from ventilation throughput and remains simulation-only.
    pub co2_scrubber_removal_rate_l_min: f64,
    /// Finite total cartridge capacity in L CO2-equivalent at the same reference conditions.
    pub co2_scrubber_capacity_l: f64,
    /// Maximum water-vapor condensation/removal rate, g water/min.
    pub humidity_removal_rate_g_min: f64,
    /// Finite liquid-condensate storage capacity, g water.
    pub humidity_collector_capacity_g: f64,
    pub primary_heat_rejection_w: f64,
    pub secondary_heat_rejection_w: f64,
    pub thermal_capacitance_j_k: f64,
    pub evidence: ExosuitEvidenceLevel,
}

impl PlssConfig {
    pub fn simulation_reference() -> Self {
        Self {
            primary_o2_l: 2_000.0,
            secondary_o2_l: 400.0,
            ventilation_flow_l_min: 170.0,
            co2_scrubber_removal_rate_l_min: 1.6,
            co2_scrubber_capacity_l: 120.0,
            humidity_removal_rate_g_min: 0.5,
            humidity_collector_capacity_g: 100.0,
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
            && self.co2_scrubber_removal_rate_l_min.is_finite()
            && self.co2_scrubber_removal_rate_l_min >= 0.0
            && self.co2_scrubber_capacity_l.is_finite()
            && self.co2_scrubber_capacity_l >= 0.0
            && self.humidity_removal_rate_g_min.is_finite()
            && self.humidity_removal_rate_g_min >= 0.0
            && self.humidity_collector_capacity_g.is_finite()
            && self.humidity_collector_capacity_g >= 0.0
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
    /// Unremoved CO2 generation as gas-equivalent liters at declared reference
    /// conditions; this is not a partial-pressure or exposure estimate.
    pub co2_burden_l: f64,
    pub co2_scrubber_capacity_remaining_l: f64,
    /// Water vapor awaiting removal, represented as water mass in grams.
    pub humidity_vapor_burden_g: f64,
    /// Remaining physical storage capacity for collected condensate, grams.
    pub humidity_collector_capacity_remaining_g: f64,
    /// Carbon-dioxide sorbent path; failure means zero modeled capture.
    pub co2_scrubber: PlssPathState,
    /// Humidity/condensation path; failure means zero modeled collection.
    pub humidity_control: PlssPathState,
    /// Relative stored-energy/thermal-debt proxy in kelvin-equivalent.
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
    pub equipment_heat_w: f64,
    /// Water-vapor generation, grams of water per minute.
    pub humidity_generation_g_per_min: f64,
    pub dt_s: f64,
}

impl PlssStepInput {
    pub fn is_valid(&self) -> bool {
        self.metabolism.metabolic_power_w.is_finite()
            && self.metabolism.metabolic_power_w >= 0.0
            && self.metabolism.oxygen_l_min.is_finite()
            && self.metabolism.oxygen_l_min >= 0.0
            && self.metabolism.co2_l_min.is_finite()
            && self.metabolism.co2_l_min >= 0.0
            && self.metabolism.metabolic_heat_w.is_finite()
            && self.metabolism.metabolic_heat_w >= 0.0
            && self.equipment_heat_w.is_finite()
            && self.equipment_heat_w >= 0.0
            && self.humidity_generation_g_per_min.is_finite()
            && self.humidity_generation_g_per_min >= 0.0
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
    /// Environment-established fraction of nominal heat rejection [0,1].
    /// This is simulation state only and never commands a thermal controller.
    environment_heat_rejection_fraction: f64,
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
                co2_scrubber_capacity_remaining_l: config.co2_scrubber_capacity_l,
                humidity_vapor_burden_g: 0.0,
                humidity_collector_capacity_remaining_g: config.humidity_collector_capacity_g,
                co2_scrubber: PlssPathState::Available,
                humidity_control: PlssPathState::Available,
                thermal_store_k: 0.0,
                primary_oxygen: PlssPathState::Available,
                secondary_oxygen: PlssPathState::Available,
                ventilation: PlssPathState::Available,
                primary_thermal: PlssPathState::Available,
                secondary_thermal: PlssPathState::Available,
            },
            config,
            environment_heat_rejection_fraction: 1.0,
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

    pub fn environment_heat_rejection_fraction(&self) -> f64 {
        self.environment_heat_rejection_fraction
    }

    pub fn set_environment_heat_rejection_fraction(
        &mut self,
        fraction: f64,
    ) -> Result<(), PlssStepError> {
        if !fraction.is_finite() || !(0.0..=1.0).contains(&fraction) {
            return Err(PlssStepError::InvalidInput);
        }
        self.environment_heat_rejection_fraction = fraction;
        Ok(())
    }

    pub fn step(&mut self, input: PlssStepInput) -> Result<(), PlssStepError> {
        self.step_with_heat_rejection_fraction(input, self.environment_heat_rejection_fraction)
    }

    pub fn step_with_heat_rejection_fraction(
        &mut self,
        input: PlssStepInput,
        heat_rejection_fraction: f64,
    ) -> Result<(), PlssStepError> {
        if !self.config.is_valid() {
            return Err(PlssStepError::InvalidConfig);
        }
        if !input.is_valid()
            || !heat_rejection_fraction.is_finite()
            || !(0.0..=1.0).contains(&heat_rejection_fraction)
        {
            return Err(PlssStepError::InvalidInput);
        }
        if self.state.ventilation == PlssPathState::Failed {
            return Err(PlssStepError::VentilationFailed);
        }

        let minutes = input.dt_s / 60.0;
        let oxygen_needed_l = input.metabolism.oxygen_l_min * minutes;
        let co2_added_l = input.metabolism.co2_l_min * minutes;
        let humidity_added_g = input.humidity_generation_g_per_min * minutes;

        // Check arithmetic and mutable simulation state before consuming
        // resources. Non-finite intermediate values must not become zero burden.
        let co2_available_l = self.state.co2_burden_l + co2_added_l;
        let humidity_available_g = self.state.humidity_vapor_burden_g + humidity_added_g;
        if !minutes.is_finite()
            || !oxygen_needed_l.is_finite()
            || !co2_added_l.is_finite()
            || !humidity_added_g.is_finite()
            || !co2_available_l.is_finite()
            || !humidity_available_g.is_finite()
            || !self.state.co2_burden_l.is_finite()
            || self.state.co2_burden_l < 0.0
            || !self.state.co2_scrubber_capacity_remaining_l.is_finite()
            || !(0.0..=self.config.co2_scrubber_capacity_l)
                .contains(&self.state.co2_scrubber_capacity_remaining_l)
            || !self.state.humidity_vapor_burden_g.is_finite()
            || self.state.humidity_vapor_burden_g < 0.0
            || !self.state.humidity_collector_capacity_remaining_g.is_finite()
            || !(0.0..=self.config.humidity_collector_capacity_g)
                .contains(&self.state.humidity_collector_capacity_remaining_g)
        {
            return Err(PlssStepError::InvalidInput);
        }

        self.consume_oxygen(oxygen_needed_l)?;

        let ventilation_factor = match self.state.ventilation {
            PlssPathState::Available => 1.0,
            PlssPathState::Degraded => 0.6,
            PlssPathState::Failed => 0.0,
        };
        let co2_scrubber_factor = match self.state.co2_scrubber {
            PlssPathState::Available => 1.0,
            PlssPathState::Degraded => 0.5,
            PlssPathState::Failed => 0.0,
        };
        let humidity_control_factor = match self.state.humidity_control {
            PlssPathState::Available => 1.0,
            PlssPathState::Degraded => 0.5,
            PlssPathState::Failed => 0.0,
        };

        // Fan flow does not equal CO2 captured. Capture is separately bounded
        // by a declared maximum rate, finite cartridge capacity, and the CO2
        // gas-equivalent burden actually present.
        let max_co2_by_rate_l = self.config.co2_scrubber_removal_rate_l_min
            * minutes
            * ventilation_factor
            * co2_scrubber_factor;
        let co2_removed_l = co2_available_l
            .min(max_co2_by_rate_l)
            .min(self.state.co2_scrubber_capacity_remaining_l);
        let co2_burden_l = co2_available_l - co2_removed_l;
        let co2_capacity_remaining_l = self.state.co2_scrubber_capacity_remaining_l - co2_removed_l;

        // Humidity is tracked as water mass. Collected condensate consumes finite
        // storage capacity instead of disappearing via an unbounded efficiency.
        let max_humidity_by_rate_g = self.config.humidity_removal_rate_g_min
            * minutes
            * ventilation_factor
            * humidity_control_factor;
        let humidity_removed_g = humidity_available_g
            .min(max_humidity_by_rate_g)
            .min(self.state.humidity_collector_capacity_remaining_g);
        let humidity_vapor_burden_g = humidity_available_g - humidity_removed_g;
        let humidity_capacity_remaining_g =
            self.state.humidity_collector_capacity_remaining_g - humidity_removed_g;

        if !max_co2_by_rate_l.is_finite()
            || !co2_removed_l.is_finite()
            || !co2_burden_l.is_finite()
            || !co2_capacity_remaining_l.is_finite()
            || !max_humidity_by_rate_g.is_finite()
            || !humidity_removed_g.is_finite()
            || !humidity_vapor_burden_g.is_finite()
            || !humidity_capacity_remaining_g.is_finite()
        {
            return Err(PlssStepError::InvalidInput);
        }

        let heat_in_w = input.metabolism.metabolic_heat_w + input.equipment_heat_w;
        let heat_rejection_w = self.available_heat_rejection_w()? * heat_rejection_fraction;
        let net_heat_j = (heat_in_w - heat_rejection_w) * input.dt_s;
        let thermal_store_k =
            self.state.thermal_store_k + net_heat_j / self.config.thermal_capacitance_j_k;
        if !heat_in_w.is_finite()
            || !heat_rejection_w.is_finite()
            || !net_heat_j.is_finite()
            || !thermal_store_k.is_finite()
        {
            return Err(PlssStepError::InvalidInput);
        }

        self.state.co2_burden_l = co2_burden_l;
        self.state.co2_scrubber_capacity_remaining_l = co2_capacity_remaining_l;
        self.state.humidity_vapor_burden_g = humidity_vapor_burden_g;
        self.state.humidity_collector_capacity_remaining_g = humidity_capacity_remaining_g;
        self.state.thermal_store_k = thermal_store_k;

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

    fn input() -> PlssStepInput {
        PlssStepInput {
            metabolism: metabolism(),
            equipment_heat_w: 50.0,
            humidity_generation_g_per_min: 0.2,
            dt_s: 60.0,
        }
    }

    #[test]
    fn nominal_step_consumes_oxygen() {
        let mut twin = PlssReferenceTwin::simulation_reference();
        let before = twin.state().primary_o2_remaining_l;
        twin.step(input()).unwrap();
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
            humidity_generation_g_per_min: 0.0,
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
            humidity_generation_g_per_min: 0.0,
            dt_s: 1.0,
        });
        assert_eq!(result, Err(PlssStepError::NoThermalPath));
    }

    #[test]
    fn co2_removal_is_bounded_by_finite_sorbent_capacity_even_at_high_fan_flow() {
        let mut config = PlssConfig::simulation_reference();
        config.co2_scrubber_removal_rate_l_min = 100.0;
        config.co2_scrubber_capacity_l = 0.10;
        let mut twin = PlssReferenceTwin::new(config).unwrap();
        let metabolic = metabolism();
        let generated_l = metabolic.co2_l_min;
        twin.step(PlssStepInput {
            metabolism: metabolic,
            equipment_heat_w: 0.0,
            humidity_generation_g_per_min: 0.0,
            dt_s: 60.0,
        })
        .unwrap();
        assert!((twin.state().co2_burden_l - (generated_l - 0.10)).abs() < 1e-9);
        assert_eq!(twin.state().co2_scrubber_capacity_remaining_l, 0.0);
    }

    #[test]
    fn failed_scrubber_accumulates_co2_despite_nominal_ventilation() {
        let mut twin = PlssReferenceTwin::simulation_reference();
        twin.state_mut_for_fault_injection().co2_scrubber = PlssPathState::Failed;
        let generated_l = metabolism().co2_l_min;
        twin.step(PlssStepInput {
            metabolism: metabolism(),
            equipment_heat_w: 0.0,
            humidity_generation_g_per_min: 0.0,
            dt_s: 60.0,
        })
        .unwrap();
        assert!((twin.state().co2_burden_l - generated_l).abs() < 1e-9);
        assert_eq!(twin.state().co2_scrubber_capacity_remaining_l, 120.0);
    }

    #[test]
    fn humidity_removal_is_bounded_by_finite_condensate_capacity() {
        let mut config = PlssConfig::simulation_reference();
        config.humidity_removal_rate_g_min = 100.0;
        config.humidity_collector_capacity_g = 0.10;
        let mut twin = PlssReferenceTwin::new(config).unwrap();
        twin.step(PlssStepInput {
            metabolism: metabolism(),
            equipment_heat_w: 0.0,
            humidity_generation_g_per_min: 0.25,
            dt_s: 60.0,
        })
        .unwrap();
        assert!((twin.state().humidity_vapor_burden_g - 0.15).abs() < 1e-9);
        assert_eq!(twin.state().humidity_collector_capacity_remaining_g, 0.0);
    }

    #[test]
    fn failed_humidity_control_accumulates_water_vapor_mass() {
        let mut twin = PlssReferenceTwin::simulation_reference();
        twin.state_mut_for_fault_injection().humidity_control = PlssPathState::Failed;
        twin.step(PlssStepInput {
            metabolism: metabolism(),
            equipment_heat_w: 0.0,
            humidity_generation_g_per_min: 0.25,
            dt_s: 60.0,
        })
        .unwrap();
        assert!((twin.state().humidity_vapor_burden_g - 0.25).abs() < 1e-9);
    }

    #[test]
    fn overflowed_mass_balance_is_rejected_before_oxygen_consumption() {
        let mut twin = PlssReferenceTwin::simulation_reference();
        let oxygen_before =
            twin.state().primary_o2_remaining_l + twin.state().secondary_o2_remaining_l;
        let result = twin.step(PlssStepInput {
            metabolism: metabolism(),
            equipment_heat_w: 0.0,
            humidity_generation_g_per_min: f64::MAX,
            dt_s: 120.0,
        });
        assert_eq!(result, Err(PlssStepError::InvalidInput));
        assert_eq!(
            twin.state().primary_o2_remaining_l + twin.state().secondary_o2_remaining_l,
            oxygen_before
        );
    }

    #[test]
    fn radiator_derating_increases_stored_heat() {
        let mut clean = PlssReferenceTwin::simulation_reference();
        let mut dusted = PlssReferenceTwin::simulation_reference();
        clean.step_with_heat_rejection_fraction(input(), 1.0).unwrap();
        dusted.step_with_heat_rejection_fraction(input(), 0.5).unwrap();
        assert!(dusted.state().thermal_store_k > clean.state().thermal_store_k);
    }

    #[test]
    fn persistent_environment_derating_affects_nominal_step() {
        let mut clean = PlssReferenceTwin::simulation_reference();
        let mut dusted = PlssReferenceTwin::simulation_reference();
        dusted.set_environment_heat_rejection_fraction(0.5).unwrap();
        clean.step(input()).unwrap();
        dusted.step(input()).unwrap();
        assert!(dusted.state().thermal_store_k > clean.state().thermal_store_k);
        assert_eq!(dusted.environment_heat_rejection_fraction(), 0.5);
    }

    #[test]
    fn malformed_heat_rejection_fraction_fails_closed() {
        let mut twin = PlssReferenceTwin::simulation_reference();
        assert_eq!(
            twin.step_with_heat_rejection_fraction(input(), f64::NAN),
            Err(PlssStepError::InvalidInput)
        );
        assert_eq!(
            twin.set_environment_heat_rejection_fraction(1.1),
            Err(PlssStepError::InvalidInput)
        );
    }
}
