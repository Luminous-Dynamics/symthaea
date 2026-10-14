// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Lumped electrical-to-thermal reference model for Joule heating.
//!
//! This is an intentionally small verification case, not a spatial EM or heat
//! solver. It assumes constant resistance and specific heat capacity, a
//! well-mixed conductor, no phase change, and no heat loss to the surroundings.
//! Under those assumptions, electrical work must equal the increase in stored
//! sensible heat:
//!
//!     P_J = I² R
//!     Q_in = P_J Δt
//!     ΔT = Q_in / (m c_p)
//!
//! All values use SI units. Temperature is an absolute temperature in kelvin.

use std::error::Error;
use std::fmt;

/// Invalid input or state transition in this model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JouleHeatingError {
    /// Resistance must be finite and greater than zero.
    InvalidResistance,
    /// Conductor mass must be finite and greater than zero.
    InvalidMass,
    /// Specific heat capacity must be finite and greater than zero.
    InvalidSpecificHeatCapacity,
    /// Initial absolute temperature must be finite and greater than zero.
    InvalidTemperature,
    /// Current must be finite. Negative current is permitted.
    InvalidCurrent,
    /// Time step must be finite and greater than zero.
    InvalidTimeStep,
    /// A valid-looking input produced a non-finite derived quantity.
    NonFiniteResult,
}

impl fmt::Display for JouleHeatingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidResistance => write!(f, "resistance must be finite and positive"),
            Self::InvalidMass => write!(f, "mass must be finite and positive"),
            Self::InvalidSpecificHeatCapacity => {
                write!(f, "specific heat capacity must be finite and positive")
            }
            Self::InvalidTemperature => {
                write!(f, "absolute temperature must be finite and positive")
            }
            Self::InvalidCurrent => write!(f, "current must be finite"),
            Self::InvalidTimeStep => write!(f, "time step must be finite and positive"),
            Self::NonFiniteResult => write!(f, "model step produced a non-finite result"),
        }
    }
}

impl Error for JouleHeatingError {}

/// Energy-accounted result for one adiabatic Joule-heating step.
///
/// The residual is defined as input electrical energy minus the increase in
/// sensible thermal energy. It should be close to floating-point roundoff for
/// this lumped model; it is not a general-purpose estimate of discretization
/// error in spatially resolved simulations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JouleHeatingStep {
    /// Current used for this step (A; sign does not affect resistive heating).
    pub current_a: f64,
    /// Duration of this step (s).
    pub time_step_s: f64,
    /// Initial conductor temperature (K).
    pub initial_temperature_k: f64,
    /// Final conductor temperature (K).
    pub final_temperature_k: f64,
    /// Resistive heating power, I²R (W).
    pub heating_power_w: f64,
    /// Electrical energy converted to heat during the step (J).
    pub electrical_energy_in_j: f64,
    /// Increase in conductor sensible thermal energy (J).
    pub stored_energy_change_j: f64,
    /// Energy residual: electrical input minus stored thermal-energy change (J).
    pub energy_balance_residual_j: f64,
}

/// Single lumped conductor with fixed electrical resistance and heat capacity.
///
/// This small model is useful as an analytical reference for later spatially
/// resolved EM/thermal coupling. It intentionally has no ambient cooling or
/// temperature-dependent material properties.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JouleHeatingModel {
    resistance_ohm: f64,
    mass_kg: f64,
    specific_heat_capacity_j_per_kg_k: f64,
    temperature_k: f64,
}

impl JouleHeatingModel {
    /// Create a conductor model using SI units.
    ///
    /// Returns an error if a parameter is non-finite or outside its positive
    /// physical domain, or if the derived heat capacity overflows.
    pub fn new(
        resistance_ohm: f64,
        mass_kg: f64,
        specific_heat_capacity_j_per_kg_k: f64,
        initial_temperature_k: f64,
    ) -> Result<Self, JouleHeatingError> {
        if !resistance_ohm.is_finite() || resistance_ohm <= 0.0 {
            return Err(JouleHeatingError::InvalidResistance);
        }
        if !mass_kg.is_finite() || mass_kg <= 0.0 {
            return Err(JouleHeatingError::InvalidMass);
        }
        if !specific_heat_capacity_j_per_kg_k.is_finite()
            || specific_heat_capacity_j_per_kg_k <= 0.0
        {
            return Err(JouleHeatingError::InvalidSpecificHeatCapacity);
        }
        if !initial_temperature_k.is_finite() || initial_temperature_k <= 0.0 {
            return Err(JouleHeatingError::InvalidTemperature);
        }
        let heat_capacity = mass_kg * specific_heat_capacity_j_per_kg_k;
        if !heat_capacity.is_finite() || heat_capacity <= 0.0 {
            return Err(JouleHeatingError::NonFiniteResult);
        }

        Ok(Self {
            resistance_ohm,
            mass_kg,
            specific_heat_capacity_j_per_kg_k,
            temperature_k: initial_temperature_k,
        })
    }

    /// Current conductor temperature in kelvin.
    pub fn temperature_k(&self) -> f64 {
        self.temperature_k
    }

    /// Total lumped thermal capacitance, m·cₚ (J/K).
    pub fn thermal_capacity_j_per_k(&self) -> f64 {
        self.mass_kg * self.specific_heat_capacity_j_per_kg_k
    }

    /// Advance one adiabatic step using Joule heating.
    ///
    /// Signed current is allowed because reversing current in a fixed,
    /// ohmic resistor does not change I²R. The state is updated only after all
    /// derived values have been checked for finiteness.
    pub fn step(
        &mut self,
        current_a: f64,
        time_step_s: f64,
    ) -> Result<JouleHeatingStep, JouleHeatingError> {
        if !current_a.is_finite() {
            return Err(JouleHeatingError::InvalidCurrent);
        }
        if !time_step_s.is_finite() || time_step_s <= 0.0 {
            return Err(JouleHeatingError::InvalidTimeStep);
        }

        let heating_power_w = current_a * current_a * self.resistance_ohm;
        let electrical_energy_in_j = heating_power_w * time_step_s;
        let initial_temperature_k = self.temperature_k;
        let temperature_rise_k =
            electrical_energy_in_j / self.thermal_capacity_j_per_k();
        let final_temperature_k = initial_temperature_k + temperature_rise_k;

        if !heating_power_w.is_finite()
            || !electrical_energy_in_j.is_finite()
            || !temperature_rise_k.is_finite()
            || !final_temperature_k.is_finite()
            || final_temperature_k <= 0.0
        {
            return Err(JouleHeatingError::NonFiniteResult);
        }

        let stored_energy_change_j =
            self.thermal_capacity_j_per_k() * (final_temperature_k - initial_temperature_k);
        let energy_balance_residual_j = electrical_energy_in_j - stored_energy_change_j;
        if !stored_energy_change_j.is_finite() || !energy_balance_residual_j.is_finite() {
            return Err(JouleHeatingError::NonFiniteResult);
        }

        self.temperature_k = final_temperature_k;

        Ok(JouleHeatingStep {
            current_a,
            time_step_s,
            initial_temperature_k,
            final_temperature_k,
            heating_power_w,
            electrical_energy_in_j,
            stored_energy_change_j,
            energy_balance_residual_j,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_model() -> JouleHeatingModel {
        // m cₚ = 0.5 kg × 900 J/(kg K) = 450 J/K.
        JouleHeatingModel::new(2.0, 0.5, 900.0, 293.15).unwrap()
    }

    #[test]
    fn textbook_case_matches_power_energy_and_temperature_rise() {
        // I = 3 A, R = 2 Ω => P = 18 W; over 10 s, Q = 180 J;
        // ΔT = 180 J / 450 J/K = 0.4 K.
        let mut model = reference_model();
        let step = model.step(3.0, 10.0).unwrap();

        assert!((step.heating_power_w - 18.0).abs() < 1e-12);
        assert!((step.electrical_energy_in_j - 180.0).abs() < 1e-12);
        assert!((step.final_temperature_k - 293.55).abs() < 1e-12);
        assert!((model.temperature_k() - step.final_temperature_k).abs() < 1e-12);
    }

    #[test]
    fn electrical_input_equals_stored_energy_change_within_roundoff() {
        let mut model = reference_model();
        let step = model.step(3.0, 10.0).unwrap();

        assert!(step.energy_balance_residual_j.abs() < 1e-10);
        assert!((step.electrical_energy_in_j - step.stored_energy_change_j).abs() < 1e-10);
    }

    #[test]
    fn reversing_current_preserves_resistive_heating() {
        let mut forward = reference_model();
        let mut reverse = reference_model();

        let forward_step = forward.step(4.0, 2.5).unwrap();
        let reverse_step = reverse.step(-4.0, 2.5).unwrap();

        assert_eq!(forward_step.heating_power_w, reverse_step.heating_power_w);
        assert_eq!(forward_step.final_temperature_k, reverse_step.final_temperature_k);
    }

    #[test]
    fn zero_current_produces_no_temperature_change() {
        let mut model = reference_model();
        let before = model.temperature_k();
        let step = model.step(0.0, 5.0).unwrap();

        assert_eq!(step.heating_power_w, 0.0);
        assert_eq!(step.electrical_energy_in_j, 0.0);
        assert_eq!(step.final_temperature_k, before);
    }

    #[test]
    fn rejects_invalid_physical_parameters() {
        assert_eq!(
            JouleHeatingModel::new(0.0, 1.0, 500.0, 300.0).unwrap_err(),
            JouleHeatingError::InvalidResistance
        );
        assert_eq!(
            JouleHeatingModel::new(1.0, -1.0, 500.0, 300.0).unwrap_err(),
            JouleHeatingError::InvalidMass
        );
        assert_eq!(
            JouleHeatingModel::new(1.0, 1.0, f64::INFINITY, 300.0).unwrap_err(),
            JouleHeatingError::InvalidSpecificHeatCapacity
        );
        assert_eq!(
            JouleHeatingModel::new(1.0, 1.0, 500.0, f64::NAN).unwrap_err(),
            JouleHeatingError::InvalidTemperature
        );
    }

    #[test]
    fn invalid_step_does_not_mutate_the_state() {
        let mut model = reference_model();
        let before = model.temperature_k();

        assert_eq!(model.step(f64::NAN, 1.0), Err(JouleHeatingError::InvalidCurrent));
        assert_eq!(model.step(2.0, 0.0), Err(JouleHeatingError::InvalidTimeStep));
        assert_eq!(model.temperature_k(), before);
    }

    #[test]
    fn rejects_overflow_without_mutating_state() {
        let mut model = JouleHeatingModel::new(1e308, 1.0, 500.0, 300.0).unwrap();
        let before = model.temperature_k();

        assert_eq!(
            model.step(1e308, 1.0),
            Err(JouleHeatingError::NonFiniteResult)
        );
        assert_eq!(model.temperature_k(), before);
    }
}
