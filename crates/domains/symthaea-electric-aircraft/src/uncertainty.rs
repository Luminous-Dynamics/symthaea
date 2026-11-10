// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic adverse-corner evaluation for bounded mission uncertainty.
//!
//! This module applies explicitly declared one-sided fractional bounds to
//! conservative directions (higher mass/speed/loads/resistance; lower L/D,
//! capacity, voltage, efficiencies, limits and thermal headroom). It evaluates
//! the resulting corner with the ordinary scalar mission kernel.
//!
//! This is not Monte Carlo, a probability distribution, a confidence interval,
//! a certified worst-case proof, or a substitute for correlated uncertainty
//! analysis. It is a transparent screening tool for identifying whether the
//! current nominal design has obvious margin fragility.

use std::fmt;

use crate::{evaluate_mission, MissionInput, MissionReport, ModelError};

pub const ROBUST_UNCERTAINTY_MODEL_VERSION: &str =
    "electric-fixed-wing-adverse-corner-v0.1.0";

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MissionUncertaintyBudget {
    pub mass_fraction: f64,
    pub airspeed_fraction: f64,
    pub lift_to_drag_fraction: f64,
    pub positive_climb_rate_fraction: f64,
    pub auxiliary_power_fraction: f64,
    pub battery_capacity_fraction: f64,
    pub battery_voltage_fraction: f64,
    pub battery_resistance_fraction: f64,
    pub battery_power_limit_fraction: f64,
    pub battery_current_limit_fraction: f64,
    pub battery_thermal_capacity_fraction: f64,
    pub battery_thermal_conductance_fraction: f64,
    pub battery_temperature_headroom_fraction: f64,
    pub propulsion_efficiency_fraction: f64,
}

impl Default for MissionUncertaintyBudget {
    fn default() -> Self {
        Self {
            mass_fraction: 0.0,
            airspeed_fraction: 0.0,
            lift_to_drag_fraction: 0.0,
            positive_climb_rate_fraction: 0.0,
            auxiliary_power_fraction: 0.0,
            battery_capacity_fraction: 0.0,
            battery_voltage_fraction: 0.0,
            battery_resistance_fraction: 0.0,
            battery_power_limit_fraction: 0.0,
            battery_current_limit_fraction: 0.0,
            battery_thermal_capacity_fraction: 0.0,
            battery_thermal_conductance_fraction: 0.0,
            battery_temperature_headroom_fraction: 0.0,
            propulsion_efficiency_fraction: 0.0,
        }
    }
}

impl MissionUncertaintyBudget {
    /// Validate the declared fractions. A value of 0.05 means a 5% one-sided
    /// bound. Values must be finite and in [0, 1); unity would erase a positive
    /// parameter or a required limit and is therefore rejected.
    pub fn validate(&self) -> Result<(), RobustEvaluationError> {
        let values = [
            self.mass_fraction,
            self.airspeed_fraction,
            self.lift_to_drag_fraction,
            self.positive_climb_rate_fraction,
            self.auxiliary_power_fraction,
            self.battery_capacity_fraction,
            self.battery_voltage_fraction,
            self.battery_resistance_fraction,
            self.battery_power_limit_fraction,
            self.battery_current_limit_fraction,
            self.battery_thermal_capacity_fraction,
            self.battery_thermal_conductance_fraction,
            self.battery_temperature_headroom_fraction,
            self.propulsion_efficiency_fraction,
        ];
        if values
            .iter()
            .any(|value| !value.is_finite() || !(0.0..1.0).contains(value))
        {
            return Err(RobustEvaluationError::InvalidBudget(
                "all uncertainty fractions must be finite and in [0, 1)",
            ));
        }
        Ok(())
    }

    /// Materialize the simultaneous adverse corner so the exact adjusted
    /// assumptions can be inspected, persisted, compared, or independently
    /// replayed. This does not claim the corner is statistically likely.
    pub fn apply_adverse_corner(
        &self,
        input: &MissionInput,
    ) -> Result<MissionInput, RobustEvaluationError> {
        self.validate()?;
        let mut adjusted = input.clone();

        let mass_scale = 1.0 + self.mass_fraction;
        adjusted.aircraft.empty_mass_kg *= mass_scale;
        adjusted.aircraft.payload_mass_kg *= mass_scale;
        adjusted.battery.pack_mass_kg *= mass_scale;

        let speed_scale = 1.0 + self.airspeed_fraction;
        let aero_scale = 1.0 - self.lift_to_drag_fraction;
        let climb_scale = 1.0 + self.positive_climb_rate_fraction;
        let auxiliary_scale = 1.0 + self.auxiliary_power_fraction;
        for segment in &mut adjusted.segments {
            segment.true_airspeed_mps *= speed_scale;
            segment.lift_to_drag_ratio *= aero_scale;
            if segment.climb_rate_mps > 0.0 {
                segment.climb_rate_mps *= climb_scale;
            }
            segment.auxiliary_power_w *= auxiliary_scale;
        }

        let capacity_scale = 1.0 - self.battery_capacity_fraction;
        adjusted.battery.state_of_health *= capacity_scale;
        adjusted.battery.nominal_open_circuit_voltage_v *=
            1.0 - self.battery_voltage_fraction;
        adjusted.battery.internal_resistance_ohm *=
            1.0 + self.battery_resistance_fraction;
        adjusted.battery.maximum_continuous_power_w *=
            1.0 - self.battery_power_limit_fraction;
        adjusted.battery.maximum_continuous_current_a *=
            1.0 - self.battery_current_limit_fraction;
        adjusted.battery.thermal_capacity_j_per_k *=
            1.0 - self.battery_thermal_capacity_fraction;
        adjusted.battery.thermal_conductance_w_per_k *=
            1.0 - self.battery_thermal_conductance_fraction;
        let thermal_headroom = adjusted.battery.maximum_temperature_k
            - adjusted.battery.ambient_temperature_k;
        adjusted.battery.maximum_temperature_k =
            adjusted.battery.ambient_temperature_k
                + thermal_headroom * (1.0 - self.battery_temperature_headroom_fraction);

        let efficiency_scale = 1.0 - self.propulsion_efficiency_fraction;
        adjusted.propulsion.motor_efficiency *= efficiency_scale;
        adjusted.propulsion.inverter_efficiency *= efficiency_scale;
        adjusted.propulsion.propeller_efficiency *= efficiency_scale;

        Ok(adjusted)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum RobustEvaluationError {
    InvalidBudget(&'static str),
    NominalMission(ModelError),
}

impl fmt::Display for RobustEvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for RobustEvaluationError {}

#[derive(Debug, Clone, PartialEq)]
pub struct RobustMissionReport {
    pub uncertainty_model_version: &'static str,
    pub budget: MissionUncertaintyBudget,
    pub nominal: MissionReport,
    /// Exact input after applying all adverse one-sided bounds.
    pub adverse_corner_input: MissionInput,
    /// Failure here means the declared adverse corner is infeasible under
    /// the current model. The nominal run remains available for diagnosis.
    pub adverse_corner_result: Result<MissionReport, ModelError>,
}

impl RobustMissionReport {
    pub fn is_feasible_at_adverse_corner(&self) -> bool {
        self.adverse_corner_result.is_ok()
    }

    pub fn adverse_corner_energy_margin_wh(&self) -> Option<f64> {
        self.adverse_corner_result
            .as_ref()
            .ok()
            .map(|report| report.energy_margin_wh)
    }
}

/// Evaluate the nominal mission and the explicitly bounded adverse corner.
///
/// A nominally infeasible mission is returned as RobustEvaluationError.
/// An infeasible adverse corner is instead retained inside the report so
/// callers can distinguish nominal failure from uncertainty-corner failure.
pub fn evaluate_mission_robust(
    input: &MissionInput,
    budget: &MissionUncertaintyBudget,
) -> Result<RobustMissionReport, RobustEvaluationError> {
    budget.validate()?;
    let nominal = evaluate_mission(input).map_err(RobustEvaluationError::NominalMission)?;
    let adverse_corner_input = budget.apply_adverse_corner(input)?;
    let adverse_corner_result = evaluate_mission(&adverse_corner_input);

    Ok(RobustMissionReport {
        uncertainty_model_version: ROBUST_UNCERTAINTY_MODEL_VERSION,
        budget: *budget,
        nominal,
        adverse_corner_input,
        adverse_corner_result,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AircraftConfig, BatteryConfig, FlightPhase, MissionSegment, PropulsionConfig};

    fn baseline() -> MissionInput {
        MissionInput {
            aircraft: AircraftConfig {
                empty_mass_kg: 350.0,
                payload_mass_kg: 50.0,
                maximum_takeoff_mass_kg: 700.0,
            },
            battery: BatteryConfig {
                nominal_energy_wh: 40_000.0,
                pack_mass_kg: 240.0,
                nominal_open_circuit_voltage_v: 400.0,
                maximum_continuous_power_w: 75_000.0,
                maximum_continuous_current_a: 250.0,
                internal_resistance_ohm: 0.04,
                thermal_capacity_j_per_k: 80_000.0,
                thermal_conductance_w_per_k: 140.0,
                maximum_temperature_k: 333.15,
                initial_temperature_k: 298.15,
                ambient_temperature_k: 293.15,
                state_of_health: 0.95,
                initial_state_of_charge: 0.95,
                reserve_state_of_charge: 0.20,
            },
            propulsion: PropulsionConfig {
                motor_efficiency: 0.94,
                inverter_efficiency: 0.97,
                propeller_efficiency: 0.82,
            },
            mission_uncertainty_fraction: 0.20,
            segments: vec![
                MissionSegment {
                    id: "climb".into(),
                    phase: FlightPhase::Climb,
                    duration_s: 120.0,
                    true_airspeed_mps: 35.0,
                    lift_to_drag_ratio: 12.0,
                    climb_rate_mps: 3.0,
                    auxiliary_power_w: 1_000.0,
                },
                MissionSegment {
                    id: "cruise".into(),
                    phase: FlightPhase::Cruise,
                    duration_s: 1_200.0,
                    true_airspeed_mps: 55.0,
                    lift_to_drag_ratio: 15.0,
                    climb_rate_mps: 0.0,
                    auxiliary_power_w: 800.0,
                },
                MissionSegment {
                    id: "descent".into(),
                    phase: FlightPhase::Descent,
                    duration_s: 300.0,
                    true_airspeed_mps: 40.0,
                    lift_to_drag_ratio: 12.0,
                    climb_rate_mps: -2.0,
                    auxiliary_power_w: 700.0,
                },
            ],
        }
    }

    #[test]
    fn zero_uncertainty_reproduces_nominal_result() {
        let input = baseline();
        let nominal = evaluate_mission(&input).unwrap();
        let robust = evaluate_mission_robust(&input, &MissionUncertaintyBudget::default()).unwrap();
        let adverse = robust.adverse_corner_result.unwrap();

        assert_eq!(robust.uncertainty_model_version, ROBUST_UNCERTAINTY_MODEL_VERSION);
        assert_eq!(robust.budget, MissionUncertaintyBudget::default());
        assert_eq!(robust.nominal, nominal);
        assert_eq!(robust.adverse_corner_input, input);
        assert_eq!(adverse, nominal);
    }

    #[test]
    fn adverse_corner_reduces_margin_and_exposes_adjusted_inputs() {
        let input = baseline();
        let budget = MissionUncertaintyBudget {
            mass_fraction: 0.01,
            airspeed_fraction: 0.01,
            lift_to_drag_fraction: 0.01,
            positive_climb_rate_fraction: 0.01,
            auxiliary_power_fraction: 0.01,
            battery_capacity_fraction: 0.01,
            battery_voltage_fraction: 0.01,
            battery_resistance_fraction: 0.01,
            battery_power_limit_fraction: 0.01,
            battery_current_limit_fraction: 0.01,
            battery_thermal_capacity_fraction: 0.01,
            battery_thermal_conductance_fraction: 0.01,
            battery_temperature_headroom_fraction: 0.01,
            propulsion_efficiency_fraction: 0.01,
        };
        let robust = evaluate_mission_robust(&input, &budget).unwrap();
        let adverse = robust.adverse_corner_result.unwrap();

        assert_eq!(robust.uncertainty_model_version, ROBUST_UNCERTAINTY_MODEL_VERSION);
        assert_eq!(robust.budget, budget);
        assert!(adverse.mission_terminal_energy_wh > robust.nominal.mission_terminal_energy_wh);
        assert!(adverse.energy_margin_wh < robust.nominal.energy_margin_wh);
        assert!(adverse.all_up_mass_kg > robust.nominal.all_up_mass_kg);
        assert!(adverse.maximum_pack_temperature_k >= robust.nominal.maximum_pack_temperature_k);
    }

    #[test]
    fn adverse_corner_can_fail_mtow_without_hiding_nominal_pass() {
        let mut input = baseline();
        input.aircraft.maximum_takeoff_mass_kg = 645.0;
        let budget = MissionUncertaintyBudget {
            mass_fraction: 0.02,
            ..MissionUncertaintyBudget::default()
        };
        let report = evaluate_mission_robust(&input, &budget).unwrap();

        assert!(report.nominal.all_up_mass_kg <= input.aircraft.maximum_takeoff_mass_kg);
        assert!(matches!(
            &report.adverse_corner_result,
            Err(ModelError::MaximumTakeoffMassExceeded { .. })
        ));
        assert!(!report.is_feasible_at_adverse_corner());
        assert!(report.adverse_corner_energy_margin_wh().is_none());
    }

    #[test]
    fn adverse_capacity_corner_can_fail_reserve_while_nominal_passes() {
        let input = baseline();
        let budget = MissionUncertaintyBudget {
            battery_capacity_fraction: 0.50,
            ..MissionUncertaintyBudget::default()
        };
        let report = evaluate_mission_robust(&input, &budget).unwrap();

        assert!(report.nominal.energy_margin_wh > 0.0);
        assert!(matches!(
            &report.adverse_corner_result,
            Err(ModelError::ReserveInfringed { .. })
        ));
        assert!(!report.is_feasible_at_adverse_corner());
    }

    #[test]
    fn uncertainty_budget_rejects_non_finite_and_unit_fractions() {
        let input = baseline();
        let mut invalid = MissionUncertaintyBudget::default();
        invalid.mass_fraction = 1.0;
        assert!(matches!(
            evaluate_mission_robust(&input, &invalid),
            Err(RobustEvaluationError::InvalidBudget(_))
        ));

        invalid.mass_fraction = f64::NAN;
        assert!(matches!(
            evaluate_mission_robust(&input, &invalid),
            Err(RobustEvaluationError::InvalidBudget(_))
        ));
    }
}
