// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! PIE-002 first-order electrical and thermal utility accounting.
//!
//! This module mirrors the independent `scripts/pie-utility-accounting-oracle.py`
//! contract. It deliberately separates gross/current-cycle supply screening from
//! steady-cycle recovery accounting. It does not prove thermodynamic feasibility,
//! heat-exchanger performance, storage dispatch, or plant authority.

use serde::{Deserialize, Serialize};

use crate::{
    DurationRangeS, EnergyRangeJ, OntologyError, PowerRangeW, TemperatureRangeK,
};

/// Conservative demand-versus-capacity classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UtilityFeasibility {
    /// Every admissible demand is within every admissible capacity realization.
    Guaranteed,
    /// The declared intervals overlap, so feasibility depends on the realization.
    Possible,
    /// Even the minimum demand exceeds the maximum available capacity.
    Impossible,
}

/// Inclusive dimensionless fraction constrained to `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FractionRange {
    min: f64,
    max: f64,
}

impl FractionRange {
    /// Construct a validated inclusive fraction range.
    pub fn new(min: f64, max: f64) -> Result<Self, OntologyError> {
        let range = Self { min, max };
        range.validate()?;
        Ok(range)
    }

    /// Conservative lower bound.
    pub fn min(self) -> f64 {
        self.min
    }

    /// Conservative upper bound.
    pub fn max(self) -> f64 {
        self.max
    }

    fn validate(self) -> Result<(), OntologyError> {
        if !self.min.is_finite()
            || !self.max.is_finite()
            || self.min > self.max
            || !(0.0..=1.0).contains(&self.min)
            || !(0.0..=1.0).contains(&self.max)
        {
            Err(OntologyError::InvalidRange("utility_fraction"))
        } else {
            Ok(())
        }
    }
}

/// Explicit electrical accounting case for one declared process basis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElectricalUtilityCase {
    /// Gross electrical energy required by the process cycle.
    pub gross_energy_j: EnergyRangeJ,
    /// Duration of the process batch/cycle; must be strictly positive.
    pub batch_duration_s: DurationRangeS,
    /// Gross process peak electrical power; recovery does not reduce this field.
    pub peak_power_w: PowerRangeW,
    /// Energy physically recoverable from the process before storage limits.
    pub recoverable_energy_j: EnergyRangeJ,
    /// Duration over which the recoverable stream is available; strictly positive.
    pub recovery_duration_s: DurationRangeS,
    /// Remaining storage energy acceptance available to the recovery stream.
    pub storage_acceptance_j: EnergyRangeJ,
    /// Storage charge-power acceptance during the recovery window.
    pub storage_charge_power_w: PowerRangeW,
    /// Storage discharge-power capability available to a subsequent cycle.
    pub storage_discharge_power_w: PowerRangeW,
    /// Fraction of accepted recovery deliverable after storage/conversion losses.
    pub recovery_delivery_fraction: FractionRange,
    /// Electrical energy capacity available to the process basis.
    pub available_energy_capacity_j: EnergyRangeJ,
    /// Sustained source power available over the process basis.
    pub available_continuous_power_w: PowerRangeW,
    /// Short-duration/peak source or buffer power available to the process.
    pub available_peak_power_w: PowerRangeW,
}

/// Electrical utility accounting and screening result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElectricalUtilityReport {
    /// Gross process electrical energy, preserved without netting.
    pub gross_energy_j: EnergyRangeJ,
    /// Recovery accepted after source, storage-energy and charge-power limits.
    pub accepted_recovery_j: EnergyRangeJ,
    /// Recovery reusable by a subsequent steady cycle after delivery and discharge limits.
    pub usable_steady_cycle_recovery_j: EnergyRangeJ,
    /// Gross energy minus usable steady-cycle recovery.
    pub net_steady_cycle_energy_j: EnergyRangeJ,
    /// Gross process energy divided by batch duration.
    pub gross_average_power_w: PowerRangeW,
    /// Net steady-cycle energy divided by batch duration.
    pub net_steady_cycle_average_power_w: PowerRangeW,
    /// Gross process peak power, unchanged by recovery accounting.
    pub peak_power_w: PowerRangeW,
    /// Current/gross energy-capacity screen with no temporal recovery credit.
    pub gross_energy_capacity: UtilityFeasibility,
    /// Separately named steady-cycle energy-capacity screen.
    pub steady_cycle_energy_capacity: UtilityFeasibility,
    /// Current/gross sustained-power screen with no temporal recovery credit.
    pub gross_continuous_power: UtilityFeasibility,
    /// Separately named steady-cycle average-power screen.
    pub steady_cycle_continuous_power: UtilityFeasibility,
    /// Gross peak-power screen.
    pub peak_power: UtilityFeasibility,
}

/// Candidate thermal-energy supply for first-order screening.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThermalSupply {
    /// Thermal energy available from the source.
    pub energy_j: EnergyRangeJ,
    /// Source temperature envelope in kelvin; must be strictly positive.
    pub source_temperature_k: TemperatureRangeK,
}

/// Thermal-energy demand for first-order screening.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThermalDemand {
    /// Thermal energy required by the process.
    pub energy_j: EnergyRangeJ,
    /// Required process-temperature envelope in kelvin; strictly positive.
    pub required_temperature_k: TemperatureRangeK,
}

/// Thermal quantity/temperature compatibility screen.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThermalUtilityReport {
    /// Energy-quantity feasibility only.
    pub energy_feasibility: UtilityFeasibility,
    /// Temperature-envelope compatibility only.
    pub temperature_compatibility: UtilityFeasibility,
    /// Conservative combination of quantity and temperature screens.
    pub combined_screening: UtilityFeasibility,
}

fn classify(
    demand_min: f64,
    demand_max: f64,
    capacity_min: f64,
    capacity_max: f64,
) -> UtilityFeasibility {
    if demand_max <= capacity_min {
        UtilityFeasibility::Guaranteed
    } else if demand_min > capacity_max {
        UtilityFeasibility::Impossible
    } else {
        UtilityFeasibility::Possible
    }
}

fn energy_feasibility(
    demand: EnergyRangeJ,
    capacity: EnergyRangeJ,
) -> Result<UtilityFeasibility, OntologyError> {
    demand.validate()?;
    capacity.validate()?;
    Ok(classify(
        demand.min.value(),
        demand.max.value(),
        capacity.min.value(),
        capacity.max.value(),
    ))
}

fn power_feasibility(
    demand: PowerRangeW,
    capacity: PowerRangeW,
) -> Result<UtilityFeasibility, OntologyError> {
    demand.validate()?;
    capacity.validate()?;
    Ok(classify(
        demand.min.value(),
        demand.max.value(),
        capacity.min.value(),
        capacity.max.value(),
    ))
}

fn validate_positive_duration(
    duration: DurationRangeS,
    label: &'static str,
) -> Result<(), OntologyError> {
    duration.validate()?;
    if duration.min.value() > 0.0 {
        Ok(())
    } else {
        Err(OntologyError::InvalidRange(label))
    }
}

fn validate_positive_temperature(
    temperature: TemperatureRangeK,
    label: &'static str,
) -> Result<(), OntologyError> {
    temperature.validate()?;
    if temperature.min.value() > 0.0 {
        Ok(())
    } else {
        Err(OntologyError::InvalidRange(label))
    }
}

fn finite_product(a: f64, b: f64, label: &'static str) -> Result<f64, OntologyError> {
    let value = a * b;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(OntologyError::InvalidQuantity(label))
    }
}

fn finite_quotient(a: f64, b: f64, label: &'static str) -> Result<f64, OntologyError> {
    let value = a / b;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(OntologyError::InvalidQuantity(label))
    }
}

fn power_times_duration(
    power: PowerRangeW,
    duration: DurationRangeS,
    label: &'static str,
) -> Result<EnergyRangeJ, OntologyError> {
    power.validate()?;
    validate_positive_duration(duration, label)?;
    EnergyRangeJ::new(
        finite_product(power.min.value(), duration.min.value(), label)?,
        finite_product(power.max.value(), duration.max.value(), label)?,
    )
}

fn energy_times_fraction(
    energy: EnergyRangeJ,
    fraction: FractionRange,
    label: &'static str,
) -> Result<EnergyRangeJ, OntologyError> {
    energy.validate()?;
    fraction.validate()?;
    EnergyRangeJ::new(
        finite_product(energy.min.value(), fraction.min(), label)?,
        finite_product(energy.max.value(), fraction.max(), label)?,
    )
}

fn minimum_energy_ranges(
    ranges: &[EnergyRangeJ],
    label: &'static str,
) -> Result<EnergyRangeJ, OntologyError> {
    let first = ranges
        .first()
        .copied()
        .ok_or(OntologyError::InvalidRange(label))?;
    first.validate()?;
    let mut min = first.min.value();
    let mut max = first.max.value();
    for range in &ranges[1..] {
        range.validate()?;
        min = min.min(range.min.value());
        max = max.min(range.max.value());
    }
    EnergyRangeJ::new(min, max)
}

fn subtract_nonnegative_energy(
    lhs: EnergyRangeJ,
    rhs: EnergyRangeJ,
    label: &'static str,
) -> Result<EnergyRangeJ, OntologyError> {
    lhs.validate()?;
    rhs.validate()?;
    let min = lhs.min.value() - rhs.max.value();
    let max = lhs.max.value() - rhs.min.value();
    if !min.is_finite() || !max.is_finite() {
        return Err(OntologyError::InvalidQuantity(label));
    }
    EnergyRangeJ::new(min.max(0.0), max.max(0.0))
}

fn energy_over_duration(
    energy: EnergyRangeJ,
    duration: DurationRangeS,
    label: &'static str,
) -> Result<PowerRangeW, OntologyError> {
    energy.validate()?;
    validate_positive_duration(duration, label)?;
    PowerRangeW::new(
        finite_quotient(energy.min.value(), duration.max.value(), label)?,
        finite_quotient(energy.max.value(), duration.min.value(), label)?,
    )
}

/// Evaluate electrical utility accounting for one explicit process basis.
///
/// Gross/current-cycle screens never consume recovery credit. Lower net energy
/// is reported only under a separately named steady-cycle assumption after
/// charge-power, delivery-fraction and discharge-power limits are applied.
pub fn evaluate_electrical_utility(
    case: &ElectricalUtilityCase,
) -> Result<ElectricalUtilityReport, OntologyError> {
    case.gross_energy_j.validate()?;
    validate_positive_duration(case.batch_duration_s, "batch_duration_s")?;
    case.peak_power_w.validate()?;
    case.recoverable_energy_j.validate()?;
    validate_positive_duration(case.recovery_duration_s, "recovery_duration_s")?;
    case.storage_acceptance_j.validate()?;
    case.storage_charge_power_w.validate()?;
    case.storage_discharge_power_w.validate()?;
    case.recovery_delivery_fraction.validate()?;
    case.available_energy_capacity_j.validate()?;
    case.available_continuous_power_w.validate()?;
    case.available_peak_power_w.validate()?;

    let charge_limited_recovery = power_times_duration(
        case.storage_charge_power_w,
        case.recovery_duration_s,
        "charge_limited_recovery_j",
    )?;
    let accepted_recovery_j = minimum_energy_ranges(
        &[
            case.recoverable_energy_j,
            case.storage_acceptance_j,
            charge_limited_recovery,
            case.gross_energy_j,
        ],
        "accepted_recovery_j",
    )?;

    let delivered_recovery = energy_times_fraction(
        accepted_recovery_j,
        case.recovery_delivery_fraction,
        "delivered_recovery_j",
    )?;
    let discharge_limited_recovery = power_times_duration(
        case.storage_discharge_power_w,
        case.batch_duration_s,
        "discharge_limited_recovery_j",
    )?;
    let usable_steady_cycle_recovery_j = minimum_energy_ranges(
        &[
            delivered_recovery,
            discharge_limited_recovery,
            case.gross_energy_j,
        ],
        "usable_steady_cycle_recovery_j",
    )?;
    let net_steady_cycle_energy_j = subtract_nonnegative_energy(
        case.gross_energy_j,
        usable_steady_cycle_recovery_j,
        "net_steady_cycle_energy_j",
    )?;

    let gross_average_power_w = energy_over_duration(
        case.gross_energy_j,
        case.batch_duration_s,
        "gross_average_power_w",
    )?;
    let net_steady_cycle_average_power_w = energy_over_duration(
        net_steady_cycle_energy_j,
        case.batch_duration_s,
        "net_steady_cycle_average_power_w",
    )?;

    Ok(ElectricalUtilityReport {
        gross_energy_j: case.gross_energy_j,
        accepted_recovery_j,
        usable_steady_cycle_recovery_j,
        net_steady_cycle_energy_j,
        gross_average_power_w,
        net_steady_cycle_average_power_w,
        peak_power_w: case.peak_power_w,
        gross_energy_capacity: energy_feasibility(
            case.gross_energy_j,
            case.available_energy_capacity_j,
        )?,
        steady_cycle_energy_capacity: energy_feasibility(
            net_steady_cycle_energy_j,
            case.available_energy_capacity_j,
        )?,
        gross_continuous_power: power_feasibility(
            gross_average_power_w,
            case.available_continuous_power_w,
        )?,
        steady_cycle_continuous_power: power_feasibility(
            net_steady_cycle_average_power_w,
            case.available_continuous_power_w,
        )?,
        peak_power: power_feasibility(case.peak_power_w, case.available_peak_power_w)?,
    })
}

/// Screen a thermal supply against energy quantity and temperature compatibility.
///
/// This is not an entropy/exergy calculation and does not establish that a real
/// heat exchanger can deliver the declared energy at the declared temperatures.
pub fn evaluate_thermal_utility(
    supply: &ThermalSupply,
    demand: &ThermalDemand,
) -> Result<ThermalUtilityReport, OntologyError> {
    supply.energy_j.validate()?;
    validate_positive_temperature(supply.source_temperature_k, "source_temperature_k")?;
    demand.energy_j.validate()?;
    validate_positive_temperature(
        demand.required_temperature_k,
        "required_temperature_k",
    )?;

    let energy_state = energy_feasibility(demand.energy_j, supply.energy_j)?;
    let temperature_state = if supply.source_temperature_k.min.value()
        >= demand.required_temperature_k.max.value()
    {
        UtilityFeasibility::Guaranteed
    } else if supply.source_temperature_k.max.value()
        < demand.required_temperature_k.min.value()
    {
        UtilityFeasibility::Impossible
    } else {
        UtilityFeasibility::Possible
    };

    let combined = if energy_state == UtilityFeasibility::Impossible
        || temperature_state == UtilityFeasibility::Impossible
    {
        UtilityFeasibility::Impossible
    } else if energy_state == UtilityFeasibility::Guaranteed
        && temperature_state == UtilityFeasibility::Guaranteed
    {
        UtilityFeasibility::Guaranteed
    } else {
        UtilityFeasibility::Possible
    };

    Ok(ThermalUtilityReport {
        energy_feasibility: energy_state,
        temperature_compatibility: temperature_state,
        combined_screening: combined,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_case() -> ElectricalUtilityCase {
        ElectricalUtilityCase {
            gross_energy_j: EnergyRangeJ::new(100.0, 100.0).unwrap(),
            batch_duration_s: DurationRangeS::new(10.0, 10.0).unwrap(),
            peak_power_w: PowerRangeW::new(50.0, 50.0).unwrap(),
            recoverable_energy_j: EnergyRangeJ::new(0.0, 0.0).unwrap(),
            recovery_duration_s: DurationRangeS::new(10.0, 10.0).unwrap(),
            storage_acceptance_j: EnergyRangeJ::new(0.0, 0.0).unwrap(),
            storage_charge_power_w: PowerRangeW::new(0.0, 0.0).unwrap(),
            storage_discharge_power_w: PowerRangeW::new(0.0, 0.0).unwrap(),
            recovery_delivery_fraction: FractionRange::new(1.0, 1.0).unwrap(),
            available_energy_capacity_j: EnergyRangeJ::new(200.0, 200.0).unwrap(),
            available_continuous_power_w: PowerRangeW::new(100.0, 100.0).unwrap(),
            available_peak_power_w: PowerRangeW::new(100.0, 100.0).unwrap(),
        }
    }

    #[test]
    fn duration_changes_gross_average_power() {
        let mut fast = base_case();
        fast.batch_duration_s = DurationRangeS::new(1.0, 1.0).unwrap();
        let fast_report = evaluate_electrical_utility(&fast).unwrap();
        let slow_report = evaluate_electrical_utility(&base_case()).unwrap();
        assert_eq!(
            fast_report.gross_average_power_w,
            PowerRangeW::new(100.0, 100.0).unwrap()
        );
        assert_eq!(
            slow_report.gross_average_power_w,
            PowerRangeW::new(10.0, 10.0).unwrap()
        );
    }

    #[test]
    fn peak_can_fail_while_gross_energy_and_average_power_pass() {
        let mut case = base_case();
        case.peak_power_w = PowerRangeW::new(150.0, 150.0).unwrap();
        case.available_continuous_power_w = PowerRangeW::new(20.0, 20.0).unwrap();
        let report = evaluate_electrical_utility(&case).unwrap();
        assert_eq!(
            report.gross_energy_capacity,
            UtilityFeasibility::Guaranteed
        );
        assert_eq!(
            report.gross_continuous_power,
            UtilityFeasibility::Guaranteed
        );
        assert_eq!(report.peak_power, UtilityFeasibility::Impossible);
    }

    #[test]
    fn recovery_preserves_gross_and_reports_net_steady_cycle_energy() {
        let mut case = base_case();
        case.recoverable_energy_j = EnergyRangeJ::new(30.0, 30.0).unwrap();
        case.storage_acceptance_j = EnergyRangeJ::new(50.0, 50.0).unwrap();
        case.storage_charge_power_w = PowerRangeW::new(10.0, 10.0).unwrap();
        case.storage_discharge_power_w = PowerRangeW::new(10.0, 10.0).unwrap();
        let report = evaluate_electrical_utility(&case).unwrap();
        assert_eq!(
            report.gross_energy_j,
            EnergyRangeJ::new(100.0, 100.0).unwrap()
        );
        assert_eq!(
            report.accepted_recovery_j,
            EnergyRangeJ::new(30.0, 30.0).unwrap()
        );
        assert_eq!(
            report.usable_steady_cycle_recovery_j,
            EnergyRangeJ::new(30.0, 30.0).unwrap()
        );
        assert_eq!(
            report.net_steady_cycle_energy_j,
            EnergyRangeJ::new(70.0, 70.0).unwrap()
        );
    }

    #[test]
    fn charge_power_caps_accepted_recovery() {
        let mut case = base_case();
        case.recoverable_energy_j = EnergyRangeJ::new(80.0, 80.0).unwrap();
        case.recovery_duration_s = DurationRangeS::new(2.0, 2.0).unwrap();
        case.storage_acceptance_j = EnergyRangeJ::new(100.0, 100.0).unwrap();
        case.storage_charge_power_w = PowerRangeW::new(10.0, 10.0).unwrap();
        case.storage_discharge_power_w = PowerRangeW::new(100.0, 100.0).unwrap();
        let report = evaluate_electrical_utility(&case).unwrap();
        assert_eq!(
            report.accepted_recovery_j,
            EnergyRangeJ::new(20.0, 20.0).unwrap()
        );
    }

    #[test]
    fn delivery_fraction_caps_usable_recovery() {
        let mut case = base_case();
        case.recoverable_energy_j = EnergyRangeJ::new(80.0, 80.0).unwrap();
        case.storage_acceptance_j = EnergyRangeJ::new(80.0, 80.0).unwrap();
        case.storage_charge_power_w = PowerRangeW::new(100.0, 100.0).unwrap();
        case.storage_discharge_power_w = PowerRangeW::new(100.0, 100.0).unwrap();
        case.recovery_delivery_fraction = FractionRange::new(0.5, 0.5).unwrap();
        let report = evaluate_electrical_utility(&case).unwrap();
        assert_eq!(
            report.accepted_recovery_j,
            EnergyRangeJ::new(80.0, 80.0).unwrap()
        );
        assert_eq!(
            report.usable_steady_cycle_recovery_j,
            EnergyRangeJ::new(40.0, 40.0).unwrap()
        );
        assert_eq!(
            report.net_steady_cycle_energy_j,
            EnergyRangeJ::new(60.0, 60.0).unwrap()
        );
    }

    #[test]
    fn discharge_power_caps_usable_recovery() {
        let mut case = base_case();
        case.recoverable_energy_j = EnergyRangeJ::new(80.0, 80.0).unwrap();
        case.storage_acceptance_j = EnergyRangeJ::new(80.0, 80.0).unwrap();
        case.storage_charge_power_w = PowerRangeW::new(100.0, 100.0).unwrap();
        case.storage_discharge_power_w = PowerRangeW::new(2.0, 2.0).unwrap();
        let report = evaluate_electrical_utility(&case).unwrap();
        assert_eq!(
            report.usable_steady_cycle_recovery_j,
            EnergyRangeJ::new(20.0, 20.0).unwrap()
        );
        assert_eq!(
            report.net_steady_cycle_energy_j,
            EnergyRangeJ::new(80.0, 80.0).unwrap()
        );
    }

    #[test]
    fn steady_cycle_credit_cannot_rewrite_gross_supply_screen() {
        let mut case = base_case();
        case.recoverable_energy_j = EnergyRangeJ::new(60.0, 60.0).unwrap();
        case.storage_acceptance_j = EnergyRangeJ::new(60.0, 60.0).unwrap();
        case.storage_charge_power_w = PowerRangeW::new(100.0, 100.0).unwrap();
        case.storage_discharge_power_w = PowerRangeW::new(100.0, 100.0).unwrap();
        case.available_energy_capacity_j = EnergyRangeJ::new(50.0, 50.0).unwrap();
        case.available_continuous_power_w = PowerRangeW::new(5.0, 5.0).unwrap();
        let report = evaluate_electrical_utility(&case).unwrap();
        assert_eq!(report.gross_energy_capacity, UtilityFeasibility::Impossible);
        assert_eq!(
            report.steady_cycle_energy_capacity,
            UtilityFeasibility::Guaranteed
        );
        assert_eq!(
            report.gross_continuous_power,
            UtilityFeasibility::Impossible
        );
        assert_eq!(
            report.steady_cycle_continuous_power,
            UtilityFeasibility::Guaranteed
        );
    }

    #[test]
    fn low_temperature_heat_is_incompatible_despite_sufficient_energy() {
        let report = evaluate_thermal_utility(
            &ThermalSupply {
                energy_j: EnergyRangeJ::new(100.0, 100.0).unwrap(),
                source_temperature_k: TemperatureRangeK::new(350.0, 400.0).unwrap(),
            },
            &ThermalDemand {
                energy_j: EnergyRangeJ::new(50.0, 50.0).unwrap(),
                required_temperature_k: TemperatureRangeK::new(500.0, 500.0).unwrap(),
            },
        )
        .unwrap();
        assert_eq!(
            report.energy_feasibility,
            UtilityFeasibility::Guaranteed
        );
        assert_eq!(
            report.temperature_compatibility,
            UtilityFeasibility::Impossible
        );
        assert_eq!(report.combined_screening, UtilityFeasibility::Impossible);
    }

    #[test]
    fn sufficiently_hot_heat_passes_screening() {
        let report = evaluate_thermal_utility(
            &ThermalSupply {
                energy_j: EnergyRangeJ::new(80.0, 100.0).unwrap(),
                source_temperature_k: TemperatureRangeK::new(800.0, 900.0).unwrap(),
            },
            &ThermalDemand {
                energy_j: EnergyRangeJ::new(50.0, 70.0).unwrap(),
                required_temperature_k: TemperatureRangeK::new(600.0, 700.0).unwrap(),
            },
        )
        .unwrap();
        assert_eq!(report.combined_screening, UtilityFeasibility::Guaranteed);
    }

    #[test]
    fn overlapping_thermal_uncertainty_is_only_possible() {
        let report = evaluate_thermal_utility(
            &ThermalSupply {
                energy_j: EnergyRangeJ::new(40.0, 80.0).unwrap(),
                source_temperature_k: TemperatureRangeK::new(650.0, 750.0).unwrap(),
            },
            &ThermalDemand {
                energy_j: EnergyRangeJ::new(50.0, 70.0).unwrap(),
                required_temperature_k: TemperatureRangeK::new(700.0, 800.0).unwrap(),
            },
        )
        .unwrap();
        assert_eq!(report.combined_screening, UtilityFeasibility::Possible);
    }

    #[test]
    fn wider_uncertainty_does_not_strengthen_feasibility() {
        let narrow = energy_feasibility(
            EnergyRangeJ::new(90.0, 90.0).unwrap(),
            EnergyRangeJ::new(100.0, 100.0).unwrap(),
        )
        .unwrap();
        let wide = energy_feasibility(
            EnergyRangeJ::new(80.0, 120.0).unwrap(),
            EnergyRangeJ::new(100.0, 100.0).unwrap(),
        )
        .unwrap();
        assert_eq!(narrow, UtilityFeasibility::Guaranteed);
        assert_eq!(wide, UtilityFeasibility::Possible);
    }

    #[test]
    fn malformed_utility_inputs_fail_closed() {
        let mut zero_batch = base_case();
        zero_batch.batch_duration_s = DurationRangeS::new(0.0, 1.0).unwrap();
        assert!(evaluate_electrical_utility(&zero_batch).is_err());

        let mut zero_recovery_window = base_case();
        zero_recovery_window.recovery_duration_s = DurationRangeS::new(0.0, 1.0).unwrap();
        assert!(evaluate_electrical_utility(&zero_recovery_window).is_err());

        assert!(FractionRange::new(-0.1, 1.0).is_err());
        assert!(FractionRange::new(0.0, 1.1).is_err());
    }

    #[test]
    fn derived_arithmetic_overflow_fails_closed() {
        let mut case = base_case();
        case.storage_charge_power_w = PowerRangeW::new(f64::MAX, f64::MAX).unwrap();
        case.recovery_duration_s = DurationRangeS::new(2.0, 2.0).unwrap();
        assert_eq!(
            evaluate_electrical_utility(&case),
            Err(OntologyError::InvalidQuantity("charge_limited_recovery_j"))
        );
    }
}
