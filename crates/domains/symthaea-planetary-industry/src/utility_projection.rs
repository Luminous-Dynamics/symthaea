// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Fail-closed projection from neutral process utility declarations into a
//! PIE-002 electrical accounting basis.
//!
//! This module mirrors the independent
//! `scripts/pie-process-utility-projection-oracle.py` semantics. Projection is
//! not feasibility: it decides only whether the generic utility declarations
//! provide one unambiguous electrical basis and which thermal/cooling semantics
//! remain unresolved.

use serde::{Deserialize, Serialize};

use crate::{
    DurationRangeS, EnergyRangeJ, OntologyError, PowerRangeW, ProcessDefinition, UtilityDemand,
};

/// Completeness of the electrical utility projection for one process basis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UtilityProjectionStatus {
    /// Electrical energy exists, and peak power and process time are each unique.
    Complete,
    /// One or more required electrical basis fields are absent, with no multiplicity ambiguity.
    Incomplete,
    /// Peak power or process time has multiple declarations with no composition semantics.
    Ambiguous,
}

/// Explicit reason that a generic utility declaration cannot yet support a stronger claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UtilityProjectionReason {
    /// No electrical-energy declaration exists; absence is not interpreted as zero demand.
    MissingElectricalEnergy,
    /// No process-basis peak-power declaration exists.
    MissingPeakPower,
    /// More than one peak-power declaration exists without coincidence/concurrency semantics.
    MultiplePeakPower,
    /// No process-time declaration exists.
    MissingProcessTime,
    /// More than one process-time declaration exists without sequencing semantics.
    MultipleProcessTime,
    /// Thermal energy exists but its required temperature/quality semantics are not bound here.
    ThermalTemperatureUnbound,
    /// Cooling duty exists but sink temperature, rejection path, and timing are not bound here.
    CoolingRejectionUnbound,
}

/// Projection of generic utility declarations onto one auditable process basis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProcessUtilityProjection {
    /// Process identifier whose utility declarations were projected.
    pub process_id: String,
    /// Completeness/ambiguity of the electrical basis only.
    pub electrical_status: UtilityProjectionStatus,
    /// Sum of all declared electrical-energy ranges; `None` means missing, never zero.
    pub electrical_energy_j: Option<EnergyRangeJ>,
    /// Unique peak-power declaration, or `None` when missing/ambiguous.
    pub peak_electrical_power_w: Option<PowerRangeW>,
    /// Unique strictly-positive process-time declaration, or `None` when missing/ambiguous.
    pub process_time_s: Option<DurationRangeS>,
    /// Sum of declared thermal-energy quantities, without temperature-feasibility authority.
    pub thermal_energy_j: Option<EnergyRangeJ>,
    /// Sum of declared cooling-energy quantities, without heat-rejection authority.
    pub cooling_energy_j: Option<EnergyRangeJ>,
    /// Deterministically ordered unresolved semantics.
    pub unresolved: Vec<UtilityProjectionReason>,
}

fn checked_add_energy(
    total_min: &mut f64,
    total_max: &mut f64,
    value: EnergyRangeJ,
    label: &'static str,
) -> Result<(), OntologyError> {
    value.validate()?;
    *total_min += value.min.value();
    *total_max += value.max.value();
    if total_min.is_finite() && total_max.is_finite() {
        Ok(())
    } else {
        Err(OntologyError::InvalidQuantity(label))
    }
}

fn optional_energy(
    count: usize,
    min: f64,
    max: f64,
    label: &'static str,
) -> Result<Option<EnergyRangeJ>, OntologyError> {
    if count == 0 {
        Ok(None)
    } else {
        EnergyRangeJ::new(min, max)
            .map(Some)
            .map_err(|_| OntologyError::InvalidRange(label))
    }
}

/// Project generic process utilities into an unambiguous PIE-002 electrical basis.
///
/// This function intentionally validates only the projection boundary it owns:
/// the process identifier and utility declarations. Call `ProcessDefinition::validate`
/// independently when full process structural validity is required.
///
/// Electrical, thermal, and cooling *energy quantities* are additive. Peak power
/// and process time are not combined without an explicit concurrency/sequencing
/// model; multiplicity therefore yields `Ambiguous`. Missing fields never become
/// zero. Thermal/cooling quantity is retained while stronger quality/rejection
/// semantics remain explicit unresolved reasons.
pub fn project_process_utilities(
    process: &ProcessDefinition,
) -> Result<ProcessUtilityProjection, OntologyError> {
    if process.process_id.trim().is_empty() {
        return Err(OntologyError::EmptyField("process_id"));
    }

    let mut electrical_count = 0usize;
    let mut electrical_min = 0.0;
    let mut electrical_max = 0.0;
    let mut thermal_count = 0usize;
    let mut thermal_min = 0.0;
    let mut thermal_max = 0.0;
    let mut cooling_count = 0usize;
    let mut cooling_min = 0.0;
    let mut cooling_max = 0.0;
    let mut peak_count = 0usize;
    let mut peak_value = None;
    let mut time_count = 0usize;
    let mut time_value = None;

    for utility in &process.utilities {
        utility.validate()?;
        match utility {
            UtilityDemand::ElectricalEnergy(value) => {
                electrical_count += 1;
                checked_add_energy(
                    &mut electrical_min,
                    &mut electrical_max,
                    *value,
                    "utility_projection_electrical_energy_total",
                )?;
            }
            UtilityDemand::ThermalEnergy(value) => {
                thermal_count += 1;
                checked_add_energy(
                    &mut thermal_min,
                    &mut thermal_max,
                    *value,
                    "utility_projection_thermal_energy_total",
                )?;
            }
            UtilityDemand::CoolingEnergy(value) => {
                cooling_count += 1;
                checked_add_energy(
                    &mut cooling_min,
                    &mut cooling_max,
                    *value,
                    "utility_projection_cooling_energy_total",
                )?;
            }
            UtilityDemand::PeakElectricalPower(value) => {
                peak_count += 1;
                if peak_count == 1 {
                    peak_value = Some(*value);
                }
            }
            UtilityDemand::ProcessTime(value) => {
                if value.min.value() <= 0.0 {
                    return Err(OntologyError::InvalidRange(
                        "utility_projection_process_time_s",
                    ));
                }
                time_count += 1;
                if time_count == 1 {
                    time_value = Some(*value);
                }
            }
        }
    }

    let electrical_energy_j = optional_energy(
        electrical_count,
        electrical_min,
        electrical_max,
        "utility_projection_electrical_energy_total",
    )?;
    let thermal_energy_j = optional_energy(
        thermal_count,
        thermal_min,
        thermal_max,
        "utility_projection_thermal_energy_total",
    )?;
    let cooling_energy_j = optional_energy(
        cooling_count,
        cooling_min,
        cooling_max,
        "utility_projection_cooling_energy_total",
    )?;

    let mut unresolved = Vec::new();
    if electrical_count == 0 {
        unresolved.push(UtilityProjectionReason::MissingElectricalEnergy);
    }
    match peak_count {
        0 => unresolved.push(UtilityProjectionReason::MissingPeakPower),
        1 => {}
        _ => unresolved.push(UtilityProjectionReason::MultiplePeakPower),
    }
    match time_count {
        0 => unresolved.push(UtilityProjectionReason::MissingProcessTime),
        1 => {}
        _ => unresolved.push(UtilityProjectionReason::MultipleProcessTime),
    }
    if thermal_count > 0 {
        unresolved.push(UtilityProjectionReason::ThermalTemperatureUnbound);
    }
    if cooling_count > 0 {
        unresolved.push(UtilityProjectionReason::CoolingRejectionUnbound);
    }

    let electrical_status = if peak_count > 1 || time_count > 1 {
        UtilityProjectionStatus::Ambiguous
    } else if electrical_count == 0 || peak_count != 1 || time_count != 1 {
        UtilityProjectionStatus::Incomplete
    } else {
        UtilityProjectionStatus::Complete
    };

    Ok(ProcessUtilityProjection {
        process_id: process.process_id.clone(),
        electrical_status,
        electrical_energy_j,
        peak_electrical_power_w: (peak_count == 1).then_some(peak_value).flatten(),
        process_time_s: (time_count == 1).then_some(time_value).flatten(),
        thermal_energy_j,
        cooling_energy_j,
        unresolved,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn process(utilities: Vec<UtilityDemand>) -> ProcessDefinition {
        ProcessDefinition {
            process_id: "p1".into(),
            name: String::new(),
            inputs: vec![],
            outputs: vec![],
            utilities,
            equipment: vec![],
            environment: vec![],
            evidence: vec![],
        }
    }

    fn baseline() -> Vec<UtilityDemand> {
        vec![
            UtilityDemand::ElectricalEnergy(EnergyRangeJ::new(40.0, 50.0).unwrap()),
            UtilityDemand::ElectricalEnergy(EnergyRangeJ::new(10.0, 20.0).unwrap()),
            UtilityDemand::PeakElectricalPower(PowerRangeW::new(15.0, 20.0).unwrap()),
            UtilityDemand::ProcessTime(DurationRangeS::new(5.0, 10.0).unwrap()),
        ]
    }

    #[test]
    fn additive_energy_and_unique_basis_are_complete() {
        let report = project_process_utilities(&process(baseline())).unwrap();
        assert_eq!(report.electrical_status, UtilityProjectionStatus::Complete);
        assert_eq!(report.electrical_energy_j, Some(EnergyRangeJ::new(50.0, 70.0).unwrap()));
        assert_eq!(report.peak_electrical_power_w, Some(PowerRangeW::new(15.0, 20.0).unwrap()));
        assert_eq!(report.process_time_s, Some(DurationRangeS::new(5.0, 10.0).unwrap()));
        assert!(report.unresolved.is_empty());
    }

    #[test]
    fn missing_process_time_is_incomplete_not_zero() {
        let mut utilities = baseline();
        utilities.pop();
        let report = project_process_utilities(&process(utilities)).unwrap();
        assert_eq!(report.electrical_status, UtilityProjectionStatus::Incomplete);
        assert_eq!(report.process_time_s, None);
        assert!(report.unresolved.contains(&UtilityProjectionReason::MissingProcessTime));
    }

    #[test]
    fn duplicate_peak_power_is_ambiguous_not_summed_or_maxed() {
        let mut utilities = baseline();
        utilities.push(UtilityDemand::PeakElectricalPower(PowerRangeW::new(1.0, 1.0).unwrap()));
        let report = project_process_utilities(&process(utilities)).unwrap();
        assert_eq!(report.electrical_status, UtilityProjectionStatus::Ambiguous);
        assert_eq!(report.peak_electrical_power_w, None);
        assert!(report.unresolved.contains(&UtilityProjectionReason::MultiplePeakPower));
    }

    #[test]
    fn duplicate_process_time_is_ambiguous_not_combined() {
        let mut utilities = baseline();
        utilities.push(UtilityDemand::ProcessTime(DurationRangeS::new(1.0, 1.0).unwrap()));
        let report = project_process_utilities(&process(utilities)).unwrap();
        assert_eq!(report.electrical_status, UtilityProjectionStatus::Ambiguous);
        assert_eq!(report.process_time_s, None);
        assert!(report.unresolved.contains(&UtilityProjectionReason::MultipleProcessTime));
    }

    #[test]
    fn thermal_and_cooling_quantities_remain_visible_but_unresolved() {
        let mut utilities = baseline();
        utilities.push(UtilityDemand::ThermalEnergy(EnergyRangeJ::new(20.0, 25.0).unwrap()));
        utilities.push(UtilityDemand::ThermalEnergy(EnergyRangeJ::new(5.0, 5.0).unwrap()));
        utilities.push(UtilityDemand::CoolingEnergy(EnergyRangeJ::new(5.0, 8.0).unwrap()));
        let report = project_process_utilities(&process(utilities)).unwrap();
        assert_eq!(report.electrical_status, UtilityProjectionStatus::Complete);
        assert_eq!(report.thermal_energy_j, Some(EnergyRangeJ::new(25.0, 30.0).unwrap()));
        assert_eq!(report.cooling_energy_j, Some(EnergyRangeJ::new(5.0, 8.0).unwrap()));
        assert!(report.unresolved.contains(&UtilityProjectionReason::ThermalTemperatureUnbound));
        assert!(report.unresolved.contains(&UtilityProjectionReason::CoolingRejectionUnbound));
    }

    #[test]
    fn missing_electrical_energy_is_incomplete() {
        let report = project_process_utilities(&process(vec![
            UtilityDemand::PeakElectricalPower(PowerRangeW::new(15.0, 15.0).unwrap()),
            UtilityDemand::ProcessTime(DurationRangeS::new(5.0, 5.0).unwrap()),
        ]))
        .unwrap();
        assert_eq!(report.electrical_status, UtilityProjectionStatus::Incomplete);
        assert_eq!(report.electrical_energy_j, None);
        assert!(report.unresolved.contains(&UtilityProjectionReason::MissingElectricalEnergy));
    }

    #[test]
    fn missing_peak_power_is_incomplete() {
        let report = project_process_utilities(&process(vec![
            UtilityDemand::ElectricalEnergy(EnergyRangeJ::new(10.0, 10.0).unwrap()),
            UtilityDemand::ProcessTime(DurationRangeS::new(5.0, 5.0).unwrap()),
        ]))
        .unwrap();
        assert_eq!(report.electrical_status, UtilityProjectionStatus::Incomplete);
        assert!(report.unresolved.contains(&UtilityProjectionReason::MissingPeakPower));
    }

    #[test]
    fn aggregate_energy_overflow_fails_closed() {
        let report = project_process_utilities(&process(vec![
            UtilityDemand::ElectricalEnergy(EnergyRangeJ::new(f64::MAX, f64::MAX).unwrap()),
            UtilityDemand::ElectricalEnergy(EnergyRangeJ::new(f64::MAX, f64::MAX).unwrap()),
            UtilityDemand::PeakElectricalPower(PowerRangeW::new(1.0, 1.0).unwrap()),
            UtilityDemand::ProcessTime(DurationRangeS::new(1.0, 1.0).unwrap()),
        ]));
        assert_eq!(
            report,
            Err(OntologyError::InvalidQuantity(
                "utility_projection_electrical_energy_total"
            ))
        );
    }

    #[test]
    fn zero_inclusive_process_time_fails_closed() {
        let report = project_process_utilities(&process(vec![
            UtilityDemand::ElectricalEnergy(EnergyRangeJ::new(10.0, 10.0).unwrap()),
            UtilityDemand::PeakElectricalPower(PowerRangeW::new(1.0, 1.0).unwrap()),
            UtilityDemand::ProcessTime(DurationRangeS::new(0.0, 1.0).unwrap()),
        ]));
        assert_eq!(
            report,
            Err(OntologyError::InvalidRange(
                "utility_projection_process_time_s"
            ))
        );
    }

    #[test]
    fn blank_process_id_fails_closed() {
        let mut fixture = process(baseline());
        fixture.process_id = "   ".into();
        assert_eq!(
            project_process_utilities(&fixture),
            Err(OntologyError::EmptyField("process_id"))
        );
    }
}
