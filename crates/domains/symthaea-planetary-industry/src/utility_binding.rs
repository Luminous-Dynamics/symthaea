// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! PIE-002D fail-closed binding from a complete process-utility projection to
//! one explicit electrical utility-accounting case.
//!
//! This module mirrors the independently qualified
//! `scripts/pie-utility-case-binding-oracle.py` contract. Binding is not
//! feasibility: it validates that the projected demand basis is internally
//! consistent, validates every externally supplied recovery/storage/supply
//! fact, preserves process identity and unresolved non-electrical semantics,
//! and constructs an accounting case without inventing defaults.

use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

use crate::{
    DurationRangeS, ElectricalUtilityCase, EnergyRangeJ, FractionRange, OntologyError, PowerRangeW,
    ProcessUtilityProjection, UtilityProjectionReason, UtilityProjectionStatus,
};

/// Failures specific to projection-to-accounting binding.
#[derive(Debug, Clone, PartialEq)]
pub enum UtilityBindingError {
    /// A numeric range, identifier, or other ontology value is malformed.
    Ontology(OntologyError),
    /// Only a projection explicitly classified as `Complete` may bind.
    ProjectionNotComplete(UtilityProjectionStatus),
    /// A purported complete projection omits one required electrical field.
    MissingProjectionField(&'static str),
    /// A purported complete projection still carries an electrical blocking reason.
    CompleteProjectionHasBlockingReason(UtilityProjectionReason),
    /// The unresolved-reason set contains the same semantic reason more than once.
    DuplicateUnresolvedReason(UtilityProjectionReason),
}

impl fmt::Display for UtilityBindingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ontology(error) => write!(f, "{error}"),
            Self::ProjectionNotComplete(status) => {
                write!(
                    f,
                    "electrical utility projection is not complete: {status:?}"
                )
            }
            Self::MissingProjectionField(field) => {
                write!(
                    f,
                    "complete electrical utility projection is missing: {field}"
                )
            }
            Self::CompleteProjectionHasBlockingReason(reason) => write!(
                f,
                "complete electrical utility projection carries blocking reason: {reason:?}"
            ),
            Self::DuplicateUnresolvedReason(reason) => {
                write!(
                    f,
                    "duplicate unresolved utility projection reason: {reason:?}"
                )
            }
        }
    }
}

impl Error for UtilityBindingError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Ontology(error) => Some(error),
            _ => None,
        }
    }
}

impl From<OntologyError> for UtilityBindingError {
    fn from(value: OntologyError) -> Self {
        Self::Ontology(value)
    }
}

/// Explicit external recovery, storage, and electrical-supply facts required
/// to bind one complete process demand projection into an accounting case.
///
/// No field has an implicit zero, infinity, nominal-capacity, or guessed value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElectricalSupplyRecoveryContext {
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
    pub available_sustained_power_w: PowerRangeW,
    /// Short-duration/peak source or buffer power available to the process.
    pub available_peak_power_w: PowerRangeW,
}

impl ElectricalSupplyRecoveryContext {
    fn validate(&self) -> Result<(), UtilityBindingError> {
        self.recoverable_energy_j.validate()?;
        validate_positive_duration(self.recovery_duration_s, "recovery_duration_s")?;
        self.storage_acceptance_j.validate()?;
        self.storage_charge_power_w.validate()?;
        self.storage_discharge_power_w.validate()?;

        // Reconstruct through the validating constructor so even a value created
        // through deserialization cannot bypass the fraction invariant.
        FractionRange::new(
            self.recovery_delivery_fraction.min(),
            self.recovery_delivery_fraction.max(),
        )?;

        self.available_energy_capacity_j.validate()?;
        self.available_sustained_power_w.validate()?;
        self.available_peak_power_w.validate()?;
        Ok(())
    }
}

/// Bound but unevaluated electrical accounting case with the originating
/// process identity and any unresolved non-electrical projection semantics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundElectricalAccountingCase {
    /// Process identifier preserved exactly from the demand projection.
    pub process_id: String,
    /// Gross electrical energy required by the projected process basis.
    pub gross_energy_j: EnergyRangeJ,
    /// Projected process duration; strictly positive.
    pub batch_duration_s: DurationRangeS,
    /// Projected gross peak electrical power.
    pub peak_power_w: PowerRangeW,
    /// Explicit recoverable electrical energy from the external context.
    pub recoverable_energy_j: EnergyRangeJ,
    /// Explicit recovery-window duration from the external context.
    pub recovery_duration_s: DurationRangeS,
    /// Explicit storage energy acceptance from the external context.
    pub storage_acceptance_j: EnergyRangeJ,
    /// Explicit storage charge-power capability from the external context.
    pub storage_charge_power_w: PowerRangeW,
    /// Explicit storage discharge-power capability from the external context.
    pub storage_discharge_power_w: PowerRangeW,
    /// Explicit recovery delivery / round-trip fraction from the external context.
    pub recovery_delivery_fraction: FractionRange,
    /// Explicit available electrical-energy capacity from the external context.
    pub available_energy_capacity_j: EnergyRangeJ,
    /// Explicit available sustained-power capacity from the external context.
    pub available_sustained_power_w: PowerRangeW,
    /// Explicit available peak-power capacity from the external context.
    pub available_peak_power_w: PowerRangeW,
    /// Non-electrical unresolved semantics carried through without promotion.
    pub unresolved_non_electrical: Vec<UtilityProjectionReason>,
}

impl BoundElectricalAccountingCase {
    /// Materialize the already-bound numerical fields as the PIE-002 accounting
    /// kernel input without evaluating feasibility.
    pub fn electrical_utility_case(&self) -> ElectricalUtilityCase {
        ElectricalUtilityCase {
            gross_energy_j: self.gross_energy_j,
            batch_duration_s: self.batch_duration_s,
            peak_power_w: self.peak_power_w,
            recoverable_energy_j: self.recoverable_energy_j,
            recovery_duration_s: self.recovery_duration_s,
            storage_acceptance_j: self.storage_acceptance_j,
            storage_charge_power_w: self.storage_charge_power_w,
            storage_discharge_power_w: self.storage_discharge_power_w,
            recovery_delivery_fraction: self.recovery_delivery_fraction,
            available_energy_capacity_j: self.available_energy_capacity_j,
            available_continuous_power_w: self.available_sustained_power_w,
            available_peak_power_w: self.available_peak_power_w,
        }
    }
}

fn validate_positive_duration(
    duration: DurationRangeS,
    label: &'static str,
) -> Result<(), UtilityBindingError> {
    duration.validate()?;
    if duration.min.value() > 0.0 {
        Ok(())
    } else {
        Err(OntologyError::InvalidRange(label).into())
    }
}

fn is_electrical_blocking_reason(reason: UtilityProjectionReason) -> bool {
    matches!(
        reason,
        UtilityProjectionReason::MissingElectricalEnergy
            | UtilityProjectionReason::MissingPeakPower
            | UtilityProjectionReason::MultiplePeakPower
            | UtilityProjectionReason::MissingProcessTime
            | UtilityProjectionReason::MultipleProcessTime
    )
}

fn validate_complete_projection(
    projection: &ProcessUtilityProjection,
) -> Result<(EnergyRangeJ, PowerRangeW, DurationRangeS), UtilityBindingError> {
    if projection.process_id.trim().is_empty() {
        return Err(OntologyError::EmptyField("process_id").into());
    }
    if projection.electrical_status != UtilityProjectionStatus::Complete {
        return Err(UtilityBindingError::ProjectionNotComplete(
            projection.electrical_status,
        ));
    }

    let electrical_energy_j =
        projection
            .electrical_energy_j
            .ok_or(UtilityBindingError::MissingProjectionField(
                "electrical_energy_j",
            ))?;
    let peak_power_w =
        projection
            .peak_electrical_power_w
            .ok_or(UtilityBindingError::MissingProjectionField(
                "peak_electrical_power_w",
            ))?;
    let process_time_s =
        projection
            .process_time_s
            .ok_or(UtilityBindingError::MissingProjectionField(
                "process_time_s",
            ))?;

    electrical_energy_j.validate()?;
    peak_power_w.validate()?;
    validate_positive_duration(process_time_s, "utility_binding_process_time_s")?;

    for (index, reason) in projection.unresolved.iter().copied().enumerate() {
        if projection.unresolved[..index].contains(&reason) {
            return Err(UtilityBindingError::DuplicateUnresolvedReason(reason));
        }
        if is_electrical_blocking_reason(reason) {
            return Err(UtilityBindingError::CompleteProjectionHasBlockingReason(
                reason,
            ));
        }
    }

    Ok((electrical_energy_j, peak_power_w, process_time_s))
}

/// Bind a complete PIE-002B process utility projection to explicit external
/// supply/recovery facts without evaluating feasibility or manufacturing defaults.
///
/// Success proves only that one internally consistent electrical demand basis and
/// one fully explicit external context have been joined losslessly. Thermal and
/// cooling unresolved reasons remain visible and do not become electrical facts.
pub fn bind_electrical_accounting_case(
    projection: &ProcessUtilityProjection,
    context: &ElectricalSupplyRecoveryContext,
) -> Result<BoundElectricalAccountingCase, UtilityBindingError> {
    let (gross_energy_j, peak_power_w, batch_duration_s) =
        validate_complete_projection(projection)?;
    context.validate()?;

    Ok(BoundElectricalAccountingCase {
        process_id: projection.process_id.clone(),
        gross_energy_j,
        batch_duration_s,
        peak_power_w,
        recoverable_energy_j: context.recoverable_energy_j,
        recovery_duration_s: context.recovery_duration_s,
        storage_acceptance_j: context.storage_acceptance_j,
        storage_charge_power_w: context.storage_charge_power_w,
        storage_discharge_power_w: context.storage_discharge_power_w,
        recovery_delivery_fraction: context.recovery_delivery_fraction,
        available_energy_capacity_j: context.available_energy_capacity_j,
        available_sustained_power_w: context.available_sustained_power_w,
        available_peak_power_w: context.available_peak_power_w,
        unresolved_non_electrical: projection.unresolved.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn baseline_projection() -> ProcessUtilityProjection {
        ProcessUtilityProjection {
            process_id: "p1".into(),
            electrical_status: UtilityProjectionStatus::Complete,
            electrical_energy_j: Some(EnergyRangeJ::new(90.0, 110.0).unwrap()),
            peak_electrical_power_w: Some(PowerRangeW::new(20.0, 25.0).unwrap()),
            process_time_s: Some(DurationRangeS::new(9.0, 11.0).unwrap()),
            thermal_energy_j: None,
            cooling_energy_j: None,
            unresolved: vec![],
        }
    }

    fn baseline_context() -> ElectricalSupplyRecoveryContext {
        ElectricalSupplyRecoveryContext {
            recoverable_energy_j: EnergyRangeJ::new(30.0, 40.0).unwrap(),
            recovery_duration_s: DurationRangeS::new(4.0, 5.0).unwrap(),
            storage_acceptance_j: EnergyRangeJ::new(50.0, 60.0).unwrap(),
            storage_charge_power_w: PowerRangeW::new(10.0, 12.0).unwrap(),
            storage_discharge_power_w: PowerRangeW::new(8.0, 9.0).unwrap(),
            recovery_delivery_fraction: FractionRange::new(0.8, 0.9).unwrap(),
            available_energy_capacity_j: EnergyRangeJ::new(120.0, 140.0).unwrap(),
            available_sustained_power_w: PowerRangeW::new(15.0, 20.0).unwrap(),
            available_peak_power_w: PowerRangeW::new(30.0, 35.0).unwrap(),
        }
    }

    #[test]
    fn exact_demand_context_and_identity_are_preserved() {
        let projection = baseline_projection();
        let context = baseline_context();
        let bound = bind_electrical_accounting_case(&projection, &context).unwrap();

        assert_eq!(bound.process_id, projection.process_id);
        assert_eq!(
            bound.gross_energy_j,
            projection.electrical_energy_j.unwrap()
        );
        assert_eq!(bound.batch_duration_s, projection.process_time_s.unwrap());
        assert_eq!(
            bound.peak_power_w,
            projection.peak_electrical_power_w.unwrap()
        );
        assert_eq!(bound.recoverable_energy_j, context.recoverable_energy_j);
        assert_eq!(
            bound.recovery_delivery_fraction,
            context.recovery_delivery_fraction
        );
        assert_eq!(
            bound.available_sustained_power_w,
            context.available_sustained_power_w
        );

        let case = bound.electrical_utility_case();
        assert_eq!(case.gross_energy_j, bound.gross_energy_j);
        assert_eq!(case.batch_duration_s, bound.batch_duration_s);
        assert_eq!(
            case.available_continuous_power_w,
            bound.available_sustained_power_w
        );
    }

    #[test]
    fn incomplete_and_ambiguous_projections_do_not_bind() {
        for status in [
            UtilityProjectionStatus::Incomplete,
            UtilityProjectionStatus::Ambiguous,
        ] {
            let mut projection = baseline_projection();
            projection.electrical_status = status;
            assert_eq!(
                bind_electrical_accounting_case(&projection, &baseline_context()),
                Err(UtilityBindingError::ProjectionNotComplete(status))
            );
        }
    }

    #[test]
    fn forged_complete_projection_missing_peak_fails_closed() {
        let mut projection = baseline_projection();
        projection.peak_electrical_power_w = None;
        assert_eq!(
            bind_electrical_accounting_case(&projection, &baseline_context()),
            Err(UtilityBindingError::MissingProjectionField(
                "peak_electrical_power_w"
            ))
        );
    }

    #[test]
    fn forged_complete_projection_with_blocking_reason_fails_closed() {
        let mut projection = baseline_projection();
        projection
            .unresolved
            .push(UtilityProjectionReason::MultiplePeakPower);
        assert_eq!(
            bind_electrical_accounting_case(&projection, &baseline_context()),
            Err(UtilityBindingError::CompleteProjectionHasBlockingReason(
                UtilityProjectionReason::MultiplePeakPower
            ))
        );
    }

    #[test]
    fn unresolved_thermal_and_cooling_reasons_survive_without_promotion() {
        let mut projection = baseline_projection();
        projection.unresolved = vec![
            UtilityProjectionReason::ThermalTemperatureUnbound,
            UtilityProjectionReason::CoolingRejectionUnbound,
        ];
        let bound = bind_electrical_accounting_case(&projection, &baseline_context()).unwrap();
        assert_eq!(bound.unresolved_non_electrical, projection.unresolved);
    }

    #[test]
    fn duplicate_unresolved_reason_fails_closed() {
        let mut projection = baseline_projection();
        projection.unresolved = vec![
            UtilityProjectionReason::ThermalTemperatureUnbound,
            UtilityProjectionReason::ThermalTemperatureUnbound,
        ];
        assert_eq!(
            bind_electrical_accounting_case(&projection, &baseline_context()),
            Err(UtilityBindingError::DuplicateUnresolvedReason(
                UtilityProjectionReason::ThermalTemperatureUnbound
            ))
        );
    }

    #[test]
    fn zero_inclusive_projected_process_time_fails_closed() {
        let mut projection = baseline_projection();
        projection.process_time_s = Some(DurationRangeS::new(0.0, 10.0).unwrap());
        assert_eq!(
            bind_electrical_accounting_case(&projection, &baseline_context()),
            Err(UtilityBindingError::Ontology(OntologyError::InvalidRange(
                "utility_binding_process_time_s"
            )))
        );
    }

    #[test]
    fn zero_inclusive_recovery_window_fails_closed() {
        let projection = baseline_projection();
        let mut context = baseline_context();
        context.recovery_duration_s = DurationRangeS::new(0.0, 1.0).unwrap();
        assert_eq!(
            bind_electrical_accounting_case(&projection, &context),
            Err(UtilityBindingError::Ontology(OntologyError::InvalidRange(
                "recovery_duration_s"
            )))
        );
    }

    #[test]
    fn blank_process_identity_fails_closed() {
        let mut projection = baseline_projection();
        projection.process_id = "   ".into();
        assert_eq!(
            bind_electrical_accounting_case(&projection, &baseline_context()),
            Err(UtilityBindingError::Ontology(OntologyError::EmptyField(
                "process_id"
            )))
        );
    }
}
