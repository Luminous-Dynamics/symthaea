// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! PIE-002E invocation-subject-bound process-to-accounting composition plus
//! PIE-002G construction integrity for the preferred bound result.
//!
//! The preferred composition path accepts the process record itself rather than
//! a detached utility projection. It validates the full `ProcessDefinition`,
//! recomputes PIE-002B utility projection inside the call, and immediately binds
//! that derived projection through PIE-002D to an explicit supply/recovery
//! context. PIE-002G then seals that validated result behind an opaque Rust type
//! whose private state cannot be reconstructed through ordinary deserialization.
//! Composition is not feasibility evaluation and does not establish that the
//! caller supplied the newest authoritative process revision in storage.

use serde::Serialize;
use std::error::Error;
use std::fmt;

use crate::{
    BoundElectricalAccountingCase, DurationRangeS, ElectricalSupplyRecoveryContext,
    ElectricalUtilityCase, EnergyRangeJ, FractionRange, OntologyError, PowerRangeW,
    ProcessDefinition, UtilityBindingError, UtilityProjectionReason,
    bind_electrical_accounting_case, project_process_utilities,
};

/// Failures at the PIE-002E process-to-accounting composition boundary.
#[derive(Debug, Clone, PartialEq)]
pub enum UtilityCompositionError {
    /// Full process structural or ontology validation failed before binding.
    Ontology(OntologyError),
    /// The freshly derived utility projection could not bind to the explicit context.
    Binding(UtilityBindingError),
}

impl fmt::Display for UtilityCompositionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ontology(error) => write!(f, "process composition ontology failure: {error}"),
            Self::Binding(error) => write!(f, "process composition binding failure: {error}"),
        }
    }
}

impl Error for UtilityCompositionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Ontology(error) => Some(error),
            Self::Binding(error) => Some(error),
        }
    }
}

impl From<OntologyError> for UtilityCompositionError {
    fn from(value: OntologyError) -> Self {
        Self::Ontology(value)
    }
}

impl From<UtilityBindingError> for UtilityCompositionError {
    fn from(value: UtilityBindingError) -> Self {
        Self::Binding(value)
    }
}

/// Opaque proof-carrying result of the preferred PIE-002E composition path.
///
/// A value of this type can be obtained publicly only by successfully calling
/// [`compose_subject_bound_electrical_accounting_case`]. Its state is private,
/// there is no public unchecked constructor, and the type intentionally does
/// **not** implement `Deserialize`. Serialization is one-way and intended only
/// for evidence/reporting; serialized bytes are not a trusted restoration path.
///
/// The witness certifies only invocation-subject validation, fresh utility
/// projection, and PIE-002D numerical binding. It does not certify provenance,
/// evidence applicability, persistence currentness, feasibility, dispatch,
/// thermodynamic closure, economics, or execution authority.
///
/// External code cannot inspect or mutate the private lower-level DTO directly:
///
/// ```compile_fail
/// use symthaea_planetary_industry::SubjectBoundElectricalAccountingCase;
///
/// fn bypass(witness: SubjectBoundElectricalAccountingCase) {
///     let _ = &witness.bound;
/// }
/// ```
///
/// The witness is deliberately not a generally deserializable DTO:
///
/// ```compile_fail
/// use serde::de::DeserializeOwned;
/// use symthaea_planetary_industry::SubjectBoundElectricalAccountingCase;
///
/// fn require_deserialize<T: DeserializeOwned>() {}
///
/// fn probe() {
///     require_deserialize::<SubjectBoundElectricalAccountingCase>();
/// }
/// ```
#[must_use = "a validated subject-bound utility witness should be retained or explicitly consumed"]
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(transparent)]
pub struct SubjectBoundElectricalAccountingCase {
    bound: BoundElectricalAccountingCase,
}

impl SubjectBoundElectricalAccountingCase {
    /// Process identifier preserved from the freshly derived utility projection.
    pub fn process_id(&self) -> &str {
        &self.bound.process_id
    }

    /// Gross electrical energy demanded by the composed process basis.
    pub fn gross_energy_j(&self) -> EnergyRangeJ {
        self.bound.gross_energy_j
    }

    /// Process duration used by the bound accounting basis.
    pub fn batch_duration_s(&self) -> DurationRangeS {
        self.bound.batch_duration_s
    }

    /// Gross peak electrical power demanded by the process basis.
    pub fn peak_power_w(&self) -> PowerRangeW {
        self.bound.peak_power_w
    }

    /// Explicit recoverable electrical energy from the supplied context.
    pub fn recoverable_energy_j(&self) -> EnergyRangeJ {
        self.bound.recoverable_energy_j
    }

    /// Explicit recovery-window duration from the supplied context.
    pub fn recovery_duration_s(&self) -> DurationRangeS {
        self.bound.recovery_duration_s
    }

    /// Explicit storage energy acceptance from the supplied context.
    pub fn storage_acceptance_j(&self) -> EnergyRangeJ {
        self.bound.storage_acceptance_j
    }

    /// Explicit storage charge-power capability from the supplied context.
    pub fn storage_charge_power_w(&self) -> PowerRangeW {
        self.bound.storage_charge_power_w
    }

    /// Explicit storage discharge-power capability from the supplied context.
    pub fn storage_discharge_power_w(&self) -> PowerRangeW {
        self.bound.storage_discharge_power_w
    }

    /// Explicit recovery delivery / round-trip fraction from the supplied context.
    pub fn recovery_delivery_fraction(&self) -> FractionRange {
        self.bound.recovery_delivery_fraction
    }

    /// Explicit available electrical-energy capacity from the supplied context.
    pub fn available_energy_capacity_j(&self) -> EnergyRangeJ {
        self.bound.available_energy_capacity_j
    }

    /// Explicit available sustained-power capacity from the supplied context.
    pub fn available_sustained_power_w(&self) -> PowerRangeW {
        self.bound.available_sustained_power_w
    }

    /// Explicit available peak-power capacity from the supplied context.
    pub fn available_peak_power_w(&self) -> PowerRangeW {
        self.bound.available_peak_power_w
    }

    /// Non-electrical unresolved projection semantics preserved without promotion.
    pub fn unresolved_non_electrical(&self) -> &[UtilityProjectionReason] {
        &self.bound.unresolved_non_electrical
    }

    /// Materialize the lower-level PIE-002 numerical accounting input.
    ///
    /// The returned DTO is useful for explicit expert/testing evaluation, but an
    /// arbitrary `ElectricalUtilityCase` can never be converted back into this
    /// authoritative witness through a public API.
    pub fn electrical_utility_case(&self) -> ElectricalUtilityCase {
        self.bound.electrical_utility_case()
    }
}

/// Validate one exact process value, recompute its utility projection in-call,
/// bind it to explicit recovery/storage/supply facts, and return an opaque
/// construction-integrity witness without evaluating feasibility.
///
/// This API deliberately has no `ProcessUtilityProjection` parameter. A caller
/// therefore cannot make the preferred path consume a detached/cached projection
/// after mutating or replacing the process record. Success establishes only that
/// the exact `ProcessDefinition` value supplied to this invocation was fully
/// validated, freshly projected, and bound. It does not establish persistence
/// currentness, source provenance/applicability, feasibility, dispatch, or action
/// authority.
pub fn compose_subject_bound_electrical_accounting_case(
    process: &ProcessDefinition,
    context: &ElectricalSupplyRecoveryContext,
) -> Result<SubjectBoundElectricalAccountingCase, UtilityCompositionError> {
    process.validate()?;
    let projection = project_process_utilities(process)?;
    let bound = bind_electrical_accounting_case(&projection, context)?;
    Ok(SubjectBoundElectricalAccountingCase { bound })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DurationRangeS, EnergyRangeJ, FractionRange, MassRangeKg, MaterialGrade, OutputDisposition,
        PhysicalForm, PowerRangeW, ProcessInput, ProcessInputRole, ProcessOutput,
        ProcessOutputRole, UtilityDemand, UtilityProjectionReason, UtilityProjectionStatus,
    };

    fn grade(label: &str) -> MaterialGrade {
        MaterialGrade {
            label: label.into(),
            specification_ref: None,
        }
    }

    fn baseline_process() -> ProcessDefinition {
        ProcessDefinition {
            process_id: "p1".into(),
            name: "Synthetic electrical process".into(),
            inputs: vec![ProcessInput {
                material_key: "feed".into(),
                required_grade: None,
                role: ProcessInputRole::Feedstock,
                mass_kg: MassRangeKg::new(9.0, 11.0).unwrap(),
            }],
            outputs: vec![ProcessOutput {
                material_key: "product".into(),
                grade: grade("synthetic"),
                form: PhysicalForm::Bulk,
                role: ProcessOutputRole::Product,
                mass_kg: MassRangeKg::new(8.0, 10.0).unwrap(),
                disposition: OutputDisposition::Inventory,
            }],
            utilities: vec![
                UtilityDemand::ElectricalEnergy(EnergyRangeJ::new(90.0, 110.0).unwrap()),
                UtilityDemand::PeakElectricalPower(PowerRangeW::new(20.0, 25.0).unwrap()),
                UtilityDemand::ProcessTime(DurationRangeS::new(9.0, 11.0).unwrap()),
            ],
            equipment: vec![],
            environment: vec![],
            evidence: vec![],
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
    fn valid_process_is_validated_projected_bound_and_sealed_losslessly() {
        let process = baseline_process();
        let context = baseline_context();
        let bound = compose_subject_bound_electrical_accounting_case(&process, &context).unwrap();

        assert_eq!(bound.process_id(), process.process_id.as_str());
        assert_eq!(
            bound.gross_energy_j(),
            EnergyRangeJ::new(90.0, 110.0).unwrap()
        );
        assert_eq!(bound.peak_power_w(), PowerRangeW::new(20.0, 25.0).unwrap());
        assert_eq!(
            bound.batch_duration_s(),
            DurationRangeS::new(9.0, 11.0).unwrap()
        );
        assert_eq!(bound.recoverable_energy_j(), context.recoverable_energy_j);
        assert_eq!(
            bound.available_sustained_power_w(),
            context.available_sustained_power_w
        );
    }

    #[test]
    fn witness_is_one_way_serializable_at_the_type_level() {
        fn require_serialize<T: serde::Serialize>() {}
        require_serialize::<SubjectBoundElectricalAccountingCase>();
    }

    #[test]
    fn full_process_structural_invalidity_fails_before_binding() {
        let mut process = baseline_process();
        process.inputs.clear();
        assert_eq!(
            compose_subject_bound_electrical_accounting_case(&process, &baseline_context()),
            Err(UtilityCompositionError::Ontology(
                OntologyError::MissingProcessInputs
            ))
        );
    }

    #[test]
    fn detached_stale_projection_cannot_influence_preferred_path() {
        let process = baseline_process();
        let stale = project_process_utilities(&process).unwrap();
        assert_eq!(
            stale.electrical_energy_j,
            Some(EnergyRangeJ::new(90.0, 110.0).unwrap())
        );

        let mut changed = process.clone();
        changed.utilities = vec![
            UtilityDemand::ElectricalEnergy(EnergyRangeJ::new(140.0, 160.0).unwrap()),
            UtilityDemand::PeakElectricalPower(PowerRangeW::new(30.0, 35.0).unwrap()),
            UtilityDemand::ProcessTime(DurationRangeS::new(12.0, 14.0).unwrap()),
        ];

        let bound = compose_subject_bound_electrical_accounting_case(&changed, &baseline_context())
            .unwrap();
        assert_eq!(
            bound.gross_energy_j(),
            EnergyRangeJ::new(140.0, 160.0).unwrap()
        );
        assert_eq!(bound.peak_power_w(), PowerRangeW::new(30.0, 35.0).unwrap());
        assert_eq!(
            bound.batch_duration_s(),
            DurationRangeS::new(12.0, 14.0).unwrap()
        );
        assert_ne!(bound.gross_energy_j(), stale.electrical_energy_j.unwrap());
    }

    #[test]
    fn incomplete_electrical_basis_fails_through_fresh_projection() {
        let mut process = baseline_process();
        process
            .utilities
            .retain(|utility| !matches!(utility, UtilityDemand::ElectricalEnergy(_)));
        assert_eq!(
            compose_subject_bound_electrical_accounting_case(&process, &baseline_context()),
            Err(UtilityCompositionError::Binding(
                UtilityBindingError::ProjectionNotComplete(UtilityProjectionStatus::Incomplete)
            ))
        );
    }

    #[test]
    fn duplicate_peak_power_remains_ambiguous() {
        let mut process = baseline_process();
        process.utilities.push(UtilityDemand::PeakElectricalPower(
            PowerRangeW::new(1.0, 2.0).unwrap(),
        ));
        assert_eq!(
            compose_subject_bound_electrical_accounting_case(&process, &baseline_context()),
            Err(UtilityCompositionError::Binding(
                UtilityBindingError::ProjectionNotComplete(UtilityProjectionStatus::Ambiguous)
            ))
        );
    }

    #[test]
    fn duplicate_process_time_remains_ambiguous() {
        let mut process = baseline_process();
        process.utilities.push(UtilityDemand::ProcessTime(
            DurationRangeS::new(1.0, 2.0).unwrap(),
        ));
        assert_eq!(
            compose_subject_bound_electrical_accounting_case(&process, &baseline_context()),
            Err(UtilityCompositionError::Binding(
                UtilityBindingError::ProjectionNotComplete(UtilityProjectionStatus::Ambiguous)
            ))
        );
    }

    #[test]
    fn unresolved_thermal_and_cooling_semantics_survive_composition() {
        let mut process = baseline_process();
        process.utilities.extend([
            UtilityDemand::ThermalEnergy(EnergyRangeJ::new(50.0, 60.0).unwrap()),
            UtilityDemand::CoolingEnergy(EnergyRangeJ::new(10.0, 20.0).unwrap()),
        ]);

        let bound = compose_subject_bound_electrical_accounting_case(&process, &baseline_context())
            .unwrap();
        assert_eq!(
            bound.unresolved_non_electrical(),
            &[
                UtilityProjectionReason::ThermalTemperatureUnbound,
                UtilityProjectionReason::CoolingRejectionUnbound,
            ]
        );
    }

    #[test]
    fn explicit_context_is_preserved_without_feasibility_evaluation() {
        let process = baseline_process();
        let context = baseline_context();
        let bound = compose_subject_bound_electrical_accounting_case(&process, &context).unwrap();
        let case = bound.electrical_utility_case();

        assert_eq!(case.recoverable_energy_j, context.recoverable_energy_j);
        assert_eq!(case.recovery_duration_s, context.recovery_duration_s);
        assert_eq!(case.storage_acceptance_j, context.storage_acceptance_j);
        assert_eq!(case.storage_charge_power_w, context.storage_charge_power_w);
        assert_eq!(
            case.storage_discharge_power_w,
            context.storage_discharge_power_w
        );
        assert_eq!(
            case.recovery_delivery_fraction,
            context.recovery_delivery_fraction
        );
        assert_eq!(
            case.available_energy_capacity_j,
            context.available_energy_capacity_j
        );
        assert_eq!(
            case.available_continuous_power_w,
            context.available_sustained_power_w
        );
        assert_eq!(case.available_peak_power_w, context.available_peak_power_w);
    }
}
