// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Neutral, evidence-bearing ontology for planetary industrial ecology.
//!
//! PIE-000 intentionally models *what* an industrial process graph contains,
//! not whether a particular lunar or Martian process is feasible. Conservation,
//! chemistry, utility optimization, equipment reproduction, and control authority
//! belong to later layers.

#![deny(unsafe_code)]
#![warn(missing_docs)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

/// Validation failures for PIE ontology records.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OntologyError {
    /// A required textual identifier or label is empty.
    EmptyField(&'static str),
    /// A numeric quantity is negative or non-finite.
    InvalidQuantity(&'static str),
    /// A bounded range has invalid endpoints or `min > max`.
    InvalidRange(&'static str),
    /// Evidence stronger than a hypothesis lacks a source reference.
    MissingEvidenceSource,
    /// An identifier is duplicated within one graph namespace.
    DuplicateId(String),
    /// A graph edge references an unknown process.
    UnknownProcess(String),
    /// An equipment requirement declares zero units.
    ZeroEquipmentCount,
    /// A process definition contains no inputs.
    MissingProcessInputs,
    /// A process definition contains no outputs.
    MissingProcessOutputs,
}

impl fmt::Display for OntologyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "required field is empty: {field}"),
            Self::InvalidQuantity(kind) => write!(f, "invalid quantity: {kind}"),
            Self::InvalidRange(kind) => write!(f, "invalid range: {kind}"),
            Self::MissingEvidenceSource => write!(f, "non-hypothesis evidence requires a source"),
            Self::DuplicateId(id) => write!(f, "duplicate id: {id}"),
            Self::UnknownProcess(id) => write!(f, "unknown process id: {id}"),
            Self::ZeroEquipmentCount => write!(f, "equipment quantity must be greater than zero"),
            Self::MissingProcessInputs => write!(f, "process requires at least one input"),
            Self::MissingProcessOutputs => write!(f, "process requires at least one output"),
        }
    }
}

impl Error for OntologyError {}

macro_rules! quantity_type {
    ($name:ident, $doc:literal, $label:literal) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
        pub struct $name(pub f64);

        impl $name {
            /// Construct a finite, non-negative quantity.
            pub fn new(value: f64) -> Result<Self, OntologyError> {
                let candidate = Self(value);
                candidate.validate()?;
                Ok(candidate)
            }

            /// Return the underlying SI-unit value.
            pub fn value(self) -> f64 {
                self.0
            }

            fn validate(self) -> Result<(), OntologyError> {
                if self.0.is_finite() && self.0 >= 0.0 {
                    Ok(())
                } else {
                    Err(OntologyError::InvalidQuantity($label))
                }
            }
        }
    };
}

macro_rules! range_type {
    ($name:ident, $quantity:ident, $doc:literal, $label:literal) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
        pub struct $name {
            /// Conservative lower bound.
            pub min: $quantity,
            /// Conservative upper bound.
            pub max: $quantity,
        }

        impl $name {
            /// Construct a validated inclusive range.
            pub fn new(min: f64, max: f64) -> Result<Self, OntologyError> {
                let range = Self {
                    min: $quantity::new(min)?,
                    max: $quantity::new(max)?,
                };
                range.validate()?;
                Ok(range)
            }

            /// Validate finite, non-negative, ordered bounds.
            pub fn validate(&self) -> Result<(), OntologyError> {
                self.min.validate()?;
                self.max.validate()?;
                if self.min <= self.max {
                    Ok(())
                } else {
                    Err(OntologyError::InvalidRange($label))
                }
            }
        }
    };
}

quantity_type!(MassKg, "Mass in kilograms.", "mass_kg");
quantity_type!(EnergyJ, "Energy in joules.", "energy_j");
quantity_type!(PowerW, "Power in watts.", "power_w");
quantity_type!(DurationS, "Duration in seconds.", "duration_s");
quantity_type!(TemperatureK, "Absolute temperature in kelvin.", "temperature_k");
quantity_type!(PressurePa, "Absolute pressure in pascals.", "pressure_pa");
quantity_type!(GravityMps2, "Acceleration magnitude in metres per second squared.", "gravity_m_s2");

range_type!(MassRangeKg, MassKg, "Inclusive mass range in kilograms.", "mass_kg");
range_type!(EnergyRangeJ, EnergyJ, "Inclusive energy range in joules.", "energy_j");
range_type!(PowerRangeW, PowerW, "Inclusive power range in watts.", "power_w");
range_type!(DurationRangeS, DurationS, "Inclusive duration range in seconds.", "duration_s");
range_type!(TemperatureRangeK, TemperatureK, "Inclusive temperature range in kelvin.", "temperature_k");
range_type!(PressureRangePa, PressurePa, "Inclusive pressure range in pascals.", "pressure_pa");
range_type!(GravityRangeMps2, GravityMps2, "Inclusive gravity range in m/s².", "gravity_m_s2");

/// Planetary body associated with a resource or process constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CelestialBody {
    /// Earth, primarily for imported/reference chains.
    Earth,
    /// Earth's Moon.
    Moon,
    /// Mars.
    Mars,
    /// A body not yet represented by a dedicated variant.
    Other,
}

/// Strength and origin class of supporting evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EvidenceClass {
    /// Explicit conjecture or placeholder requiring no external source.
    Hypothesis,
    /// Published analytical or numerical model.
    LiteratureModel,
    /// Supplier/developer projection not independently qualified.
    VendorProjection,
    /// Laboratory measurement.
    LabMeasured,
    /// Measurement in a materially relevant environment.
    RelevantEnvironmentMeasured,
    /// Integrated system demonstration.
    IntegratedDemonstration,
    /// Qualification-grade evidence for the declared scope.
    Qualified,
}

/// Provenance record attached to a resource, process, requirement, or lot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceRef {
    /// Stable evidence identifier within the study/program.
    pub evidence_id: String,
    /// Evidence maturity/class.
    pub class: EvidenceClass,
    /// DOI, URL, report identifier, receipt hash, or other resolvable source.
    pub source: String,
    /// Optional scope caveat or interpretation note.
    pub note: Option<String>,
}

impl EvidenceRef {
    /// Validate identifier and evidence-source requirements.
    pub fn validate(&self) -> Result<(), OntologyError> {
        require_text(&self.evidence_id, "evidence_id")?;
        if self.class != EvidenceClass::Hypothesis && self.source.trim().is_empty() {
            return Err(OntologyError::MissingEvidenceSource);
        }
        Ok(())
    }
}

/// Broad physical state of an in-situ or recycled resource occurrence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceState {
    /// Unconsolidated regolith.
    Regolith,
    /// Consolidated rock or ore body.
    Rock,
    /// Frozen volatile-bearing material.
    Ice,
    /// Atmospheric or captured gas.
    Gas,
    /// Liquid resource.
    Liquid,
    /// Brine or dissolved-resource stream.
    Brine,
    /// Biological resource or biomass.
    Biomass,
    /// Scrap material awaiting reprocessing.
    Scrap,
    /// Waste stream with potential resource value.
    Waste,
    /// Other declared physical state.
    Other,
}

/// Evidence-bearing occurrence of a resource in an environment.
///
/// An occurrence is deliberately *not* a usable material lot. Acquisition,
/// beneficiation, recovery loss, and grade conversion are separate processes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceOccurrence {
    /// Stable resource-occurrence identifier.
    pub resource_id: String,
    /// Human-readable resource name.
    pub name: String,
    /// Body where the occurrence exists.
    pub body: CelestialBody,
    /// Broad physical state.
    pub state: ResourceState,
    /// Estimated in-place quantity when known.
    pub in_place_mass_kg: Option<MassRangeKg>,
    /// Supporting evidence.
    pub evidence: Vec<EvidenceRef>,
}

impl ResourceOccurrence {
    /// Validate the occurrence record without implying recoverability.
    pub fn validate(&self) -> Result<(), OntologyError> {
        require_text(&self.resource_id, "resource_id")?;
        require_text(&self.name, "resource_name")?;
        if let Some(range) = self.in_place_mass_kg {
            range.validate()?;
        }
        validate_evidence(&self.evidence)
    }
}

/// Broad physical form of a material lot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhysicalForm {
    /// Unclassified bulk material.
    Bulk,
    /// Powder.
    Powder,
    /// Granular material.
    Granules,
    /// Solid ingot or billet.
    Ingot,
    /// Wire.
    Wire,
    /// Sheet or plate.
    Sheet,
    /// Gas.
    Gas,
    /// Liquid.
    Liquid,
    /// Slurry or suspension.
    Slurry,
    /// Ceramic/glass-like consolidated form.
    Ceramic,
    /// Composite form.
    Composite,
    /// Biological material.
    Biological,
    /// Other declared form.
    Other,
}

/// Declared material quality/grade label.
///
/// PIE-003 will add composition, contaminants, specifications, and grade
/// transition logic. PIE-000 keeps the grade identity explicit so a material
/// name alone never implies fitness for use.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterialGrade {
    /// Human-readable grade label.
    pub label: String,
    /// Optional external specification or acceptance reference.
    pub specification_ref: Option<String>,
}

impl MaterialGrade {
    /// Validate the grade label.
    pub fn validate(&self) -> Result<(), OntologyError> {
        require_text(&self.label, "material_grade")
    }
}

/// Origin class for a material lot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaterialOriginKind {
    /// Derived from an identified local resource occurrence.
    LocalResource,
    /// Imported from Earth or another external supplier.
    Imported,
    /// Recovered from an existing product or waste stream.
    Recycled,
    /// Produced locally from one or more prior process steps.
    ProcessedLocal,
    /// Blended local/imported/recycled provenance.
    Mixed,
}

/// Provenance pointer for a material lot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterialOrigin {
    /// Origin category.
    pub kind: MaterialOriginKind,
    /// Identifier for the source occurrence, process batch, import receipt, or recycle batch.
    pub source_id: String,
}

impl MaterialOrigin {
    /// Validate the provenance pointer.
    pub fn validate(&self) -> Result<(), OntologyError> {
        require_text(&self.source_id, "material_origin_source_id")
    }
}

/// Inventory lot suitable for process/fabrication planning.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaterialLot {
    /// Stable lot/batch identifier.
    pub lot_id: String,
    /// Material identity key; may later bridge to `symthaea-materials` or a chemistry registry.
    pub material_key: String,
    /// Declared quality grade.
    pub grade: MaterialGrade,
    /// Physical form.
    pub form: PhysicalForm,
    /// Current mass.
    pub mass_kg: MassKg,
    /// Provenance/origin.
    pub origin: MaterialOrigin,
    /// Evidence supporting identity/grade/provenance.
    pub evidence: Vec<EvidenceRef>,
}

impl MaterialLot {
    /// Validate lot identity, quantity, grade, origin, and evidence.
    pub fn validate(&self) -> Result<(), OntologyError> {
        require_text(&self.lot_id, "lot_id")?;
        require_text(&self.material_key, "material_key")?;
        self.grade.validate()?;
        self.mass_kg.validate()?;
        self.origin.validate()?;
        validate_evidence(&self.evidence)
    }
}

/// Functional role of a process input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProcessInputRole {
    /// Principal material converted into products.
    Feedstock,
    /// Consumed process material.
    Consumable,
    /// Catalyst whose recovery/loss will be modeled in later tranches.
    Catalyst,
    /// Working fluid or process gas.
    WorkingFluid,
    /// Recycled material returning to the process chain.
    RecycledFeed,
}

/// Required material input for one process execution/batch basis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProcessInput {
    /// Material identity key.
    pub material_key: String,
    /// Optional minimum required grade.
    pub required_grade: Option<MaterialGrade>,
    /// Input function.
    pub role: ProcessInputRole,
    /// Required mass range on the declared process basis.
    pub mass_kg: MassRangeKg,
}

impl ProcessInput {
    fn validate(&self) -> Result<(), OntologyError> {
        require_text(&self.material_key, "process_input_material_key")?;
        if let Some(grade) = &self.required_grade {
            grade.validate()?;
        }
        self.mass_kg.validate()
    }
}

/// Functional role of a process output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProcessOutputRole {
    /// Primary intended product.
    Product,
    /// Intended co-product.
    Coproduct,
    /// Secondary output that may have value.
    Byproduct,
    /// Waste requiring explicit handling.
    Waste,
    /// Output explicitly intended for recycling/reprocessing.
    RecycleCandidate,
}

/// Declared destination for a process output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputDisposition {
    /// Retain as usable inventory.
    Inventory,
    /// Route to a declared sink/treatment/storage destination.
    Sink(String),
    /// Route toward a declared downstream/recycle process.
    Process(String),
    /// Consequence/destination is not yet known; kept explicit rather than disappearing.
    Unknown,
}

impl OutputDisposition {
    fn validate(&self) -> Result<(), OntologyError> {
        match self {
            Self::Sink(id) => require_text(id, "output_sink_id"),
            Self::Process(id) => require_text(id, "output_process_id"),
            Self::Inventory | Self::Unknown => Ok(()),
        }
    }
}

/// Material output on one declared process basis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProcessOutput {
    /// Material identity key.
    pub material_key: String,
    /// Declared output grade.
    pub grade: MaterialGrade,
    /// Output physical form.
    pub form: PhysicalForm,
    /// Output role.
    pub role: ProcessOutputRole,
    /// Output mass range.
    pub mass_kg: MassRangeKg,
    /// Explicit disposition so by-products/wastes never vanish from the graph.
    pub disposition: OutputDisposition,
}

impl ProcessOutput {
    fn validate(&self) -> Result<(), OntologyError> {
        require_text(&self.material_key, "process_output_material_key")?;
        self.grade.validate()?;
        self.mass_kg.validate()?;
        self.disposition.validate()
    }
}

/// Utility or service required by a process.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UtilityDemand {
    /// Electrical energy consumption.
    ElectricalEnergy(EnergyRangeJ),
    /// Thermal energy supplied to the process.
    ThermalEnergy(EnergyRangeJ),
    /// Cooling/heat-rejection duty expressed as energy.
    CoolingEnergy(EnergyRangeJ),
    /// Peak electrical power requirement.
    PeakElectricalPower(PowerRangeW),
    /// Process duration/residence time.
    ProcessTime(DurationRangeS),
    /// Process-water mass requirement.
    ProcessWater(MassRangeKg),
    /// Required operating-pressure envelope.
    OperatingPressure(PressureRangePa),
}

impl UtilityDemand {
    fn validate(&self) -> Result<(), OntologyError> {
        match self {
            Self::ElectricalEnergy(v) | Self::ThermalEnergy(v) | Self::CoolingEnergy(v) => v.validate(),
            Self::PeakElectricalPower(v) => v.validate(),
            Self::ProcessTime(v) => v.validate(),
            Self::ProcessWater(v) => v.validate(),
            Self::OperatingPressure(v) => v.validate(),
        }
    }
}

/// Importance of a required equipment class to process execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DependencyCriticality {
    /// Process cannot execute without it.
    Essential,
    /// Process can execute in a degraded mode or at reduced rate.
    RateLimiting,
    /// Optional enhancement.
    Optional,
}

/// Equipment class required to execute a process.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EquipmentRequirement {
    /// Equipment class/key; detailed machine instances live elsewhere.
    pub equipment_class: String,
    /// Number of units required on the declared process basis.
    pub quantity: u32,
    /// Dependency importance.
    pub criticality: DependencyCriticality,
    /// Evidence for the requirement when available.
    pub evidence: Vec<EvidenceRef>,
}

impl EquipmentRequirement {
    fn validate(&self) -> Result<(), OntologyError> {
        require_text(&self.equipment_class, "equipment_class")?;
        if self.quantity == 0 {
            return Err(OntologyError::ZeroEquipmentCount);
        }
        validate_evidence(&self.evidence)
    }
}

/// Environmental envelope/compatibility requirement for a process.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EnvironmentConstraint {
    /// Process is scoped to a specific body.
    Body(CelestialBody),
    /// Temperature operating range.
    Temperature(TemperatureRangeK),
    /// Ambient/process pressure range.
    Pressure(PressureRangePa),
    /// Gravity magnitude range.
    Gravity(GravityRangeMps2),
    /// Process/equipment must support operation in vacuum.
    VacuumCompatible,
}

impl EnvironmentConstraint {
    fn validate(&self) -> Result<(), OntologyError> {
        match self {
            Self::Temperature(v) => v.validate(),
            Self::Pressure(v) => v.validate(),
            Self::Gravity(v) => v.validate(),
            Self::Body(_) | Self::VacuumCompatible => Ok(()),
        }
    }
}

/// Neutral process definition. It describes requirements and flows but grants no control authority.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProcessDefinition {
    /// Stable process identifier.
    pub process_id: String,
    /// Human-readable process name.
    pub name: String,
    /// Material inputs.
    pub inputs: Vec<ProcessInput>,
    /// Material outputs.
    pub outputs: Vec<ProcessOutput>,
    /// Utility requirements.
    pub utilities: Vec<UtilityDemand>,
    /// Required equipment classes.
    pub equipment: Vec<EquipmentRequirement>,
    /// Environmental compatibility requirements.
    pub environment: Vec<EnvironmentConstraint>,
    /// Evidence supporting this process definition/assumptions.
    pub evidence: Vec<EvidenceRef>,
}

impl ProcessDefinition {
    /// Validate structural completeness, units/ranges, and evidence provenance.
    ///
    /// PIE-000 deliberately does not enforce mass or elemental conservation;
    /// those are PIE-001 gates.
    pub fn validate(&self) -> Result<(), OntologyError> {
        require_text(&self.process_id, "process_id")?;
        require_text(&self.name, "process_name")?;
        if self.inputs.is_empty() {
            return Err(OntologyError::MissingProcessInputs);
        }
        if self.outputs.is_empty() {
            return Err(OntologyError::MissingProcessOutputs);
        }
        for input in &self.inputs {
            input.validate()?;
        }
        for output in &self.outputs {
            output.validate()?;
        }
        for utility in &self.utilities {
            utility.validate()?;
        }
        for equipment in &self.equipment {
            equipment.validate()?;
        }
        for constraint in &self.environment {
            constraint.validate()?;
        }
        validate_evidence(&self.evidence)
    }
}

/// Explicit recycle/reprocessing edge between two known processes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecycleEdge {
    /// Stable edge identifier.
    pub edge_id: String,
    /// Process producing the recyclable material.
    pub from_process_id: String,
    /// Material identity routed on this edge.
    pub material_key: String,
    /// Process consuming/reprocessing the material.
    pub to_process_id: String,
    /// Maximum recovered material routed on the declared basis.
    pub recovered_mass_kg: MassRangeKg,
    /// Supporting evidence or explicit hypothesis.
    pub evidence: Vec<EvidenceRef>,
}

impl RecycleEdge {
    fn validate(&self) -> Result<(), OntologyError> {
        require_text(&self.edge_id, "recycle_edge_id")?;
        require_text(&self.from_process_id, "recycle_from_process_id")?;
        require_text(&self.to_process_id, "recycle_to_process_id")?;
        require_text(&self.material_key, "recycle_material_key")?;
        self.recovered_mass_kg.validate()?;
        validate_evidence(&self.evidence)
    }
}

/// Structurally closed PIE process graph.
///
/// This validates IDs and recycle-edge references only; it does not claim that
/// material/energy balances close or that resources are geographically available.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct IndustrialProcessGraph {
    /// Resource occurrences considered by the study.
    pub resources: Vec<ResourceOccurrence>,
    /// Existing inventory lots considered by the study.
    pub lots: Vec<MaterialLot>,
    /// Candidate industrial processes.
    pub processes: Vec<ProcessDefinition>,
    /// Explicit recycling/reprocessing links.
    pub recycle_edges: Vec<RecycleEdge>,
}

impl IndustrialProcessGraph {
    /// Validate record integrity, namespace uniqueness, and recycle referential closure.
    pub fn validate_structure(&self) -> Result<(), OntologyError> {
        let mut resource_ids = BTreeSet::new();
        for resource in &self.resources {
            resource.validate()?;
            if !resource_ids.insert(resource.resource_id.clone()) {
                return Err(OntologyError::DuplicateId(resource.resource_id.clone()));
            }
        }

        let mut lot_ids = BTreeSet::new();
        for lot in &self.lots {
            lot.validate()?;
            if !lot_ids.insert(lot.lot_id.clone()) {
                return Err(OntologyError::DuplicateId(lot.lot_id.clone()));
            }
        }

        let mut process_ids = BTreeSet::new();
        for process in &self.processes {
            process.validate()?;
            if !process_ids.insert(process.process_id.clone()) {
                return Err(OntologyError::DuplicateId(process.process_id.clone()));
            }
        }

        let mut edge_ids = BTreeSet::new();
        for edge in &self.recycle_edges {
            edge.validate()?;
            if !edge_ids.insert(edge.edge_id.clone()) {
                return Err(OntologyError::DuplicateId(edge.edge_id.clone()));
            }
            if !process_ids.contains(&edge.from_process_id) {
                return Err(OntologyError::UnknownProcess(edge.from_process_id.clone()));
            }
            if !process_ids.contains(&edge.to_process_id) {
                return Err(OntologyError::UnknownProcess(edge.to_process_id.clone()));
            }
        }
        Ok(())
    }
}

fn require_text(value: &str, field: &'static str) -> Result<(), OntologyError> {
    if value.trim().is_empty() {
        Err(OntologyError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn validate_evidence(evidence: &[EvidenceRef]) -> Result<(), OntologyError> {
    let mut ids = BTreeSet::new();
    for item in evidence {
        item.validate()?;
        if !ids.insert(item.evidence_id.clone()) {
            return Err(OntologyError::DuplicateId(item.evidence_id.clone()));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hypothesis(id: &str) -> EvidenceRef {
        EvidenceRef {
            evidence_id: id.into(),
            class: EvidenceClass::Hypothesis,
            source: String::new(),
            note: Some("synthetic test fixture only".into()),
        }
    }

    fn grade(label: &str) -> MaterialGrade {
        MaterialGrade {
            label: label.into(),
            specification_ref: None,
        }
    }

    fn synthetic_process(id: &str, output_disposition: OutputDisposition) -> ProcessDefinition {
        ProcessDefinition {
            process_id: id.into(),
            name: format!("Synthetic process {id}"),
            inputs: vec![ProcessInput {
                material_key: "synthetic-feed".into(),
                required_grade: Some(grade("research-grade")),
                role: ProcessInputRole::Feedstock,
                mass_kg: MassRangeKg::new(9.0, 11.0).unwrap(),
            }],
            outputs: vec![ProcessOutput {
                material_key: "synthetic-product".into(),
                grade: grade("unqualified-product"),
                form: PhysicalForm::Bulk,
                role: ProcessOutputRole::Product,
                mass_kg: MassRangeKg::new(5.0, 10.0).unwrap(),
                disposition: output_disposition,
            }],
            utilities: vec![
                UtilityDemand::ElectricalEnergy(EnergyRangeJ::new(100.0, 200.0).unwrap()),
                UtilityDemand::PeakElectricalPower(PowerRangeW::new(10.0, 20.0).unwrap()),
            ],
            equipment: vec![EquipmentRequirement {
                equipment_class: "synthetic-reactor".into(),
                quantity: 1,
                criticality: DependencyCriticality::Essential,
                evidence: vec![hypothesis("equip-hyp")],
            }],
            environment: vec![EnvironmentConstraint::Body(CelestialBody::Moon)],
            evidence: vec![hypothesis("process-hyp")],
        }
    }

    #[test]
    fn ranges_fail_closed_on_negative_nonfinite_and_reversed_values() {
        assert!(MassRangeKg::new(-1.0, 2.0).is_err());
        assert!(MassRangeKg::new(3.0, 2.0).is_err());
        assert!(EnergyRangeJ::new(0.0, f64::NAN).is_err());
    }

    #[test]
    fn measured_evidence_requires_a_source() {
        let evidence = EvidenceRef {
            evidence_id: "lab-1".into(),
            class: EvidenceClass::LabMeasured,
            source: String::new(),
            note: None,
        };
        assert_eq!(evidence.validate(), Err(OntologyError::MissingEvidenceSource));
    }

    #[test]
    fn occurrence_is_not_a_material_lot() {
        let occurrence = ResourceOccurrence {
            resource_id: "occurrence-1".into(),
            name: "Synthetic icy regolith".into(),
            body: CelestialBody::Moon,
            state: ResourceState::Ice,
            in_place_mass_kg: Some(MassRangeKg::new(100.0, 200.0).unwrap()),
            evidence: vec![hypothesis("resource-hyp")],
        };
        let lot = MaterialLot {
            lot_id: "lot-1".into(),
            material_key: "processed-water".into(),
            grade: grade("industrial"),
            form: PhysicalForm::Liquid,
            mass_kg: MassKg::new(5.0).unwrap(),
            origin: MaterialOrigin {
                kind: MaterialOriginKind::ProcessedLocal,
                source_id: "batch-1".into(),
            },
            evidence: vec![hypothesis("lot-hyp")],
        };
        assert!(occurrence.validate().is_ok());
        assert!(lot.validate().is_ok());
        assert_ne!(occurrence.resource_id, lot.lot_id);
    }

    #[test]
    fn explicit_unknown_output_is_valid_but_not_hidden() {
        let process = synthetic_process("p1", OutputDisposition::Unknown);
        assert!(process.validate().is_ok());
        assert_eq!(process.outputs[0].disposition, OutputDisposition::Unknown);
    }

    #[test]
    fn zero_equipment_count_is_rejected() {
        let mut process = synthetic_process("p1", OutputDisposition::Inventory);
        process.equipment[0].quantity = 0;
        assert_eq!(process.validate(), Err(OntologyError::ZeroEquipmentCount));
    }

    #[test]
    fn graph_rejects_dangling_recycle_edges() {
        let graph = IndustrialProcessGraph {
            resources: vec![],
            lots: vec![],
            processes: vec![synthetic_process("p1", OutputDisposition::Inventory)],
            recycle_edges: vec![RecycleEdge {
                edge_id: "r1".into(),
                from_process_id: "p1".into(),
                material_key: "synthetic-product".into(),
                to_process_id: "missing".into(),
                recovered_mass_kg: MassRangeKg::new(1.0, 2.0).unwrap(),
                evidence: vec![hypothesis("recycle-hyp")],
            }],
        };
        assert_eq!(
            graph.validate_structure(),
            Err(OntologyError::UnknownProcess("missing".into()))
        );
    }

    #[test]
    fn graph_accepts_structurally_closed_recycle_edge() {
        let graph = IndustrialProcessGraph {
            resources: vec![],
            lots: vec![],
            processes: vec![
                synthetic_process("p1", OutputDisposition::Process("p2".into())),
                synthetic_process("p2", OutputDisposition::Inventory),
            ],
            recycle_edges: vec![RecycleEdge {
                edge_id: "r1".into(),
                from_process_id: "p1".into(),
                material_key: "synthetic-product".into(),
                to_process_id: "p2".into(),
                recovered_mass_kg: MassRangeKg::new(1.0, 2.0).unwrap(),
                evidence: vec![hypothesis("recycle-hyp")],
            }],
        };
        assert!(graph.validate_structure().is_ok());
    }

    #[test]
    fn duplicate_ids_fail_closed_within_namespace() {
        let graph = IndustrialProcessGraph {
            resources: vec![],
            lots: vec![],
            processes: vec![
                synthetic_process("dup", OutputDisposition::Inventory),
                synthetic_process("dup", OutputDisposition::Inventory),
            ],
            recycle_edges: vec![],
        };
        assert_eq!(
            graph.validate_structure(),
            Err(OntologyError::DuplicateId("dup".into()))
        );
    }
}
