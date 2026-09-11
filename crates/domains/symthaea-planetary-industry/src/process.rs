// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Industrial-process records for Planetary Industrial Ecology.

use serde::{Deserialize, Serialize};

use crate::types::{
    CelestialBody, DurationRangeS, EnergyRangeJ, EvidenceRef, GravityRangeMps2, MassRangeKg,
    MaterialGrade, OntologyError, PhysicalForm, PowerRangeW, PressureRangePa,
    TemperatureRangeK, require_text, validate_evidence,
};

/// Functional role of a process input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProcessInputRole {
    /// Principal material converted into products.
    Feedstock,
    /// Consumed process material.
    Consumable,
    /// Catalyst whose recovery/loss must be explicit when material balance matters.
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
    pub(crate) fn validate(&self) -> Result<(), OntologyError> {
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
    pub(crate) fn validate(&self) -> Result<(), OntologyError> {
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
    pub(crate) fn validate(&self) -> Result<(), OntologyError> {
        require_text(&self.material_key, "process_output_material_key")?;
        self.grade.validate()?;
        self.mass_kg.validate()?;
        self.disposition.validate()
    }
}

/// Non-material utility or service required by a process.
///
/// Matter such as water, process gases, reagents, catalysts and working fluids
/// must be represented as `ProcessInput`/`ProcessOutput`, never as utilities,
/// so PIE-001 can account for every kilogram crossing the process boundary.
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
}

impl UtilityDemand {
    pub(crate) fn validate(&self) -> Result<(), OntologyError> {
        match self {
            Self::ElectricalEnergy(v) | Self::ThermalEnergy(v) | Self::CoolingEnergy(v) => {
                v.validate()
            }
            Self::PeakElectricalPower(v) => v.validate(),
            Self::ProcessTime(v) => v.validate(),
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
    pub(crate) fn validate(&self) -> Result<(), OntologyError> {
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
    pub(crate) fn validate(&self) -> Result<(), OntologyError> {
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
    /// Non-material utility requirements.
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
    /// those are later gates.
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
