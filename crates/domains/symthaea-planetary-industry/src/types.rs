// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validated core types for Planetary Industrial Ecology.

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
    /// A graph reference targets an unknown process.
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
        pub struct $name(f64);

        impl $name {
            /// Construct a finite, non-negative quantity.
            pub fn new(value: f64) -> Result<Self, OntologyError> {
                if value.is_finite() && value >= 0.0 {
                    Ok(Self(value))
                } else {
                    Err(OntologyError::InvalidQuantity($label))
                }
            }

            /// Return the underlying SI-unit value.
            pub fn value(self) -> f64 {
                self.0
            }

            pub(crate) fn validate(self) -> Result<(), OntologyError> {
                Self::new(self.0).map(|_| ())
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

/// Origin/maturity class of supporting evidence.
///
/// Variants intentionally do not implement `Ord`: evidence provenance classes
/// are not a universal scalar ladder. For example, a vendor projection and a
/// literature model differ by provenance, not by an inherent total ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
    /// Evidence provenance/maturity class.
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
/// An occurrence is deliberately not a usable material lot. Acquisition,
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
    /// Material identity key; later bridges may resolve this to material/chemistry registries.
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

pub(crate) fn require_text(value: &str, field: &'static str) -> Result<(), OntologyError> {
    if value.trim().is_empty() {
        Err(OntologyError::EmptyField(field))
    } else {
        Ok(())
    }
}

pub(crate) fn validate_evidence(evidence: &[EvidenceRef]) -> Result<(), OntologyError> {
    let mut ids = BTreeSet::new();
    for item in evidence {
        item.validate()?;
        if !ids.insert(item.evidence_id.clone()) {
            return Err(OntologyError::DuplicateId(item.evidence_id.clone()));
        }
    }
    Ok(())
}
