// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Reproducible identity for physical material systems.
//!
//! A material-discovery subject is more than a chemical formula. Processing,
//! microstructure, layer order, thickness, gradients, interfaces, and porosity can
//! materially change behavior. This module therefore gives those features explicit
//! identity without granting any scientific authority to the resulting subject.
//!
//! The canonical identity is intentionally independent of human-facing labels.
//! Renaming a candidate does not create new science; changing an identity-bearing
//! physical or lineage parameter does.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Exact denominator for normalized atomic- or constituent-fraction vectors.
pub const SUBJECT_FRACTION_PPM_TOTAL: u32 = 1_000_000;

const SHA256_HEX_LEN: usize = 64;

/// One element in an exact normalized atomic-fraction composition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ElementFractionPpm {
    /// Atomic number.
    pub atomic_number: u16,
    /// Atomic fraction in parts per million of the whole composition.
    pub fraction_ppm: u32,
}

/// Exact normalized atomic composition, suitable for continuous alloy spaces.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AtomicCompositionPpm {
    components: Vec<ElementFractionPpm>,
}

impl AtomicCompositionPpm {
    /// Construct a canonical atomic-fraction composition.
    pub fn new(
        mut components: Vec<ElementFractionPpm>,
    ) -> Result<Self, MaterialSubjectError> {
        validate_fraction_components(&components)?;
        components.sort_by_key(|component| component.atomic_number);
        Ok(Self { components })
    }

    /// Canonically ordered components.
    pub fn components(&self) -> &[ElementFractionPpm] {
        &self.components
    }

    fn validate(&self) -> Result<(), MaterialSubjectError> {
        validate_fraction_components(&self.components)
    }

    fn canonical_key(&self) -> String {
        let mut components = self.components.clone();
        components.sort_by_key(|component| component.atomic_number);
        components
            .iter()
            .map(|component| {
                format!("Z{}={:06}", component.atomic_number, component.fraction_ppm)
            })
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// One integer stoichiometric coefficient in an exact formula-unit composition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ElementStoichiometry {
    /// Atomic number.
    pub atomic_number: u16,
    /// Positive integer atom count before canonical greatest-common-divisor reduction.
    pub atom_count: u32,
}

/// Exact reduced integer stoichiometry, suitable for compounds such as SiO2.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoichiometricComposition {
    components: Vec<ElementStoichiometry>,
}

impl StoichiometricComposition {
    /// Construct a canonical reduced stoichiometric composition.
    pub fn new(
        mut components: Vec<ElementStoichiometry>,
    ) -> Result<Self, MaterialSubjectError> {
        validate_stoichiometry(&components)?;
        let divisor = components
            .iter()
            .map(|component| component.atom_count)
            .reduce(gcd)
            .unwrap_or(1);
        for component in &mut components {
            component.atom_count /= divisor;
        }
        components.sort_by_key(|component| component.atomic_number);
        Ok(Self { components })
    }

    /// Canonically ordered reduced components.
    pub fn components(&self) -> &[ElementStoichiometry] {
        &self.components
    }

    fn validate(&self) -> Result<(), MaterialSubjectError> {
        validate_stoichiometry(&self.components)
    }

    fn canonical_key(&self) -> String {
        let mut components = self.components.clone();
        let divisor = components
            .iter()
            .map(|component| component.atom_count)
            .reduce(gcd)
            .unwrap_or(1);
        for component in &mut components {
            component.atom_count /= divisor;
        }
        components.sort_by_key(|component| component.atomic_number);
        components
            .iter()
            .map(|component| format!("Z{}={}", component.atomic_number, component.atom_count))
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// Composition representation used by a material phase.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaterialComposition {
    /// Exact normalized atomic fractions for alloys, solutions, or measured bulk chemistry.
    AtomicFractionPpm(AtomicCompositionPpm),
    /// Exact reduced integer formula units for compounds.
    Stoichiometric(StoichiometricComposition),
}

impl MaterialComposition {
    fn validate(&self) -> Result<(), MaterialSubjectError> {
        match self {
            Self::AtomicFractionPpm(composition) => composition.validate(),
            Self::Stoichiometric(composition) => composition.validate(),
        }
    }

    fn canonical_key(&self) -> String {
        match self {
            Self::AtomicFractionPpm(composition) => {
                format!("atomic-ppm[{}]", composition.canonical_key())
            }
            Self::Stoichiometric(composition) => {
                format!("stoich[{}]", composition.canonical_key())
            }
        }
    }
}

/// Physical phase state declared for a material phase.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaterialPhaseState {
    /// Crystalline phase with an explicitly named structure/prototype.
    Crystalline {
        /// Structure/prototype label, such as `bcc`, `fcc`, `P6_3/mmc`, or a database prototype ID.
        structure_id: String,
    },
    /// Amorphous or glassy state.
    Amorphous,
    /// Liquid state.
    Liquid,
    /// Deliberately mixed/multiphase state represented as one phase reference.
    Mixed {
        /// Bound description or external phase-mixture identifier.
        phase_mixture_id: String,
    },
    /// Phase state has not yet been established.
    Unknown,
}

/// One composition-plus-phase subject that can be referenced by larger architectures.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterialPhaseSubject {
    /// Stable local or external phase identifier.
    pub phase_id: String,
    /// Exact composition representation.
    pub composition: MaterialComposition,
    /// Physical phase state.
    pub phase_state: MaterialPhaseState,
}

/// Artifact binding used to make processing or microstructure part of subject identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubjectLineageRef {
    /// Stable route/state identifier.
    pub lineage_id: String,
    /// SHA-256 of the exact protocol, process record, or characterization artifact.
    pub artifact_sha256: String,
}

/// One constituent in an order-independent composite architecture.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompositeConstituent {
    /// Identity of a separately registered material subject or phase.
    pub material_ref: String,
    /// Constituent fraction in ppm on the declared composite basis.
    pub fraction_ppm: u32,
}

/// One ordered layer in a stack, listed from substrate outward.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterialLayer {
    /// Referenced material subject or phase identity.
    pub material_ref: String,
    /// Layer thickness in nanometres.
    pub thickness_nm: u64,
    /// Optional layer-specific process lineage.
    pub process_lineage: Option<SubjectLineageRef>,
}

/// One exact composition point in a through-thickness graded layer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompositionGradientKnot {
    /// Position through the graded layer, from 0 ppm at the substrate side to
    /// 1,000,000 ppm at the outer side.
    pub position_ppm: u32,
    /// Exact composition at this position.
    pub composition: MaterialComposition,
}

/// Physical architecture of a material subject.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaterialArchitecture {
    /// Single bulk phase or intentionally unresolved bulk phase state.
    Bulk {
        /// Bulk phase identity.
        phase: MaterialPhaseSubject,
    },
    /// Order-independent mixture/composite with exact constituent fractions.
    Composite {
        /// Exact constituent shares. Fractions must sum to one million ppm.
        constituents: Vec<CompositeConstituent>,
    },
    /// Ordered substrate-plus-layer architecture.
    LayerStack {
        /// Substrate material identity.
        substrate_ref: String,
        /// Layers ordered from substrate outward. Layer order is identity-bearing.
        layers: Vec<MaterialLayer>,
    },
    /// Continuous/discretely sampled composition gradient through one layer.
    GradedLayer {
        /// Substrate material identity.
        substrate_ref: String,
        /// Total graded-layer thickness in nanometres.
        thickness_nm: u64,
        /// Strictly increasing composition knots including 0 and 1,000,000 ppm endpoints.
        knots: Vec<CompositionGradientKnot>,
    },
    /// Porous matrix architecture.
    Porous {
        /// Matrix material identity.
        matrix_ref: String,
        /// Porosity fraction in ppm. Must be less than one million.
        porosity_ppm: u32,
        /// Optional characteristic pore scale in nanometres.
        characteristic_pore_nm: Option<u64>,
    },
    /// Directed interface where side A and side B are intentionally ordered.
    DirectedInterface {
        /// Material on side A.
        side_a_ref: String,
        /// Material on side B.
        side_b_ref: String,
        /// Optional finite interfacial width in nanometres.
        interface_width_nm: Option<u64>,
    },
}

/// Fully identified physical material subject.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterialSubject {
    /// Schema version for migration and canonical-identity stability.
    pub schema_version: u32,
    /// Human-facing label. This field is deliberately not identity-bearing.
    pub display_label: String,
    /// Physical architecture.
    pub architecture: MaterialArchitecture,
    /// Exact processing lineage. Processing changes therefore create a different subject.
    pub process_lineage: SubjectLineageRef,
    /// Optional characterized microstructure lineage. A different bound microstructure
    /// artifact creates a different subject even when chemistry/process labels match.
    pub microstructure_lineage: Option<SubjectLineageRef>,
}

impl MaterialSubject {
    /// Validate all identity-bearing fields.
    pub fn validate(&self) -> Result<(), MaterialSubjectError> {
        if self.schema_version != 1 {
            return Err(MaterialSubjectError::UnsupportedSchemaVersion(
                self.schema_version,
            ));
        }
        if self.display_label.trim().is_empty() {
            return Err(MaterialSubjectError::EmptyDisplayLabel);
        }
        validate_lineage(&self.process_lineage)?;
        if let Some(lineage) = &self.microstructure_lineage {
            validate_lineage(lineage)?;
        }
        validate_architecture(&self.architecture)
    }

    /// Canonical, deterministic identity string for evidence binding.
    ///
    /// Human-facing `display_label` is excluded by design. The returned string may
    /// be hashed by a higher layer if a compact cryptographic identifier is needed.
    pub fn canonical_identity(&self) -> Result<String, MaterialSubjectError> {
        self.validate()?;
        let microstructure = self
            .microstructure_lineage
            .as_ref()
            .map(lineage_key)
            .unwrap_or_else(|| "none".to_string());
        Ok(format!(
            "material-subject:v{}|arch={}|process={}|micro={}",
            self.schema_version,
            architecture_key(&self.architecture),
            lineage_key(&self.process_lineage),
            microstructure
        ))
    }
}

fn validate_architecture(architecture: &MaterialArchitecture) -> Result<(), MaterialSubjectError> {
    match architecture {
        MaterialArchitecture::Bulk { phase } => validate_phase(phase),
        MaterialArchitecture::Composite { constituents } => {
            if constituents.len() < 2 {
                return Err(MaterialSubjectError::NeedMultipleConstituents);
            }
            let mut seen = HashSet::new();
            let mut total = 0_u64;
            for constituent in constituents {
                validate_ref(&constituent.material_ref)?;
                if constituent.fraction_ppm == 0 {
                    return Err(MaterialSubjectError::ZeroFractionConstituent);
                }
                if !seen.insert(constituent.material_ref.as_str()) {
                    return Err(MaterialSubjectError::DuplicateMaterialRef(
                        constituent.material_ref.clone(),
                    ));
                }
                total += u64::from(constituent.fraction_ppm);
            }
            if total != u64::from(SUBJECT_FRACTION_PPM_TOTAL) {
                return Err(MaterialSubjectError::FractionsDoNotSumToOne {
                    sum: total,
                });
            }
            Ok(())
        }
        MaterialArchitecture::LayerStack {
            substrate_ref,
            layers,
        } => {
            validate_ref(substrate_ref)?;
            if layers.is_empty() {
                return Err(MaterialSubjectError::EmptyLayerStack);
            }
            for layer in layers {
                validate_ref(&layer.material_ref)?;
                if layer.thickness_nm == 0 {
                    return Err(MaterialSubjectError::ZeroThickness);
                }
                if let Some(lineage) = &layer.process_lineage {
                    validate_lineage(lineage)?;
                }
            }
            Ok(())
        }
        MaterialArchitecture::GradedLayer {
            substrate_ref,
            thickness_nm,
            knots,
        } => {
            validate_ref(substrate_ref)?;
            if *thickness_nm == 0 {
                return Err(MaterialSubjectError::ZeroThickness);
            }
            if knots.len() < 2 {
                return Err(MaterialSubjectError::NeedGradientEndpoints);
            }
            if knots.first().map(|knot| knot.position_ppm) != Some(0)
                || knots.last().map(|knot| knot.position_ppm)
                    != Some(SUBJECT_FRACTION_PPM_TOTAL)
            {
                return Err(MaterialSubjectError::NeedGradientEndpoints);
            }
            let mut previous = None;
            for knot in knots {
                if knot.position_ppm > SUBJECT_FRACTION_PPM_TOTAL {
                    return Err(MaterialSubjectError::GradientPositionOutOfRange(
                        knot.position_ppm,
                    ));
                }
                if previous.is_some_and(|value| knot.position_ppm <= value) {
                    return Err(MaterialSubjectError::GradientPositionsNotStrictlyIncreasing);
                }
                knot.composition.validate()?;
                previous = Some(knot.position_ppm);
            }
            Ok(())
        }
        MaterialArchitecture::Porous {
            matrix_ref,
            porosity_ppm,
            characteristic_pore_nm,
        } => {
            validate_ref(matrix_ref)?;
            if *porosity_ppm >= SUBJECT_FRACTION_PPM_TOTAL {
                return Err(MaterialSubjectError::InvalidPorosity(*porosity_ppm));
            }
            if characteristic_pore_nm.is_some_and(|value| value == 0) {
                return Err(MaterialSubjectError::ZeroPoreScale);
            }
            Ok(())
        }
        MaterialArchitecture::DirectedInterface {
            side_a_ref,
            side_b_ref,
            interface_width_nm,
        } => {
            validate_ref(side_a_ref)?;
            validate_ref(side_b_ref)?;
            if side_a_ref == side_b_ref {
                return Err(MaterialSubjectError::IdenticalInterfaceSides);
            }
            if interface_width_nm.is_some_and(|value| value == 0) {
                return Err(MaterialSubjectError::ZeroInterfaceWidth);
            }
            Ok(())
        }
    }
}

fn validate_phase(phase: &MaterialPhaseSubject) -> Result<(), MaterialSubjectError> {
    validate_ref(&phase.phase_id)?;
    phase.composition.validate()?;
    match &phase.phase_state {
        MaterialPhaseState::Crystalline { structure_id } => validate_ref(structure_id),
        MaterialPhaseState::Mixed { phase_mixture_id } => validate_ref(phase_mixture_id),
        MaterialPhaseState::Amorphous | MaterialPhaseState::Liquid | MaterialPhaseState::Unknown => {
            Ok(())
        }
    }
}

fn validate_fraction_components(
    components: &[ElementFractionPpm],
) -> Result<(), MaterialSubjectError> {
    if components.is_empty() {
        return Err(MaterialSubjectError::EmptyComposition);
    }
    let mut seen = HashSet::new();
    let mut total = 0_u64;
    for component in components {
        if component.atomic_number == 0 {
            return Err(MaterialSubjectError::InvalidAtomicNumber(0));
        }
        if component.fraction_ppm == 0 {
            return Err(MaterialSubjectError::ZeroElementFraction(
                component.atomic_number,
            ));
        }
        if !seen.insert(component.atomic_number) {
            return Err(MaterialSubjectError::DuplicateElement(
                component.atomic_number,
            ));
        }
        total += u64::from(component.fraction_ppm);
    }
    if total != u64::from(SUBJECT_FRACTION_PPM_TOTAL) {
        return Err(MaterialSubjectError::FractionsDoNotSumToOne { sum: total });
    }
    Ok(())
}

fn validate_stoichiometry(
    components: &[ElementStoichiometry],
) -> Result<(), MaterialSubjectError> {
    if components.is_empty() {
        return Err(MaterialSubjectError::EmptyComposition);
    }
    let mut seen = HashSet::new();
    for component in components {
        if component.atomic_number == 0 {
            return Err(MaterialSubjectError::InvalidAtomicNumber(0));
        }
        if component.atom_count == 0 {
            return Err(MaterialSubjectError::ZeroStoichiometricCount(
                component.atomic_number,
            ));
        }
        if !seen.insert(component.atomic_number) {
            return Err(MaterialSubjectError::DuplicateElement(
                component.atomic_number,
            ));
        }
    }
    Ok(())
}

fn validate_lineage(lineage: &SubjectLineageRef) -> Result<(), MaterialSubjectError> {
    validate_ref(&lineage.lineage_id)?;
    if lineage.artifact_sha256.len() != SHA256_HEX_LEN
        || !lineage.artifact_sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(MaterialSubjectError::InvalidSha256);
    }
    Ok(())
}

fn validate_ref(value: &str) -> Result<(), MaterialSubjectError> {
    if value.trim().is_empty() {
        Err(MaterialSubjectError::EmptyReference)
    } else {
        Ok(())
    }
}

fn architecture_key(architecture: &MaterialArchitecture) -> String {
    match architecture {
        MaterialArchitecture::Bulk { phase } => format!("bulk({})", phase_key(phase)),
        MaterialArchitecture::Composite { constituents } => {
            let mut entries = constituents
                .iter()
                .map(|constituent| {
                    format!(
                        "{}={:06}",
                        encode_token(&constituent.material_ref),
                        constituent.fraction_ppm
                    )
                })
                .collect::<Vec<_>>();
            entries.sort();
            format!("composite[{}]", entries.join(","))
        }
        MaterialArchitecture::LayerStack {
            substrate_ref,
            layers,
        } => {
            let layers = layers
                .iter()
                .map(|layer| {
                    let process = layer
                        .process_lineage
                        .as_ref()
                        .map(lineage_key)
                        .unwrap_or_else(|| "none".to_string());
                    format!(
                        "{}@{}nm@{}",
                        encode_token(&layer.material_ref),
                        layer.thickness_nm,
                        process
                    )
                })
                .collect::<Vec<_>>()
                .join(">");
            format!(
                "layers(substrate={};{})",
                encode_token(substrate_ref),
                layers
            )
        }
        MaterialArchitecture::GradedLayer {
            substrate_ref,
            thickness_nm,
            knots,
        } => {
            let knots = knots
                .iter()
                .map(|knot| {
                    format!(
                        "{}:{}",
                        knot.position_ppm,
                        composition_key(&knot.composition)
                    )
                })
                .collect::<Vec<_>>()
                .join(">");
            format!(
                "graded(substrate={};thickness={}nm;{})",
                encode_token(substrate_ref),
                thickness_nm,
                knots
            )
        }
        MaterialArchitecture::Porous {
            matrix_ref,
            porosity_ppm,
            characteristic_pore_nm,
        } => format!(
            "porous(matrix={};porosity={:06};pore_nm={})",
            encode_token(matrix_ref),
            porosity_ppm,
            characteristic_pore_nm
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        ),
        MaterialArchitecture::DirectedInterface {
            side_a_ref,
            side_b_ref,
            interface_width_nm,
        } => format!(
            "interface(a={};b={};width_nm={})",
            encode_token(side_a_ref),
            encode_token(side_b_ref),
            interface_width_nm
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        ),
    }
}

fn phase_key(phase: &MaterialPhaseSubject) -> String {
    let state = match &phase.phase_state {
        MaterialPhaseState::Crystalline { structure_id } => {
            format!("crystal:{}", encode_token(structure_id))
        }
        MaterialPhaseState::Amorphous => "amorphous".to_string(),
        MaterialPhaseState::Liquid => "liquid".to_string(),
        MaterialPhaseState::Mixed { phase_mixture_id } => {
            format!("mixed:{}", encode_token(phase_mixture_id))
        }
        MaterialPhaseState::Unknown => "unknown".to_string(),
    };
    format!(
        "phase={};composition={};state={}",
        encode_token(&phase.phase_id),
        composition_key(&phase.composition),
        state
    )
}

fn composition_key(composition: &MaterialComposition) -> String {
    composition.canonical_key()
}

fn lineage_key(lineage: &SubjectLineageRef) -> String {
    format!(
        "{}:{}",
        encode_token(&lineage.lineage_id),
        lineage.artifact_sha256.to_ascii_lowercase()
    )
}

fn encode_token(value: &str) -> String {
    let mut output = String::new();
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.') {
            output.push(byte as char);
        } else {
            output.push_str(&format!("%{byte:02X}"));
        }
    }
    output
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a
}

/// Material-subject validation failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaterialSubjectError {
    /// Only schema version 1 is currently defined.
    UnsupportedSchemaVersion(u32),
    /// Human-facing label was empty.
    EmptyDisplayLabel,
    /// Composition contained no elements.
    EmptyComposition,
    /// Atomic number zero is invalid.
    InvalidAtomicNumber(u16),
    /// An atomic-fraction component was zero.
    ZeroElementFraction(u16),
    /// A stoichiometric coefficient was zero.
    ZeroStoichiometricCount(u16),
    /// The same element appeared more than once.
    DuplicateElement(u16),
    /// A ppm fraction set did not sum exactly to one million.
    FractionsDoNotSumToOne {
        /// Observed integer sum.
        sum: u64,
    },
    /// Stable/material reference was empty.
    EmptyReference,
    /// Lineage artifact digest was not 64 hexadecimal characters.
    InvalidSha256,
    /// Composite needs at least two constituents.
    NeedMultipleConstituents,
    /// Composite constituent fraction was zero.
    ZeroFractionConstituent,
    /// Composite repeated the same referenced material.
    DuplicateMaterialRef(String),
    /// Layer stack contained no layers.
    EmptyLayerStack,
    /// A layer or graded region had zero thickness.
    ZeroThickness,
    /// A graded layer must explicitly bind both endpoints.
    NeedGradientEndpoints,
    /// Gradient knot exceeded the normalized position range.
    GradientPositionOutOfRange(u32),
    /// Gradient knots were not strictly increasing.
    GradientPositionsNotStrictlyIncreasing,
    /// Porosity must be strictly less than 100%.
    InvalidPorosity(u32),
    /// Declared pore scale was zero.
    ZeroPoreScale,
    /// Directed interface used the same material on both sides.
    IdenticalInterfaceSides,
    /// Declared finite interface width was zero.
    ZeroInterfaceWidth,
}

#[cfg(test)]
mod tests {
    use super::*;

    const A64: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B64: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn process(hash: &str) -> SubjectLineageRef {
        SubjectLineageRef {
            lineage_id: "vacuum-arc-v1".to_string(),
            artifact_sha256: hash.to_string(),
        }
    }

    fn ti_nb() -> MaterialComposition {
        MaterialComposition::AtomicFractionPpm(
            AtomicCompositionPpm::new(vec![
                ElementFractionPpm {
                    atomic_number: 41,
                    fraction_ppm: 500_000,
                },
                ElementFractionPpm {
                    atomic_number: 22,
                    fraction_ppm: 500_000,
                },
            ])
            .unwrap(),
        )
    }

    fn bulk_subject(label: &str, process_hash: &str) -> MaterialSubject {
        MaterialSubject {
            schema_version: 1,
            display_label: label.to_string(),
            architecture: MaterialArchitecture::Bulk {
                phase: MaterialPhaseSubject {
                    phase_id: "ti-nb-bcc".to_string(),
                    composition: ti_nb(),
                    phase_state: MaterialPhaseState::Crystalline {
                        structure_id: "bcc".to_string(),
                    },
                },
            },
            process_lineage: process(process_hash),
            microstructure_lineage: None,
        }
    }

    #[test]
    fn atomic_composition_order_is_not_identity_bearing() {
        let a = AtomicCompositionPpm::new(vec![
            ElementFractionPpm {
                atomic_number: 22,
                fraction_ppm: 500_000,
            },
            ElementFractionPpm {
                atomic_number: 41,
                fraction_ppm: 500_000,
            },
        ])
        .unwrap();
        let b = AtomicCompositionPpm::new(vec![
            ElementFractionPpm {
                atomic_number: 41,
                fraction_ppm: 500_000,
            },
            ElementFractionPpm {
                atomic_number: 22,
                fraction_ppm: 500_000,
            },
        ])
        .unwrap();
        assert_eq!(a, b);
        assert_eq!(a.canonical_key(), b.canonical_key());
    }

    #[test]
    fn stoichiometry_reduces_formula_units_exactly() {
        let silica = StoichiometricComposition::new(vec![
            ElementStoichiometry {
                atomic_number: 8,
                atom_count: 4,
            },
            ElementStoichiometry {
                atomic_number: 14,
                atom_count: 2,
            },
        ])
        .unwrap();
        assert_eq!(silica.canonical_key(), "Z8=2,Z14=1");
    }

    #[test]
    fn display_name_does_not_change_scientific_identity() {
        let a = bulk_subject("candidate A", A64);
        let b = bulk_subject("renamed for publication", A64);
        assert_eq!(a.canonical_identity().unwrap(), b.canonical_identity().unwrap());
    }

    #[test]
    fn process_artifact_changes_subject_identity() {
        let a = bulk_subject("same chemistry", A64);
        let b = bulk_subject("same chemistry", B64);
        assert_ne!(a.canonical_identity().unwrap(), b.canonical_identity().unwrap());
    }

    #[test]
    fn microstructure_artifact_changes_subject_identity() {
        let mut a = bulk_subject("same chemistry", A64);
        let mut b = a.clone();
        a.microstructure_lineage = Some(SubjectLineageRef {
            lineage_id: "sem-ebsd-state".to_string(),
            artifact_sha256: A64.to_string(),
        });
        b.microstructure_lineage = Some(SubjectLineageRef {
            lineage_id: "sem-ebsd-state".to_string(),
            artifact_sha256: B64.to_string(),
        });
        assert_ne!(a.canonical_identity().unwrap(), b.canonical_identity().unwrap());
    }

    #[test]
    fn composite_constituent_order_is_not_identity_bearing() {
        let make = |constituents| MaterialSubject {
            schema_version: 1,
            display_label: "composite".to_string(),
            architecture: MaterialArchitecture::Composite { constituents },
            process_lineage: process(A64),
            microstructure_lineage: None,
        };
        let a = make(vec![
            CompositeConstituent {
                material_ref: "phase-a".to_string(),
                fraction_ppm: 700_000,
            },
            CompositeConstituent {
                material_ref: "phase-b".to_string(),
                fraction_ppm: 300_000,
            },
        ]);
        let b = make(vec![
            CompositeConstituent {
                material_ref: "phase-b".to_string(),
                fraction_ppm: 300_000,
            },
            CompositeConstituent {
                material_ref: "phase-a".to_string(),
                fraction_ppm: 700_000,
            },
        ]);
        assert_eq!(a.canonical_identity().unwrap(), b.canonical_identity().unwrap());
    }

    #[test]
    fn layer_order_and_thickness_are_identity_bearing() {
        let make = |layers| MaterialSubject {
            schema_version: 1,
            display_label: "fusion-facing stack".to_string(),
            architecture: MaterialArchitecture::LayerStack {
                substrate_ref: "substrate".to_string(),
                layers,
            },
            process_lineage: process(A64),
            microstructure_lineage: None,
        };
        let a = make(vec![
            MaterialLayer {
                material_ref: "rhea".to_string(),
                thickness_nm: 200_000,
                process_lineage: None,
            },
            MaterialLayer {
                material_ref: "carbide".to_string(),
                thickness_nm: 10_000,
                process_lineage: None,
            },
        ]);
        let b = make(vec![
            MaterialLayer {
                material_ref: "carbide".to_string(),
                thickness_nm: 10_000,
                process_lineage: None,
            },
            MaterialLayer {
                material_ref: "rhea".to_string(),
                thickness_nm: 200_000,
                process_lineage: None,
            },
        ]);
        let mut c = a.clone();
        if let MaterialArchitecture::LayerStack { layers, .. } = &mut c.architecture {
            layers[0].thickness_nm += 1;
        }
        assert_ne!(a.canonical_identity().unwrap(), b.canonical_identity().unwrap());
        assert_ne!(a.canonical_identity().unwrap(), c.canonical_identity().unwrap());
    }

    #[test]
    fn graded_layer_requires_bound_endpoints() {
        let subject = MaterialSubject {
            schema_version: 1,
            display_label: "bad gradient".to_string(),
            architecture: MaterialArchitecture::GradedLayer {
                substrate_ref: "substrate".to_string(),
                thickness_nm: 1_000,
                knots: vec![
                    CompositionGradientKnot {
                        position_ppm: 10,
                        composition: ti_nb(),
                    },
                    CompositionGradientKnot {
                        position_ppm: SUBJECT_FRACTION_PPM_TOTAL,
                        composition: ti_nb(),
                    },
                ],
            },
            process_lineage: process(A64),
            microstructure_lineage: None,
        };
        assert_eq!(
            subject.validate(),
            Err(MaterialSubjectError::NeedGradientEndpoints)
        );
    }

    #[test]
    fn directed_interface_orientation_is_identity_bearing() {
        let make = |a: &str, b: &str| MaterialSubject {
            schema_version: 1,
            display_label: "interface".to_string(),
            architecture: MaterialArchitecture::DirectedInterface {
                side_a_ref: a.to_string(),
                side_b_ref: b.to_string(),
                interface_width_nm: Some(2),
            },
            process_lineage: process(A64),
            microstructure_lineage: None,
        };
        assert_ne!(
            make("substrate", "film").canonical_identity().unwrap(),
            make("film", "substrate").canonical_identity().unwrap()
        );
    }

    #[test]
    fn fully_porous_subject_is_rejected() {
        let subject = MaterialSubject {
            schema_version: 1,
            display_label: "void".to_string(),
            architecture: MaterialArchitecture::Porous {
                matrix_ref: "matrix".to_string(),
                porosity_ppm: SUBJECT_FRACTION_PPM_TOTAL,
                characteristic_pore_nm: Some(100),
            },
            process_lineage: process(A64),
            microstructure_lineage: None,
        };
        assert_eq!(
            subject.validate(),
            Err(MaterialSubjectError::InvalidPorosity(
                SUBJECT_FRACTION_PPM_TOTAL
            ))
        );
    }
}
