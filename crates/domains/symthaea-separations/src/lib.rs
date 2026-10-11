// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-bounded selective-separation and recovery semantics.
//!
//! This crate defines identity and planning/observation context for exact feeds,
//! separator architectures, and operating states. It does not establish that a
//! separation occurred, that a mechanism is correct, or that a process is useful.

#![deny(unsafe_code)]
#![warn(missing_docs)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;

const FEED_DOMAIN: &[u8] = b"symthaea-separations-feed-v1\0";
const ARCHITECTURE_DOMAIN: &[u8] = b"symthaea-separations-architecture-v1\0";
const OPERATING_DOMAIN: &[u8] = b"symthaea-separations-operating-state-v1\0";

/// Deterministic identity for one separation subject.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SeparationId([u8; 32]);

impl SeparationId {
    /// Returns the raw 32-byte identity.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for SeparationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in &self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// Validation failure for a separation contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeparationError {
    /// A required string field was empty.
    EmptyField(&'static str),
    /// A required collection was empty.
    EmptyCollection(&'static str),
    /// A set-like string collection contained a duplicate.
    DuplicateReference(&'static str),
    /// The same species appeared more than once in one exact feed state.
    DuplicateSpecies(String),
    /// A feed condition kind appeared more than once.
    DuplicateFeedCondition(FeedConditionKind),
    /// A driving-force kind appeared more than once.
    DuplicateDrivingForce(DrivingForceKind),
}

impl fmt::Display for SeparationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "required field `{field}` is empty"),
            Self::EmptyCollection(field) => write!(f, "required collection `{field}` is empty"),
            Self::DuplicateReference(field) => {
                write!(f, "set-like collection `{field}` contains a duplicate")
            }
            Self::DuplicateSpecies(species) => {
                write!(f, "feed contains duplicate species `{species}`")
            }
            Self::DuplicateFeedCondition(kind) => {
                write!(f, "feed condition `{kind:?}` appears more than once")
            }
            Self::DuplicateDrivingForce(kind) => {
                write!(f, "driving force `{kind:?}` appears more than once")
            }
        }
    }
}

impl std::error::Error for SeparationError {}

/// Broad physical phase of an exact feed state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeedPhase {
    /// Predominantly liquid feed.
    Liquid,
    /// Predominantly gas feed.
    Gas,
    /// Predominantly solid feed.
    Solid,
    /// Explicitly multiphase feed.
    Multiphase,
}

/// Kind of feed condition referenced by a feed-state subject.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeedConditionKind {
    /// Acidity/basicity or equivalent pH state.
    Ph,
    /// Temperature state.
    Temperature,
    /// Pressure state.
    Pressure,
    /// Solvent or water-content state.
    SolventState,
    /// Ionic-strength state.
    IonicStrength,
    /// Redox state.
    RedoxState,
    /// Suspended-solids state.
    SuspendedSolids,
    /// Organic/foulant loading state.
    OrganicLoad,
    /// Flow or hydrodynamic feed state.
    FlowHydrodynamics,
}

/// Broad separator/process mechanism class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeparationMechanismClass {
    /// Dense membrane transport.
    DenseMembrane,
    /// Porous membrane transport.
    PorousMembrane,
    /// Ion-exchange membrane transport.
    IonExchangeMembrane,
    /// Solid-electrolyte membrane transport.
    SolidElectrolyteMembrane,
    /// Sorption or adsorption on a material.
    AdsorbentOrSorbent,
    /// Ion-sieve mechanism.
    IonSieve,
    /// Electrochemical insertion/intercalation mechanism.
    ElectrochemicalIntercalation,
    /// Solvent-extraction mechanism.
    SolventExtraction,
    /// Precipitation or crystallization mechanism.
    PrecipitationOrCrystallization,
    /// Electrodialysis or related electromembrane mechanism.
    Electrodialysis,
    /// Explicit composition of multiple separation stages/mechanisms.
    HybridTrain,
    /// A separately identified mechanism not represented by another variant.
    OtherExplicit,
}

/// Kind of driving-force profile applied during a separation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DrivingForceKind {
    /// Pressure difference or gradient.
    Pressure,
    /// Applied electric potential.
    ElectricPotential,
    /// Applied electric current/current-density profile.
    ElectricCurrent,
    /// Concentration gradient.
    ConcentrationGradient,
    /// Chemical-potential or activity gradient.
    ChemicalPotential,
    /// Thermal gradient.
    ThermalGradient,
    /// Another explicitly profiled driving force.
    OtherDeclared,
}

/// One constituent in an exact feed state.
///
/// `state_ref` must identify the exact concentration/activity/speciation state
/// supplied by a stronger measurement or data layer. A friendly species label is
/// not itself quantitative evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeedConstituentV1 {
    species_id: String,
    state_ref: String,
    evidence_refs: Vec<String>,
}

impl FeedConstituentV1 {
    /// Builds a validated constituent reference.
    pub fn new(
        species_id: impl Into<String>,
        state_ref: impl Into<String>,
        evidence_refs: Vec<String>,
    ) -> Result<Self, SeparationError> {
        let value = Self {
            species_id: clean(species_id.into(), "species_id")?,
            state_ref: clean(state_ref.into(), "state_ref")?,
            evidence_refs: clean_set(evidence_refs, "evidence_refs", false)?,
        };
        value.validate()?;
        Ok(value)
    }

    /// Returns the exact species identifier used for feed uniqueness.
    pub fn species_id(&self) -> &str {
        &self.species_id
    }

    /// Validates the constituent reference.
    pub fn validate(&self) -> Result<(), SeparationError> {
        check(&self.species_id, "species_id")?;
        check(&self.state_ref, "state_ref")?;
        check_set(&self.evidence_refs, "evidence_refs", false)
    }
}

/// One exact feed-condition reference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeedConditionRefV1 {
    kind: FeedConditionKind,
    state_ref: String,
}

impl FeedConditionRefV1 {
    /// Builds a feed-condition reference.
    pub fn new(
        kind: FeedConditionKind,
        state_ref: impl Into<String>,
    ) -> Result<Self, SeparationError> {
        Ok(Self {
            kind,
            state_ref: clean(state_ref.into(), "feed_condition_state_ref")?,
        })
    }

    /// Returns the condition kind.
    pub fn kind(&self) -> FeedConditionKind {
        self.kind
    }

    /// Validates the condition reference.
    pub fn validate(&self) -> Result<(), SeparationError> {
        check(&self.state_ref, "feed_condition_state_ref")
    }
}

/// Exact source-stream/feed subject for a separation question.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeparationFeedStateV1 {
    phase: FeedPhase,
    source_ref: String,
    constituents: Vec<FeedConstituentV1>,
    conditions: Vec<FeedConditionRefV1>,
    evidence_refs: Vec<String>,
}

impl SeparationFeedStateV1 {
    /// Builds a validated exact feed state.
    pub fn new(
        phase: FeedPhase,
        source_ref: impl Into<String>,
        constituents: Vec<FeedConstituentV1>,
        conditions: Vec<FeedConditionRefV1>,
        evidence_refs: Vec<String>,
    ) -> Result<Self, SeparationError> {
        let value = Self {
            phase,
            source_ref: clean(source_ref.into(), "source_ref")?,
            constituents,
            conditions,
            evidence_refs: clean_set(evidence_refs, "evidence_refs", false)?,
        };
        value.validate()?;
        Ok(value)
    }

    /// Validates unique species, unique condition kinds, and exact references.
    pub fn validate(&self) -> Result<(), SeparationError> {
        check(&self.source_ref, "source_ref")?;
        if self.constituents.is_empty() {
            return Err(SeparationError::EmptyCollection("constituents"));
        }
        let mut species = BTreeSet::new();
        for constituent in &self.constituents {
            constituent.validate()?;
            let canonical = constituent.species_id.trim().to_owned();
            if !species.insert(canonical.clone()) {
                return Err(SeparationError::DuplicateSpecies(canonical));
            }
        }
        let mut condition_kinds = BTreeSet::new();
        for condition in &self.conditions {
            condition.validate()?;
            if !condition_kinds.insert(condition.kind()) {
                return Err(SeparationError::DuplicateFeedCondition(condition.kind()));
            }
        }
        check_set(&self.evidence_refs, "evidence_refs", false)
    }

    /// Computes an order-invariant identity over species and condition sets.
    pub fn id(&self) -> Result<SeparationId, SeparationError> {
        self.validate()?;
        let mut h = blake3::Hasher::new();
        h.update(FEED_DOMAIN);
        h.update(&[feed_phase_code(self.phase)]);
        put_str(&mut h, self.source_ref.trim());

        let mut constituents = self.constituents.iter().collect::<Vec<_>>();
        constituents.sort_by(|a, b| a.species_id.cmp(&b.species_id));
        put_u64(&mut h, constituents.len() as u64);
        for constituent in constituents {
            put_str(&mut h, constituent.species_id.trim());
            put_str(&mut h, constituent.state_ref.trim());
            put_set(&mut h, &constituent.evidence_refs);
        }

        let mut conditions = self.conditions.iter().collect::<Vec<_>>();
        conditions.sort_by_key(|condition| condition.kind());
        put_u64(&mut h, conditions.len() as u64);
        for condition in conditions {
            h.update(&[feed_condition_code(condition.kind)]);
            put_str(&mut h, condition.state_ref.trim());
        }
        put_set(&mut h, &self.evidence_refs);
        Ok(finish(h))
    }
}

/// Material/process architecture under which separation is attempted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeparationArchitectureV1 {
    mechanism: SeparationMechanismClass,
    mechanism_profile_ref: String,
    material_subject_refs: Vec<String>,
    geometry_profile_ref: String,
    process_configuration_ref: String,
    regeneration_profile_ref: Option<String>,
    evidence_refs: Vec<String>,
}

impl SeparationArchitectureV1 {
    /// Builds a validated separator/process architecture.
    pub fn new(
        mechanism: SeparationMechanismClass,
        mechanism_profile_ref: impl Into<String>,
        material_subject_refs: Vec<String>,
        geometry_profile_ref: impl Into<String>,
        process_configuration_ref: impl Into<String>,
        regeneration_profile_ref: Option<String>,
        evidence_refs: Vec<String>,
    ) -> Result<Self, SeparationError> {
        let value = Self {
            mechanism,
            mechanism_profile_ref: clean(mechanism_profile_ref.into(), "mechanism_profile_ref")?,
            material_subject_refs: clean_set(
                material_subject_refs,
                "material_subject_refs",
                false,
            )?,
            geometry_profile_ref: clean(geometry_profile_ref.into(), "geometry_profile_ref")?,
            process_configuration_ref: clean(
                process_configuration_ref.into(),
                "process_configuration_ref",
            )?,
            regeneration_profile_ref: clean_optional(
                regeneration_profile_ref,
                "regeneration_profile_ref",
            )?,
            evidence_refs: clean_set(evidence_refs, "evidence_refs", false)?,
        };
        value.validate()?;
        Ok(value)
    }

    /// Validates architecture identity inputs.
    pub fn validate(&self) -> Result<(), SeparationError> {
        check(&self.mechanism_profile_ref, "mechanism_profile_ref")?;
        check_set(&self.material_subject_refs, "material_subject_refs", false)?;
        check(&self.geometry_profile_ref, "geometry_profile_ref")?;
        check(&self.process_configuration_ref, "process_configuration_ref")?;
        if let Some(profile) = &self.regeneration_profile_ref {
            check(profile, "regeneration_profile_ref")?;
        }
        check_set(&self.evidence_refs, "evidence_refs", false)
    }

    /// Computes deterministic architecture identity.
    pub fn id(&self) -> Result<SeparationId, SeparationError> {
        self.validate()?;
        let mut h = blake3::Hasher::new();
        h.update(ARCHITECTURE_DOMAIN);
        h.update(&[mechanism_code(self.mechanism)]);
        put_str(&mut h, self.mechanism_profile_ref.trim());
        put_set(&mut h, &self.material_subject_refs);
        put_str(&mut h, self.geometry_profile_ref.trim());
        put_str(&mut h, self.process_configuration_ref.trim());
        put_optional_str(&mut h, self.regeneration_profile_ref.as_deref());
        put_set(&mut h, &self.evidence_refs);
        Ok(finish(h))
    }
}

/// One exact applied driving-force profile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DrivingForceRefV1 {
    kind: DrivingForceKind,
    profile_ref: String,
}

impl DrivingForceRefV1 {
    /// Builds a driving-force reference.
    pub fn new(
        kind: DrivingForceKind,
        profile_ref: impl Into<String>,
    ) -> Result<Self, SeparationError> {
        Ok(Self {
            kind,
            profile_ref: clean(profile_ref.into(), "driving_force_profile_ref")?,
        })
    }

    /// Returns the driving-force class.
    pub fn kind(&self) -> DrivingForceKind {
        self.kind
    }

    /// Validates the driving-force reference.
    pub fn validate(&self) -> Result<(), SeparationError> {
        check(&self.profile_ref, "driving_force_profile_ref")
    }
}

/// Exact operating state for a feed/architecture pair.
///
/// Pretreatment and post-treatment steps are ordered because process order can
/// change the scientific question. Driving forces are set-like by kind.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeparationOperatingStateV1 {
    feed_id: SeparationId,
    architecture_id: SeparationId,
    driving_forces: Vec<DrivingForceRefV1>,
    flow_profile_ref: Option<String>,
    contact_time_profile_ref: String,
    environment_refs: Vec<String>,
    pretreatment_steps: Vec<String>,
    posttreatment_steps: Vec<String>,
    cycle_state_ref: Option<String>,
}

impl SeparationOperatingStateV1 {
    /// Builds a validated operating state.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        feed_id: SeparationId,
        architecture_id: SeparationId,
        driving_forces: Vec<DrivingForceRefV1>,
        flow_profile_ref: Option<String>,
        contact_time_profile_ref: impl Into<String>,
        environment_refs: Vec<String>,
        pretreatment_steps: Vec<String>,
        posttreatment_steps: Vec<String>,
        cycle_state_ref: Option<String>,
    ) -> Result<Self, SeparationError> {
        let value = Self {
            feed_id,
            architecture_id,
            driving_forces,
            flow_profile_ref: clean_optional(flow_profile_ref, "flow_profile_ref")?,
            contact_time_profile_ref: clean(
                contact_time_profile_ref.into(),
                "contact_time_profile_ref",
            )?,
            environment_refs: clean_set(environment_refs, "environment_refs", true)?,
            pretreatment_steps: clean_sequence(pretreatment_steps, "pretreatment_steps")?,
            posttreatment_steps: clean_sequence(posttreatment_steps, "posttreatment_steps")?,
            cycle_state_ref: clean_optional(cycle_state_ref, "cycle_state_ref")?,
        };
        value.validate()?;
        Ok(value)
    }

    /// Validates operating-state references and unique driving-force kinds.
    pub fn validate(&self) -> Result<(), SeparationError> {
        if self.driving_forces.is_empty() {
            return Err(SeparationError::EmptyCollection("driving_forces"));
        }
        let mut kinds = BTreeSet::new();
        for force in &self.driving_forces {
            force.validate()?;
            if !kinds.insert(force.kind()) {
                return Err(SeparationError::DuplicateDrivingForce(force.kind()));
            }
        }
        if let Some(flow) = &self.flow_profile_ref {
            check(flow, "flow_profile_ref")?;
        }
        check(&self.contact_time_profile_ref, "contact_time_profile_ref")?;
        check_set(&self.environment_refs, "environment_refs", true)?;
        check_sequence(&self.pretreatment_steps, "pretreatment_steps")?;
        check_sequence(&self.posttreatment_steps, "posttreatment_steps")?;
        if let Some(cycle) = &self.cycle_state_ref {
            check(cycle, "cycle_state_ref")?;
        }
        Ok(())
    }

    /// Computes deterministic operating-state identity.
    pub fn id(&self) -> Result<SeparationId, SeparationError> {
        self.validate()?;
        let mut h = blake3::Hasher::new();
        h.update(OPERATING_DOMAIN);
        h.update(self.feed_id.as_bytes());
        h.update(self.architecture_id.as_bytes());

        let mut forces = self.driving_forces.iter().collect::<Vec<_>>();
        forces.sort_by_key(|force| force.kind());
        put_u64(&mut h, forces.len() as u64);
        for force in forces {
            h.update(&[driving_force_code(force.kind)]);
            put_str(&mut h, force.profile_ref.trim());
        }

        put_optional_str(&mut h, self.flow_profile_ref.as_deref());
        put_str(&mut h, self.contact_time_profile_ref.trim());
        put_set(&mut h, &self.environment_refs);
        put_sequence(&mut h, &self.pretreatment_steps);
        put_sequence(&mut h, &self.posttreatment_steps);
        put_optional_str(&mut h, self.cycle_state_ref.as_deref());
        Ok(finish(h))
    }
}

fn clean(value: String, field: &'static str) -> Result<String, SeparationError> {
    let value = value.trim();
    if value.is_empty() {
        return Err(SeparationError::EmptyField(field));
    }
    Ok(value.to_owned())
}

fn check(value: &str, field: &'static str) -> Result<(), SeparationError> {
    if value.trim().is_empty() {
        Err(SeparationError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn clean_optional(
    value: Option<String>,
    field: &'static str,
) -> Result<Option<String>, SeparationError> {
    value.map(|value| clean(value, field)).transpose()
}

fn clean_set(
    values: Vec<String>,
    field: &'static str,
    allow_empty: bool,
) -> Result<Vec<String>, SeparationError> {
    if values.is_empty() && !allow_empty {
        return Err(SeparationError::EmptyCollection(field));
    }
    let mut values = values
        .into_iter()
        .map(|value| clean(value, field))
        .collect::<Result<Vec<_>, _>>()?;
    values.sort();
    if values.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(SeparationError::DuplicateReference(field));
    }
    Ok(values)
}

fn check_set(
    values: &[String],
    field: &'static str,
    allow_empty: bool,
) -> Result<(), SeparationError> {
    if values.is_empty() && !allow_empty {
        return Err(SeparationError::EmptyCollection(field));
    }
    let mut seen = BTreeSet::new();
    for value in values {
        check(value, field)?;
        if !seen.insert(value.trim()) {
            return Err(SeparationError::DuplicateReference(field));
        }
    }
    Ok(())
}

fn clean_sequence(
    values: Vec<String>,
    field: &'static str,
) -> Result<Vec<String>, SeparationError> {
    values
        .into_iter()
        .map(|value| clean(value, field))
        .collect()
}

fn check_sequence(values: &[String], field: &'static str) -> Result<(), SeparationError> {
    for value in values {
        check(value, field)?;
    }
    Ok(())
}

fn put_u64(h: &mut blake3::Hasher, value: u64) {
    h.update(&value.to_le_bytes());
}

fn put_str(h: &mut blake3::Hasher, value: &str) {
    put_u64(h, value.len() as u64);
    h.update(value.as_bytes());
}

fn put_set(h: &mut blake3::Hasher, values: &[String]) {
    let mut values = values.iter().map(|value| value.trim()).collect::<Vec<_>>();
    values.sort_unstable();
    put_u64(h, values.len() as u64);
    for value in values {
        put_str(h, value);
    }
}

fn put_sequence(h: &mut blake3::Hasher, values: &[String]) {
    put_u64(h, values.len() as u64);
    for value in values {
        put_str(h, value.trim());
    }
}

fn put_optional_str(h: &mut blake3::Hasher, value: Option<&str>) {
    match value {
        Some(value) => {
            h.update(&[1]);
            put_str(h, value.trim());
        }
        None => {
            h.update(&[0]);
        }
    }
}

fn feed_phase_code(value: FeedPhase) -> u8 {
    match value {
        FeedPhase::Liquid => 0,
        FeedPhase::Gas => 1,
        FeedPhase::Solid => 2,
        FeedPhase::Multiphase => 3,
    }
}

fn feed_condition_code(value: FeedConditionKind) -> u8 {
    match value {
        FeedConditionKind::Ph => 0,
        FeedConditionKind::Temperature => 1,
        FeedConditionKind::Pressure => 2,
        FeedConditionKind::SolventState => 3,
        FeedConditionKind::IonicStrength => 4,
        FeedConditionKind::RedoxState => 5,
        FeedConditionKind::SuspendedSolids => 6,
        FeedConditionKind::OrganicLoad => 7,
        FeedConditionKind::FlowHydrodynamics => 8,
    }
}

fn mechanism_code(value: SeparationMechanismClass) -> u8 {
    match value {
        SeparationMechanismClass::DenseMembrane => 0,
        SeparationMechanismClass::PorousMembrane => 1,
        SeparationMechanismClass::IonExchangeMembrane => 2,
        SeparationMechanismClass::SolidElectrolyteMembrane => 3,
        SeparationMechanismClass::AdsorbentOrSorbent => 4,
        SeparationMechanismClass::IonSieve => 5,
        SeparationMechanismClass::ElectrochemicalIntercalation => 6,
        SeparationMechanismClass::SolventExtraction => 7,
        SeparationMechanismClass::PrecipitationOrCrystallization => 8,
        SeparationMechanismClass::Electrodialysis => 9,
        SeparationMechanismClass::HybridTrain => 10,
        SeparationMechanismClass::OtherExplicit => 11,
    }
}

fn driving_force_code(value: DrivingForceKind) -> u8 {
    match value {
        DrivingForceKind::Pressure => 0,
        DrivingForceKind::ElectricPotential => 1,
        DrivingForceKind::ElectricCurrent => 2,
        DrivingForceKind::ConcentrationGradient => 3,
        DrivingForceKind::ChemicalPotential => 4,
        DrivingForceKind::ThermalGradient => 5,
        DrivingForceKind::OtherDeclared => 6,
    }
}

fn finish(h: blake3::Hasher) -> SeparationId {
    SeparationId(*h.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn constituent(species: &str, state: &str) -> FeedConstituentV1 {
        FeedConstituentV1::new(
            species,
            state,
            vec![format!("assay:{species}:{state}")],
        )
        .unwrap()
    }

    fn feed(constituents: Vec<FeedConstituentV1>) -> SeparationFeedStateV1 {
        SeparationFeedStateV1::new(
            FeedPhase::Liquid,
            "feed:brine-a",
            constituents,
            vec![
                FeedConditionRefV1::new(FeedConditionKind::Ph, "ph:7.2").unwrap(),
                FeedConditionRefV1::new(FeedConditionKind::Temperature, "temp:298k").unwrap(),
            ],
            vec!["feed-assay:001".into()],
        )
        .unwrap()
    }

    fn architecture(mechanism: SeparationMechanismClass) -> SeparationArchitectureV1 {
        SeparationArchitectureV1::new(
            mechanism,
            "mechanism-profile:v1",
            vec!["mat:active-layer".into(), "mat:support".into()],
            "geometry:membrane-v1",
            "process:assembly-v1",
            Some("regen:v1".into()),
            vec!["architecture-source:001".into()],
        )
        .unwrap()
    }

    #[test]
    fn constituent_order_does_not_change_feed_identity() {
        let a = feed(vec![constituent("Li+", "li:1"), constituent("Mg2+", "mg:20")]);
        let b = feed(vec![constituent("Mg2+", "mg:20"), constituent("Li+", "li:1")]);
        assert_eq!(a.id().unwrap(), b.id().unwrap());
    }

    #[test]
    fn changed_constituent_state_changes_feed_identity() {
        let a = feed(vec![constituent("Li+", "li:1"), constituent("Mg2+", "mg:20")]);
        let b = feed(vec![constituent("Li+", "li:2"), constituent("Mg2+", "mg:20")]);
        assert_ne!(a.id().unwrap(), b.id().unwrap());
    }

    #[test]
    fn duplicate_species_is_rejected() {
        let result = SeparationFeedStateV1::new(
            FeedPhase::Liquid,
            "feed:bad",
            vec![constituent("Li+", "li:1"), constituent("Li+", "li:2")],
            Vec::new(),
            vec!["source:bad".into()],
        );
        assert!(matches!(result, Err(SeparationError::DuplicateSpecies(_))));
    }

    #[test]
    fn changed_condition_changes_feed_identity() {
        let constituents = vec![constituent("Li+", "li:1")];
        let a = SeparationFeedStateV1::new(
            FeedPhase::Liquid,
            "feed:a",
            constituents.clone(),
            vec![FeedConditionRefV1::new(FeedConditionKind::Ph, "ph:6").unwrap()],
            vec!["source:a".into()],
        )
        .unwrap();
        let b = SeparationFeedStateV1::new(
            FeedPhase::Liquid,
            "feed:a",
            constituents,
            vec![FeedConditionRefV1::new(FeedConditionKind::Ph, "ph:8").unwrap()],
            vec!["source:a".into()],
        )
        .unwrap();
        assert_ne!(a.id().unwrap(), b.id().unwrap());
    }

    #[test]
    fn mechanism_changes_architecture_identity() {
        let a = architecture(SeparationMechanismClass::SolidElectrolyteMembrane);
        let b = architecture(SeparationMechanismClass::AdsorbentOrSorbent);
        assert_ne!(a.id().unwrap(), b.id().unwrap());
    }

    #[test]
    fn driving_force_order_is_set_like() {
        let feed_id = feed(vec![constituent("Li+", "li:1")]).id().unwrap();
        let architecture_id = architecture(SeparationMechanismClass::IonSieve).id().unwrap();
        let pressure = DrivingForceRefV1::new(DrivingForceKind::Pressure, "pressure:1bar").unwrap();
        let gradient = DrivingForceRefV1::new(
            DrivingForceKind::ConcentrationGradient,
            "gradient:feed-product",
        )
        .unwrap();
        let a = SeparationOperatingStateV1::new(
            feed_id,
            architecture_id,
            vec![pressure.clone(), gradient.clone()],
            Some("flow:crossflow".into()),
            "contact:1h",
            vec!["ambient:lab".into()],
            vec!["pretreat:filter".into()],
            Vec::new(),
            None,
        )
        .unwrap();
        let b = SeparationOperatingStateV1::new(
            feed_id,
            architecture_id,
            vec![gradient, pressure],
            Some("flow:crossflow".into()),
            "contact:1h",
            vec!["ambient:lab".into()],
            vec!["pretreat:filter".into()],
            Vec::new(),
            None,
        )
        .unwrap();
        assert_eq!(a.id().unwrap(), b.id().unwrap());
    }

    #[test]
    fn pretreatment_order_is_semantic() {
        let feed_id = feed(vec![constituent("Li+", "li:1")]).id().unwrap();
        let architecture_id = architecture(SeparationMechanismClass::IonSieve).id().unwrap();
        let force = DrivingForceRefV1::new(
            DrivingForceKind::ConcentrationGradient,
            "gradient:feed-product",
        )
        .unwrap();
        let a = SeparationOperatingStateV1::new(
            feed_id,
            architecture_id,
            vec![force.clone()],
            None,
            "contact:1h",
            Vec::new(),
            vec!["pretreat:filter".into(), "pretreat:ph-adjust".into()],
            Vec::new(),
            None,
        )
        .unwrap();
        let b = SeparationOperatingStateV1::new(
            feed_id,
            architecture_id,
            vec![force],
            None,
            "contact:1h",
            Vec::new(),
            vec!["pretreat:ph-adjust".into(), "pretreat:filter".into()],
            Vec::new(),
            None,
        )
        .unwrap();
        assert_ne!(a.id().unwrap(), b.id().unwrap());
    }

    #[test]
    fn duplicate_driving_force_kind_is_rejected() {
        let feed_id = feed(vec![constituent("Li+", "li:1")]).id().unwrap();
        let architecture_id = architecture(SeparationMechanismClass::IonSieve).id().unwrap();
        let result = SeparationOperatingStateV1::new(
            feed_id,
            architecture_id,
            vec![
                DrivingForceRefV1::new(DrivingForceKind::Pressure, "pressure:a").unwrap(),
                DrivingForceRefV1::new(DrivingForceKind::Pressure, "pressure:b").unwrap(),
            ],
            None,
            "contact:1h",
            Vec::new(),
            Vec::new(),
            Vec::new(),
            None,
        );
        assert!(matches!(
            result,
            Err(SeparationError::DuplicateDrivingForce(DrivingForceKind::Pressure))
        ));
    }

    #[test]
    fn serde_round_trip_preserves_feed_identity() {
        let feed = feed(vec![constituent("Li+", "li:1"), constituent("Mg2+", "mg:20")]);
        let encoded = serde_json::to_string(&feed).unwrap();
        let decoded: SeparationFeedStateV1 = serde_json::from_str(&encoded).unwrap();
        assert_eq!(feed.id().unwrap(), decoded.id().unwrap());
    }
}
