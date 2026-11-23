// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Leakage-resistant, forward-time splits for longitudinal Mycelial morphogenesis.
//!
//! The split manifest commits the independent experimental unit and a forward
//! forecast origin/horizon. Every observation must appear exactly once as
//! context or future target. An independent unit cannot be assigned to more
//! than one partition.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::mycelial_morphogenesis_observation::MycelialMorphogenesisObservation;
use crate::mycelial_observation_coverage::mycelial_observation_id_inventory_digest;

pub const MYCELIAL_MORPHOGENESIS_SPLIT_SCHEMA_VERSION: u16 = 1;
pub const MAX_MYCELIAL_MORPHOGENESIS_SPLIT_UNITS: usize = 100_000;
pub const MAX_MYCELIAL_MORPHOGENESIS_SPLIT_OBSERVATIONS: usize = 1_000_000;

const MAX_SPLIT_IDENTIFIER_BYTES: usize = 128;
const MAX_SPLIT_STRING_BYTES: usize = 8 * 1024 * 1024;
const DIGEST_HEX_BYTES: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelialMorphogenesisSplitUnitBasis {
    BiologicalReplicate,
    CultivationUnit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelialMorphogenesisPartition {
    Train,
    Validation,
    Test,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MycelialMorphogenesisSplitUnit {
    /// The value selected from the declared split-unit basis.
    pub independent_unit_id: String,
    pub partition: MycelialMorphogenesisPartition,
    /// Context observations must have timestamps <= this cutoff.
    pub forecast_origin_ns: u64,
    /// Future targets must have origin < timestamp <= horizon_end_ns.
    pub horizon_end_ns: u64,
    /// Strictly sorted, disjoint and exhaustive with target IDs for this unit.
    pub context_observation_ids: Vec<String>,
    pub target_observation_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MycelialMorphogenesisSplitManifest {
    pub schema_version: u16,
    pub dataset_id: String,
    /// Commits each observation's source artifact, acquisition, extraction, ledger and unit.
    pub evidence_chain_inventory_digest: String,
    /// All observations are normalized into this coordinate frame and timebase.
    pub coordinate_frame_id: String,
    pub timebase_id: String,
    /// Digest over every morphogenesis observation ID in canonical sorted order.
    pub morphogenesis_observation_inventory_digest: String,
    pub split_unit_basis: MycelialMorphogenesisSplitUnitBasis,
    pub split_protocol_id: String,
    pub split_protocol_version: String,
    pub split_protocol_digest: String,
    /// Strictly sorted by independent_unit_id.
    pub units: Vec<MycelialMorphogenesisSplitUnit>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MycelialMorphogenesisSplitError {
    InvalidSchemaVersion(u16),
    InvalidIdentifier(&'static str),
    InvalidDigest(&'static str),
    EmptySplitInventory,
    ResourceLimitExceeded,
    NonCanonicalUnitOrder,
    DuplicateIndependentUnit(String),
    DuplicateObservationId(String),
    InvalidForecastWindow(String),
    EmptyContextOrTarget(String),
    NonCanonicalObservationOrder(String),
    SplitUnitInventoryMismatch,
    ObservationInventoryDigestMismatch,
    ObservationEvidenceMismatch(&'static str),
    ObservationValidationFailed(String),
    ObservationUnitMismatch(String),
    ContextAfterForecastOrigin(String),
    TargetOutsideForecastHorizon(String),
    ObservationAssignedMoreThanOnce(String),
    ObservationNotAssigned(String),
}

impl fmt::Display for MycelialMorphogenesisSplitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSchemaVersion(version) => {
                write!(f, "unsupported Mycelial morphogenesis split schema version {version}")
            }
            Self::InvalidIdentifier(field) => {
                write!(f, "invalid morphogenesis split identifier: {field}")
            }
            Self::InvalidDigest(field) => {
                write!(f, "invalid morphogenesis split evidence digest: {field}")
            }
            Self::EmptySplitInventory => {
                write!(f, "morphogenesis split manifest must contain independent units")
            }
            Self::ResourceLimitExceeded => {
                write!(f, "morphogenesis split manifest exceeds resource limits")
            }
            Self::NonCanonicalUnitOrder => {
                write!(f, "split units must be strictly sorted by independent unit ID")
            }
            Self::DuplicateIndependentUnit(id) => {
                write!(f, "independent unit {id} is assigned more than once")
            }
            Self::DuplicateObservationId(id) => {
                write!(f, "observation ID {id} appears more than once in split assignments")
            }
            Self::InvalidForecastWindow(id) => {
                write!(f, "forecast horizon is invalid for independent unit {id}")
            }
            Self::EmptyContextOrTarget(id) => {
                write!(f, "independent unit {id} requires context and future target observations")
            }
            Self::NonCanonicalObservationOrder(id) => {
                write!(f, "context/target observation IDs are not strictly sorted for unit {id}")
            }
            Self::SplitUnitInventoryMismatch => {
                write!(f, "split manifest does not exactly cover the supplied observation inventory")
            }
            Self::ObservationInventoryDigestMismatch => {
                write!(f, "morphogenesis observation inventory digest does not match split IDs")
            }
            Self::ObservationEvidenceMismatch(field) => {
                write!(f, "morphogenesis observations disagree on evidence binding: {field}")
            }
            Self::ObservationValidationFailed(id) => {
                write!(f, "morphogenesis observation {id} failed validation")
            }
            Self::ObservationUnitMismatch(id) => {
                write!(f, "observation {id} is assigned to the wrong independent unit")
            }
            Self::ContextAfterForecastOrigin(id) => {
                write!(f, "context observation {id} occurs after the forecast origin")
            }
            Self::TargetOutsideForecastHorizon(id) => {
                write!(f, "target observation {id} falls outside its forecast horizon")
            }
            Self::ObservationAssignedMoreThanOnce(id) => {
                write!(f, "observation {id} is assigned to multiple split roles")
            }
            Self::ObservationNotAssigned(id) => {
                write!(f, "observation {id} is absent from the split manifest")
            }
            Self::ObservationUnitCrossesPartitions(id) => {
                write!(f, "independent unit {id} crosses train/validation/test partitions")
            }
        }
    }
}

impl Error for MycelialMorphogenesisSplitError {}

fn validate_identifier(
    value: &str,
    field: &'static str,
) -> Result<(), MycelialMorphogenesisSplitError> {
    if value.trim().is_empty()
        || value.len() > MAX_SPLIT_IDENTIFIER_BYTES
        || value.chars().any(char::is_control)
    {
        return Err(MycelialMorphogenesisSplitError::InvalidIdentifier(field));
    }
    Ok(())
}

fn validate_digest(
    value: &str,
    field: &'static str,
) -> Result<(), MycelialMorphogenesisSplitError> {
    if value.len() != DIGEST_HEX_BYTES
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(MycelialMorphogenesisSplitError::InvalidDigest(field));
    }
    Ok(())
}

fn split_unit_id<'a>(
    observation: &'a MycelialMorphogenesisObservation,
    basis: MycelialMorphogenesisSplitUnitBasis,
) -> &'a str {
    match basis {
        MycelialMorphogenesisSplitUnitBasis::BiologicalReplicate => {
            &observation.biological_replicate_id
        }
        MycelialMorphogenesisSplitUnitBasis::CultivationUnit => &observation.cultivation_unit_id,
    }
}

fn partition_tag(partition: MycelialMorphogenesisPartition) -> u8 {
    match partition {
        MycelialMorphogenesisPartition::Train => 0,
        MycelialMorphogenesisPartition::Validation => 1,
        MycelialMorphogenesisPartition::Test => 2,
    }
}

fn split_basis_tag(basis: MycelialMorphogenesisSplitUnitBasis) -> u8 {
    match basis {
        MycelialMorphogenesisSplitUnitBasis::BiologicalReplicate => 0,
        MycelialMorphogenesisSplitUnitBasis::CultivationUnit => 1,
    }
}

/// Canonical evidence-chain inventory, ordered by observation ID. Longitudinal
/// datasets may span multiple raw artifacts and acquisition sessions.
pub fn mycelial_morphogenesis_evidence_chain_inventory_digest(
    observations: &[MycelialMorphogenesisObservation],
) -> Result<String, MycelialMorphogenesisSplitError> {
    if observations.is_empty()
        || observations.len() > MAX_MYCELIAL_MORPHOGENESIS_SPLIT_OBSERVATIONS
    {
        return Err(MycelialMorphogenesisSplitError::ResourceLimitExceeded);
    }
    let mut by_id = BTreeMap::new();
    for observation in observations {
        if by_id
            .insert(observation.observation_id.as_str(), observation)
            .is_some()
        {
            return Err(MycelialMorphogenesisSplitError::DuplicateObservationId(
                observation.observation_id.clone(),
            ));
        }
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-mycelial-morphogenesis-evidence-chain-inventory-v1");
    hasher.update(&(by_id.len() as u64).to_le_bytes());
    for (observation_id, observation) in by_id {
        for value in [
            observation_id,
            observation.source_artifact_id.as_str(),
            observation.source_artifact_digest.as_str(),
            observation.acquisition_domain_manifest_digest.as_str(),
            observation.extraction_coverage_receipt_digest.as_str(),
            observation.observation_coverage_ledger_digest.as_str(),
            observation.biological_replicate_id.as_str(),
            observation.cultivation_unit_id.as_str(),
            observation.acquisition_session_id.as_str(),
            observation.coordinate_frame_id.as_str(),
            observation.spatial_transform_digest.as_str(),
            observation.calibration_digest.as_str(),
            observation.timebase_id.as_str(),
        ] {
            hash_string(&mut hasher, value);
        }
        hasher.update(&observation.source_unit_key.tile_x.to_le_bytes());
        hasher.update(&observation.source_unit_key.tile_y.to_le_bytes());
        hasher.update(&observation.source_unit_key.z_plane.to_le_bytes());
        hasher.update(&observation.source_unit_key.channel.to_le_bytes());
        hasher.update(&observation.source_unit_key.timepoint.to_le_bytes());
        hasher.update(&observation.timestamp_ns.to_le_bytes());
    }
    Ok(hasher.finalize().to_hex().to_string())
}

fn hash_string(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

impl MycelialMorphogenesisSplitManifest {
    pub fn validate(&self) -> Result<(), MycelialMorphogenesisSplitError> {
        if self.schema_version != MYCELIAL_MORPHOGENESIS_SPLIT_SCHEMA_VERSION {
            return Err(MycelialMorphogenesisSplitError::InvalidSchemaVersion(
                self.schema_version,
            ));
        }
        for (value, field) in [
            (self.dataset_id.as_str(), "dataset_id"),
            (self.source_artifact_id.as_str(), "source_artifact_id"),
            (self.split_protocol_id.as_str(), "split_protocol_id"),
            (self.split_protocol_version.as_str(), "split_protocol_version"),
        ] {
            validate_identifier(value, field)?;
        }
        validate_identifier(&self.coordinate_frame_id, "coordinate_frame_id")?;
        validate_identifier(&self.timebase_id, "timebase_id")?;
        for (value, field) in [
            (&self.evidence_chain_inventory_digest, "evidence_chain_inventory_digest"),
            (
                &self.morphogenesis_observation_inventory_digest,
                "morphogenesis_observation_inventory_digest",
            ),
            (&self.split_protocol_digest, "split_protocol_digest"),
        ] {
            validate_digest(value, field)?;
        }
        if self.units.is_empty() {
            return Err(MycelialMorphogenesisSplitError::EmptySplitInventory);
        }
        if self.units.len() > MAX_MYCELIAL_MORPHOGENESIS_SPLIT_UNITS {
            return Err(MycelialMorphogenesisSplitError::ResourceLimitExceeded);
        }

        let mut total_string_bytes = 0usize;
        let mut previous_unit: Option<&str> = None;
        let mut observation_ids = BTreeSet::new();
        for unit in &self.units {
            validate_identifier(&unit.independent_unit_id, "independent_unit_id")?;
            total_string_bytes = total_string_bytes
                .saturating_add(unit.independent_unit_id.len());
            if let Some(previous) = previous_unit {
                if previous == unit.independent_unit_id {
                    return Err(MycelialMorphogenesisSplitError::DuplicateIndependentUnit(
                        unit.independent_unit_id.clone(),
                    ));
                }
                if previous > unit.independent_unit_id.as_str() {
                    return Err(MycelialMorphogenesisSplitError::NonCanonicalUnitOrder);
                }
            }
            previous_unit = Some(&unit.independent_unit_id);

            if unit.horizon_end_ns <= unit.forecast_origin_ns {
                return Err(MycelialMorphogenesisSplitError::InvalidForecastWindow(
                    unit.independent_unit_id.clone(),
                ));
            }
            if unit.context_observation_ids.is_empty() || unit.target_observation_ids.is_empty() {
                return Err(MycelialMorphogenesisSplitError::EmptyContextOrTarget(
                    unit.independent_unit_id.clone(),
                ));
            }
            if !unit
                .context_observation_ids
                .windows(2)
                .all(|pair| pair[0] < pair[1])
                || !unit
                    .target_observation_ids
                    .windows(2)
                    .all(|pair| pair[0] < pair[1])
            {
                return Err(MycelialMorphogenesisSplitError::NonCanonicalObservationOrder(
                    unit.independent_unit_id.clone(),
                ));
            }
            for observation_id in unit
                .context_observation_ids
                .iter()
                .chain(&unit.target_observation_ids)
            {
                validate_identifier(observation_id, "observation_id")?;
                total_string_bytes = total_string_bytes.saturating_add(observation_id.len());
                if !observation_ids.insert(observation_id.as_str()) {
                    return Err(MycelialMorphogenesisSplitError::DuplicateObservationId(
                        observation_id.clone(),
                    ));
                }
            }
        }
        if total_string_bytes > MAX_SPLIT_STRING_BYTES
            || observation_ids.len() > MAX_MYCELIAL_MORPHOGENESIS_SPLIT_OBSERVATIONS
        {
            return Err(MycelialMorphogenesisSplitError::ResourceLimitExceeded);
        }
        Ok(())
    }

    /// Validate the split against all supplied typed morphogenesis observations.
    ///
    /// This proves group disjointness and forward-time assignment relative to the
    /// declared observations. It does not prove the source dataset or experiment's
    /// independent-unit declaration is scientifically correct; that requires the
    /// paper/methods and raw sample manifest to support the chosen split-unit basis.
    pub fn validate_against(
        &self,
        observations: &[MycelialMorphogenesisObservation],
    ) -> Result<(), MycelialMorphogenesisSplitError> {
        self.validate()?;
        if observations.is_empty()
            || observations.len() > MAX_MYCELIAL_MORPHOGENESIS_SPLIT_OBSERVATIONS
        {
            return Err(MycelialMorphogenesisSplitError::ResourceLimitExceeded);
        }

        let mut observation_by_id = BTreeMap::new();
        let mut inventory_ids = Vec::with_capacity(observations.len());
        let first = &observations[0];
        for observation in observations {
            observation
                .validate()
                .map_err(|_| MycelialMorphogenesisSplitError::ObservationValidationFailed(
                    observation.observation_id.clone(),
                ))?;
            if observation.coordinate_frame_id != self.coordinate_frame_id
                || observation.timebase_id != self.timebase_id
            {
                return Err(MycelialMorphogenesisSplitError::ObservationEvidenceMismatch(
                    "normalized coordinate frame/timebase",
                ));
            }
            if observation_by_id
                .insert(observation.observation_id.as_str(), observation)
                .is_some()
            {
                return Err(MycelialMorphogenesisSplitError::DuplicateObservationId(
                    observation.observation_id.clone(),
                ));
            }
            inventory_ids.push(observation.observation_id.clone());
        }
        inventory_ids.sort();
        if mycelial_observation_id_inventory_digest(&inventory_ids)
            != self.morphogenesis_observation_inventory_digest
        {
            return Err(MycelialMorphogenesisSplitError::ObservationInventoryDigestMismatch);
        }
        if mycelial_morphogenesis_evidence_chain_inventory_digest(observations)?
            != self.evidence_chain_inventory_digest
        {
            return Err(MycelialMorphogenesisSplitError::ObservationEvidenceMismatch(
                "per-observation acquisition evidence-chain inventory",
            ));
        }
        if first.coordinate_frame_id != self.coordinate_frame_id
            || first.timebase_id != self.timebase_id
        {
            return Err(MycelialMorphogenesisSplitError::ObservationEvidenceMismatch(
                "split manifest coordinate frame/timebase",
            ));
        }

        let mut assigned_observations = BTreeSet::new();
        let mut assigned_units = BTreeMap::<String, MycelialMorphogenesisPartition>::new();
        for unit in &self.units {
            if assigned_units
                .insert(unit.independent_unit_id.clone(), unit.partition)
                .is_some()
            {
                return Err(MycelialMorphogenesisSplitError::DuplicateIndependentUnit(
                    unit.independent_unit_id.clone(),
                ));
            }
            for observation_id in &unit.context_observation_ids {
                let Some(observation) = observation_by_id.get(observation_id.as_str()) else {
                    return Err(MycelialMorphogenesisSplitError::SplitUnitInventoryMismatch);
                };
                if split_unit_id(observation, self.split_unit_basis) != unit.independent_unit_id {
                    return Err(MycelialMorphogenesisSplitError::ObservationUnitMismatch(
                        observation_id.clone(),
                    ));
                }
                if !assigned_observations.insert(observation_id.as_str()) {
                    return Err(MycelialMorphogenesisSplitError::ObservationAssignedMoreThanOnce(
                        observation_id.clone(),
                    ));
                }
                if observation.timestamp_ns > unit.forecast_origin_ns {
                    return Err(MycelialMorphogenesisSplitError::ContextAfterForecastOrigin(
                        observation_id.clone(),
                    ));
                }
            }
            for observation_id in &unit.target_observation_ids {
                let Some(observation) = observation_by_id.get(observation_id.as_str()) else {
                    return Err(MycelialMorphogenesisSplitError::SplitUnitInventoryMismatch);
                };
                if split_unit_id(observation, self.split_unit_basis) != unit.independent_unit_id {
                    return Err(MycelialMorphogenesisSplitError::ObservationUnitMismatch(
                        observation_id.clone(),
                    ));
                }
                if !assigned_observations.insert(observation_id.as_str()) {
                    return Err(MycelialMorphogenesisSplitError::ObservationAssignedMoreThanOnce(
                        observation_id.clone(),
                    ));
                }
                if observation.timestamp_ns <= unit.forecast_origin_ns
                    || observation.timestamp_ns > unit.horizon_end_ns
                {
                    return Err(MycelialMorphogenesisSplitError::TargetOutsideForecastHorizon(
                        observation_id.clone(),
                    ));
                }
            }
        }
        for (observation_id, observation) in &observation_by_id {
            if !assigned_observations.contains(observation_id) {
                return Err(MycelialMorphogenesisSplitError::ObservationNotAssigned(
                    (*observation_id).to_string(),
                ));
            }
            let unit_id = split_unit_id(observation, self.split_unit_basis);
            if !assigned_units.contains_key(unit_id) {
                return Err(MycelialMorphogenesisSplitError::ObservationUnitMismatch(
                    observation.observation_id.clone(),
                ));
            }
        }
        Ok(())
    }

    pub fn canonical_digest_hex(&self) -> Result<String, MycelialMorphogenesisSplitError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-mycelial-morphogenesis-split-manifest-v1");
        hasher.update(&self.schema_version.to_le_bytes());
        for value in [
            self.dataset_id.as_str(),
            self.evidence_chain_inventory_digest.as_str(),
            self.coordinate_frame_id.as_str(),
            self.timebase_id.as_str(),
            self.morphogenesis_observation_inventory_digest.as_str(),
            self.split_protocol_id.as_str(),
            self.split_protocol_version.as_str(),
            self.split_protocol_digest.as_str(),
        ] {
            hash_string(&mut hasher, value);
        }
        hasher.update(&[split_basis_tag(self.split_unit_basis)]);
        hasher.update(&(self.units.len() as u64).to_le_bytes());
        for unit in &self.units {
            hash_string(&mut hasher, &unit.independent_unit_id);
            hasher.update(&[partition_tag(unit.partition)]);
            hasher.update(&unit.forecast_origin_ns.to_le_bytes());
            hasher.update(&unit.horizon_end_ns.to_le_bytes());
            hasher.update(&(unit.context_observation_ids.len() as u64).to_le_bytes());
            for id in &unit.context_observation_ids {
                hash_string(&mut hasher, id);
            }
            hasher.update(&(unit.target_observation_ids.len() as u64).to_le_bytes());
            for id in &unit.target_observation_ids {
                hash_string(&mut hasher, id);
            }
        }
        Ok(hasher.finalize().to_hex().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mycelial_morphogenesis_observation::{
        MYCELIAL_MORPHOGENESIS_OBSERVATION_SCHEMA_VERSION, MycelialDevelopmentStage,
        MycelialMorphogenesisMeasurement, MycelialMorphogenesisObservation,
        MycelialMorphogenesisRegionKind,
    };
    use crate::mycelial_acquisition_domain::MycelialAcquisitionUnitKey;

    fn digest(character: char) -> String {
        std::iter::repeat_n(character, DIGEST_HEX_BYTES).collect()
    }

    fn observation(
        id: &str,
        replicate: &str,
        cultivation_unit: &str,
        timestamp_ns: u64,
    ) -> MycelialMorphogenesisObservation {
        MycelialMorphogenesisObservation {
            schema_version: MYCELIAL_MORPHOGENESIS_OBSERVATION_SCHEMA_VERSION,
            observation_id: id.to_string(),
            source_observation_digest: digest('1'),
            source_artifact_id: "agaricus-cordogenesis".to_string(),
            source_artifact_digest: digest('a'),
            acquisition_domain_manifest_digest: digest('b'),
            extraction_coverage_receipt_digest: digest('c'),
            observation_coverage_ledger_digest: digest('d'),
            biological_replicate_id: replicate.to_string(),
            cultivation_unit_id: cultivation_unit.to_string(),
            acquisition_session_id: "session-001".to_string(),
            substrate_region_id: "substrate-001".to_string(),
            region_kind: MycelialMorphogenesisRegionKind::CordSegment,
            source_unit_key: MycelialAcquisitionUnitKey {
                tile_x: 0,
                tile_y: 0,
                z_plane: 0,
                channel: 0,
                timepoint: u32::try_from(timestamp_ns).unwrap(),
            },
            timestamp_ns,
            timebase_id: "monotonic-ns-v1".to_string(),
            development_stage: MycelialDevelopmentStage::VegetativeColonization,
            coordinate_frame_id: "specimen-frame-v1".to_string(),
            spatial_transform_digest: digest('e'),
            calibration_digest: digest('f'),
            measurement: MycelialMorphogenesisMeasurement::CordAreaFraction {
                fraction: 0.25,
                analysis_method_id: "cord-segmentation-v1".to_string(),
            },
        }
    }

    fn observations() -> Vec<MycelialMorphogenesisObservation> {
        vec![
            observation("a-context", "rep-a", "box-a", 100),
            observation("a-target", "rep-a", "box-a", 200),
            observation("b-context", "rep-b", "box-b", 100),
            observation("b-target", "rep-b", "box-b", 200),
            observation("c-context", "rep-c", "box-c", 100),
            observation("c-target", "rep-c", "box-c", 200),
        ]
    }

    fn split_unit(
        id: &str,
        partition: MycelialMorphogenesisPartition,
    ) -> MycelialMorphogenesisSplitUnit {
        MycelialMorphogenesisSplitUnit {
            independent_unit_id: id.to_string(),
            partition,
            forecast_origin_ns: 150,
            horizon_end_ns: 250,
            context_observation_ids: vec![format!("{id}-context")],
            target_observation_ids: vec![format!("{id}-target")],
        }
    }

    fn manifest(
        source_observations: &[MycelialMorphogenesisObservation],
        basis: MycelialMorphogenesisSplitUnitBasis,
    ) -> MycelialMorphogenesisSplitManifest {
        let mut ids = source_observations
            .iter()
            .map(|observation| observation.observation_id.clone())
            .collect::<Vec<_>>();
        ids.sort();
        MycelialMorphogenesisSplitManifest {
            schema_version: MYCELIAL_MORPHOGENESIS_SPLIT_SCHEMA_VERSION,
            dataset_id: "agaricus-cordogenesis-2026".to_string(),
            evidence_chain_inventory_digest:
                mycelial_morphogenesis_evidence_chain_inventory_digest(source_observations).unwrap(),
            coordinate_frame_id: source_observations[0].coordinate_frame_id.clone(),
            timebase_id: source_observations[0].timebase_id.clone(),
            morphogenesis_observation_inventory_digest:
                mycelial_observation_id_inventory_digest(&ids),
            split_unit_basis: basis,
            split_protocol_id: "leave-independent-unit-out-forward-time".to_string(),
            split_protocol_version: "1.0.0".to_string(),
            split_protocol_digest: digest('9'),
            units: vec![
                split_unit("box-a", MycelialMorphogenesisPartition::Train),
                split_unit("box-b", MycelialMorphogenesisPartition::Validation),
                split_unit("box-c", MycelialMorphogenesisPartition::Test),
            ],
        }
    }

    #[test]
    fn replicate_disjoint_forward_time_manifest_validates() {
        let observations = observations();
        let manifest = manifest(&observations, MycelialMorphogenesisSplitUnitBasis::CultivationUnit);
        manifest.validate().unwrap();
        manifest.validate_against(&observations).unwrap();
        assert_eq!(manifest.canonical_digest_hex().unwrap().len(), 64);
    }

    #[test]
    fn context_observation_after_forecast_origin_is_rejected() {
        let mut observations = observations();
        observations[0].timestamp_ns = 175;
        let manifest = manifest(&observations, MycelialMorphogenesisSplitUnitBasis::CultivationUnit);
        assert!(matches!(
            manifest.validate_against(&observations),
            Err(MycelialMorphogenesisSplitError::ContextAfterForecastOrigin(_))
        ));
    }

    #[test]
    fn target_at_or_before_forecast_origin_is_rejected() {
        let mut observations = observations();
        observations[1].timestamp_ns = 150;
        let manifest = manifest(&observations, MycelialMorphogenesisSplitUnitBasis::CultivationUnit);
        assert!(matches!(
            manifest.validate_against(&observations),
            Err(MycelialMorphogenesisSplitError::TargetOutsideForecastHorizon(_))
        ));
    }

    #[test]
    fn target_after_horizon_is_rejected() {
        let mut observations = observations();
        observations[1].timestamp_ns = 300;
        let manifest = manifest(&observations, MycelialMorphogenesisSplitUnitBasis::CultivationUnit);
        assert!(matches!(
            manifest.validate_against(&observations),
            Err(MycelialMorphogenesisSplitError::TargetOutsideForecastHorizon(_))
        ));
    }

    #[test]
    fn split_manifest_must_assign_every_observation_once() {
        let observations = observations();
        let mut manifest = manifest(&observations, MycelialMorphogenesisSplitUnitBasis::CultivationUnit);
        manifest.units[2].target_observation_ids.clear();
        assert!(matches!(
            manifest.validate(),
            Err(MycelialMorphogenesisSplitError::EmptyContextOrTarget(_))
        ));
    }

    #[test]
    fn observation_assigned_to_more_than_one_role_is_rejected() {
        let observations = observations();
        let mut manifest = manifest(&observations, MycelialMorphogenesisSplitUnitBasis::CultivationUnit);
        manifest.units[0].target_observation_ids = vec!["a-context".to_string()];
        assert!(matches!(
            manifest.validate_against(&observations),
            Err(MycelialMorphogenesisSplitError::ObservationAssignedMoreThanOnce(_))
        ));
    }

    #[test]
    fn independent_unit_cannot_be_assigned_to_two_partitions() {
        let observations = observations();
        let mut manifest = manifest(&observations, MycelialMorphogenesisSplitUnitBasis::CultivationUnit);
        manifest.units = vec![
            split_unit("box-a", MycelialMorphogenesisPartition::Train),
            split_unit("box-a", MycelialMorphogenesisPartition::Validation),
        ];
        assert!(matches!(
            manifest.validate(),
            Err(MycelialMorphogenesisSplitError::DuplicateIndependentUnit(_))
        ));
    }

    #[test]
    fn biological_replicate_basis_is_supported() {
        let mut observations = observations();
        observations[0].biological_replicate_id = "replicate-a".to_string();
        observations[1].biological_replicate_id = "replicate-a".to_string();
        observations[2].biological_replicate_id = "replicate-b".to_string();
        observations[3].biological_replicate_id = "replicate-b".to_string();
        observations[4].biological_replicate_id = "replicate-c".to_string();
        observations[5].biological_replicate_id = "replicate-c".to_string();
        let mut manifest = manifest(&observations, MycelialMorphogenesisSplitUnitBasis::BiologicalReplicate);
        manifest.units = vec![
            MycelialMorphogenesisSplitUnit {
                independent_unit_id: "replicate-a".to_string(),
                partition: MycelialMorphogenesisPartition::Train,
                forecast_origin_ns: 150,
                horizon_end_ns: 250,
                context_observation_ids: vec!["a-context".to_string()],
                target_observation_ids: vec!["a-target".to_string()],
            },
            MycelialMorphogenesisSplitUnit {
                independent_unit_id: "replicate-b".to_string(),
                partition: MycelialMorphogenesisPartition::Validation,
                forecast_origin_ns: 150,
                horizon_end_ns: 250,
                context_observation_ids: vec!["b-context".to_string()],
                target_observation_ids: vec!["b-target".to_string()],
            },
            MycelialMorphogenesisSplitUnit {
                independent_unit_id: "replicate-c".to_string(),
                partition: MycelialMorphogenesisPartition::Test,
                forecast_origin_ns: 150,
                horizon_end_ns: 250,
                context_observation_ids: vec!["c-context".to_string()],
                target_observation_ids: vec!["c-target".to_string()],
            },
        ];
        manifest.validate_against(&observations).unwrap();
    }

    #[test]
    fn split_supports_multiple_source_artifacts_and_acquisition_sessions() {
        let mut observations = observations();
        observations[2].source_artifact_id = "source-session-b".to_string();
        observations[2].source_artifact_digest = digest('7');
        observations[2].acquisition_domain_manifest_digest = digest('8');
        observations[2].extraction_coverage_receipt_digest = digest('9');
        observations[2].observation_coverage_ledger_digest = digest('0');
        observations[2].acquisition_session_id = "session-002".to_string();
        observations[2].spatial_transform_digest = digest('6');
        observations[2].calibration_digest = digest('5');
        let mut manifest = manifest(
            &observations,
            MycelialMorphogenesisSplitUnitBasis::CultivationUnit,
        );
        manifest.evidence_chain_inventory_digest =
            mycelial_morphogenesis_evidence_chain_inventory_digest(&observations).unwrap();
        manifest.validate_against(&observations).unwrap();
    }

    #[test]
    fn source_evidence_mismatch_is_rejected() {
        let mut observations = observations();
        observations[5].calibration_digest = digest('7');
        let manifest = manifest(&observations, MycelialMorphogenesisSplitUnitBasis::CultivationUnit);
        assert!(matches!(
            manifest.validate_against(&observations),
            Err(MycelialMorphogenesisSplitError::ObservationEvidenceMismatch(_))
        ));
    }

    #[test]
    fn observation_not_assigned_to_any_partition_is_rejected() {
        let mut observations = observations();
        observations.push(observation("a-target-2", "rep-a", "box-a", 225));
        let manifest = manifest(
            &observations,
            MycelialMorphogenesisSplitUnitBasis::CultivationUnit,
        );
        assert!(matches!(
            manifest.validate_against(&observations),
            Err(MycelialMorphogenesisSplitError::ObservationNotAssigned(_))
        ));
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let observations = observations();
        let manifest = manifest(&observations, MycelialMorphogenesisSplitUnitBasis::CultivationUnit);
        let mut value = serde_json::to_value(manifest).unwrap();
        value["unbound_split_policy"] = serde_json::json!("random frames");
        assert!(serde_json::from_value::<MycelialMorphogenesisSplitManifest>(value).is_err());
    }
}

