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
    pub source_artifact_id: String,
    pub source_artifact_digest: String,
    pub acquisition_domain_manifest_digest: String,
    pub extraction_coverage_receipt_digest: String,
    pub observation_coverage_ledger_digest: String,
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
    ObservationUnitCrossesPartitions(String),
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
        for (value, field) in [
            (&self.source_artifact_digest, "source_artifact_digest"),
            (
                &self.acquisition_domain_manifest_digest,
                "acquisition_domain_manifest_digest",
            ),
            (
                &self.extraction_coverage_receipt_digest,
                "extraction_coverage_receipt_digest",
            ),
            (
                &self.observation_coverage_ledger_digest,
                "observation_coverage_ledger_digest",
            ),
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
            if previous_unit.is_some_and(|previous| previous >= unit.independent_unit_id.as_str()) {
                return Err(MycelialMorphogenesisSplitError::NonCanonicalUnitOrder);
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
            if observation.source_artifact_id != first.source_artifact_id
                || observation.source_artifact_digest != first.source_artifact_digest
            {
                return Err(MycelialMorphogenesisSplitError::ObservationEvidenceMismatch(
                    "source artifact",
                ));
            }
            if observation.acquisition_domain_manifest_digest
                != first.acquisition_domain_manifest_digest
                || observation.extraction_coverage_receipt_digest
                    != first.extraction_coverage_receipt_digest
                || observation.observation_coverage_ledger_digest
                    != first.observation_coverage_ledger_digest
            {
                return Err(MycelialMorphogenesisSplitError::ObservationEvidenceMismatch(
                    "acquisition/extraction/observation provenance",
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
        if first.source_artifact_id != self.source_artifact_id
            || first.source_artifact_digest != self.source_artifact_digest
            || first.acquisition_domain_manifest_digest != self.acquisition_domain_manifest_digest
            || first.extraction_coverage_receipt_digest != self.extraction_coverage_receipt_digest
            || first.observation_coverage_ledger_digest != self.observation_coverage_ledger_digest
        {
            return Err(MycelialMorphogenesisSplitError::ObservationEvidenceMismatch(
                "split manifest source digests",
            ));
        }

        let mut assigned_observations = BTreeSet::new();
        let mut assigned_units = BTreeMap::new();
        for unit in &self.units {
            if assigned_units
                .insert(&unit.independent_unit_id, unit.partition)
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
                if observation.timestamp_ns > unit.forecast_origin_ns {
                    return Err(MycelialMorphogenesisSplitError::ContextAfterForecastOrigin(
                        observation_id.clone(),
                    ));
                }
                if !assigned_observations.insert(observation_id.as_str()) {
                    return Err(MycelialMorphogenesisSplitError::ObservationAssignedMoreThanOnce(
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
                if observation.timestamp_ns <= unit.forecast_origin_ns
                    || observation.timestamp_ns > unit.horizon_end_ns
                {
                    return Err(MycelialMorphogenesisSplitError::TargetOutsideForecastHorizon(
                        observation_id.clone(),
                    ));
                }
                if !assigned_observations.insert(observation_id.as_str()) {
                    return Err(MycelialMorphogenesisSplitError::ObservationAssignedMoreThanOnce(
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
            self.source_artifact_id.as_str(),
            self.source_artifact_digest.as_str(),
            self.acquisition_domain_manifest_digest.as_str(),
            self.extraction_coverage_receipt_digest.as_str(),
            self.observation_coverage_ledger_digest.as_str(),
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

