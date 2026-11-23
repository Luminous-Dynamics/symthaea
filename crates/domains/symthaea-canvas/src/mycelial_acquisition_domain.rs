// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Source-domain completeness and extraction receipts for Mycelial microscopy.
//!
//! This module proves exact accounting against a declared acquisition domain.
//! It does not prove that the source instrument metadata is truthful, that the
//! declared acquisition scope is the entire experiment, or that an extractor
//! has perfect biological recall. Those claims require independent provenance,
//! calibration, and benchmark qualification.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::mycelial_observation_coverage::{
    MycelialObservationCoverageLedger, mycelial_observation_id_inventory_digest,
};

pub const MYCELIAL_ACQUISITION_DOMAIN_SCHEMA_VERSION: u16 = 1;
pub const MYCELIAL_EXTRACTION_RECEIPT_SCHEMA_VERSION: u16 = 1;
pub const MAX_MYCELIAL_ACQUISITION_DOMAIN_UNITS: usize = 1_000_000;

const MAX_DOMAIN_IDENTIFIER_BYTES: usize = 128;
const MAX_DOMAIN_VERSION_BYTES: usize = 64;
const MAX_DOMAIN_REASON_BYTES: usize = 512;
const DIGEST_HEX_BYTES: usize = 64;
const MAX_CANONICAL_DOMAIN_STRING_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelialAcquisitionAxisName {
    X,
    Y,
    Z,
    Channel,
    Time,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelialSourceRightsStatus {
    VerifiedForProcessing,
    AuthorPermissionRequired,
    Restricted,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MycelialAcquisitionAxis {
    pub name: MycelialAcquisitionAxisName,
    pub length: u32,
    pub unit: String,
    /// Coordinate origin for index zero in the declared coordinate frame.
    pub origin: f64,
    /// Physical units per source-axis index. None means uncalibrated/unknown.
    pub physical_scale: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MycelialAcquisitionDomainSpec {
    pub width_pixels: u32,
    pub height_pixels: u32,
    pub z_planes: u32,
    pub channels: u32,
    pub timepoints: u32,
    pub tile_width_pixels: u32,
    pub tile_height_pixels: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MycelialAcquisitionUnitKey {
    pub tile_x: u32,
    pub tile_y: u32,
    pub z_plane: u32,
    pub channel: u32,
    pub timepoint: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MycelialAcquisitionScopeExclusion {
    pub unit: MycelialAcquisitionUnitKey,
    /// Human-readable reason why this source unit is outside the declared scope.
    pub reason: String,
    /// Digest of the independently approved scope/exclusion decision.
    pub scope_decision_digest: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MycelialAcquisitionDomainManifest {
    pub schema_version: u16,
    pub source_artifact_id: String,
    pub source_artifact_digest: String,
    pub source_rights_status: MycelialSourceRightsStatus,
    /// Digest of the license/permission evidence supporting the rights status.
    pub rights_evidence_digest: String,
    pub source_format: String,
    pub acquisition_session_id: String,
    pub biological_replicate_id: String,
    pub coordinate_frame_id: String,
    /// Digest of the exact acquisition scope/ROI mask and declared exclusions context.
    pub declared_scope_digest: String,
    /// Digest of instrument, firmware and acquisition settings metadata.
    pub instrument_metadata_digest: String,
    /// Original source order is retained; dimensions are checked by axis identity.
    pub axes: Vec<MycelialAcquisitionAxis>,
    pub domain: MycelialAcquisitionDomainSpec,
    pub spatial_transform_digest: String,
    pub calibration_digest: String,
    pub manifest_producer_id: String,
    pub manifest_producer_version: String,
    pub manifest_producer_digest: String,
    /// Excluded units remain visible and require a separate decision digest.
    pub scope_exclusions: Vec<MycelialAcquisitionScopeExclusion>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelialExtractionUnitState {
    Processed,
    UnavailableOrCorrupted,
    Unmeasurable,
    ProcessingFailed,
    Pending,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MycelialExtractionUnitOutcome {
    pub unit: MycelialAcquisitionUnitKey,
    /// Digest of the exact source subregion represented by this tile/plane/channel/timepoint.
    pub source_unit_digest: String,
    pub state: MycelialExtractionUnitState,
    /// Present only for successfully processed units, including zero-observation outputs.
    pub output_observation_inventory_digest: Option<String>,
    pub output_observation_count: Option<u64>,
    /// Required for every non-processed disposition; never interpreted as a negative label.
    pub disposition_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MycelialExtractionCoverageReceipt {
    pub schema_version: u16,
    pub acquisition_domain_manifest_digest: String,
    pub extractor_id: String,
    pub extractor_version: String,
    pub extractor_config_digest: String,
    /// Exactly one ordered outcome for every in-scope unit in the manifest.
    pub units: Vec<MycelialExtractionUnitOutcome>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelialExtractionCoverageStatus {
    CompleteFullAcquisitionDomain,
    CompleteDeclaredScopeWithExclusions,
    Partial,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MycelialExtractionCoverageSummary {
    pub status: MycelialExtractionCoverageStatus,
    pub total_domain_units: usize,
    pub in_scope_units: usize,
    pub excluded_units: usize,
    pub processed_units: usize,
    pub unavailable_or_corrupted_units: usize,
    pub unmeasurable_units: usize,
    pub processing_failed_units: usize,
    pub pending_units: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MycelialAcquisitionDomainError {
    InvalidSchemaVersion(u16),
    InvalidIdentifier(&'static str),
    InvalidDigest(&'static str),
    SourceRightsNotCleared(MycelialSourceRightsStatus),
    InvalidDomainDimensions,
    DomainResourceLimitExceeded(usize),
    InvalidAxisLayout,
    DuplicateAxis(MycelialAcquisitionAxisName),
    AxisDimensionMismatch(MycelialAcquisitionAxisName),
    InvalidAxisMetadata(&'static str),
    NonCanonicalExclusionOrder,
    ExcludedUnitOutsideDomain,
    EmptyDeclaredScope,
    InvalidDispositionReason,
    ManifestDigestMismatch,
    ReceiptUnitInventoryMismatch,
    IncompleteExtractionBlocksObservationInventory,
    ObservationLedgerInvalid,
    ObservationSourceBindingMismatch,
    ObservationSourceUnitMissing(String),
    ObservationSourceUnitOutsideScope(String),
    ObservationUnitInventoryCountMismatch(MycelialAcquisitionUnitKey),
    ObservationUnitInventoryDigestMismatch(MycelialAcquisitionUnitKey),
    InvalidUnitDisposition { reason: &'static str },
    InvalidExclusionReason,
    DigestResourceLimitExceeded,
}

impl fmt::Display for MycelialAcquisitionDomainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSchemaVersion(version) => {
                write!(
                    f,
                    "unsupported Mycelial acquisition schema version {version}"
                )
            }
            Self::InvalidIdentifier(field) => {
                write!(f, "invalid or unbounded acquisition identifier: {field}")
            }
            Self::InvalidDigest(field) => {
                write!(f, "invalid acquisition evidence digest: {field}")
            }
            Self::SourceRightsNotCleared(status) => {
                write!(
                    f,
                    "source artifact is not cleared for processing: {status:?}"
                )
            }
            Self::InvalidDomainDimensions => {
                write!(
                    f,
                    "acquisition dimensions and tile dimensions must be nonzero"
                )
            }
            Self::DomainResourceLimitExceeded(count) => write!(
                f,
                "acquisition domain unit count {count} exceeds {MAX_MYCELIAL_ACQUISITION_DOMAIN_UNITS}"
            ),
            Self::InvalidAxisLayout => {
                write!(
                    f,
                    "acquisition axes must contain X, Y, Z, Channel and Time exactly once"
                )
            }
            Self::DuplicateAxis(axis) => {
                write!(f, "duplicate acquisition axis {axis:?}")
            }
            Self::AxisDimensionMismatch(axis) => {
                write!(
                    f,
                    "acquisition axis dimension does not match domain spec for {axis:?}"
                )
            }
            Self::InvalidAxisMetadata(field) => {
                write!(f, "invalid acquisition axis metadata: {field}")
            }
            Self::NonCanonicalExclusionOrder => {
                write!(
                    f,
                    "scope exclusions must be strictly sorted by acquisition unit"
                )
            }
            Self::ExcludedUnitOutsideDomain => {
                write!(
                    f,
                    "scope exclusion refers to a unit outside the acquisition domain"
                )
            }
            Self::EmptyDeclaredScope => {
                write!(
                    f,
                    "acquisition manifest must retain at least one in-scope unit"
                )
            }
            Self::InvalidDispositionReason => {
                write!(
                    f,
                    "acquisition disposition requires a bounded non-empty reason"
                )
            }
            Self::InvalidExclusionReason => {
                write!(
                    f,
                    "scope exclusion requires a bounded reason and decision digest"
                )
            }
            Self::ManifestDigestMismatch => {
                write!(
                    f,
                    "extraction receipt does not bind to the supplied acquisition manifest"
                )
            }
            Self::ReceiptUnitInventoryMismatch => {
                write!(
                    f,
                    "extraction receipt does not exactly cover the in-scope unit inventory"
                )
            }
            Self::IncompleteExtractionBlocksObservationInventory => {
                write!(
                    f,
                    "partial extraction cannot qualify a complete observation inventory"
                )
            }
            Self::ObservationLedgerInvalid => {
                write!(f, "observation coverage ledger failed semantic validation")
            }
            Self::ObservationSourceBindingMismatch => {
                write!(
                    f,
                    "observation ledger source, replicate or coordinate bindings disagree with acquisition manifest"
                )
            }
            Self::ObservationSourceUnitMissing(id) => {
                write!(f, "observation {id} has no acquisition-unit binding")
            }
            Self::ObservationSourceUnitOutsideScope(id) => {
                write!(
                    f,
                    "observation {id} references an acquisition unit outside the declared scope"
                )
            }
            Self::ObservationUnitInventoryCountMismatch(unit) => {
                write!(
                    f,
                    "observation count for acquisition unit {unit:?} disagrees with extraction receipt"
                )
            }
            Self::ObservationUnitInventoryDigestMismatch(unit) => {
                write!(
                    f,
                    "observation ID digest for acquisition unit {unit:?} disagrees with extraction receipt"
                )
            }
            Self::InvalidUnitDisposition { reason } => {
                write!(f, "invalid extraction unit disposition: {reason}")
            }
            Self::DigestResourceLimitExceeded => {
                write!(
                    f,
                    "acquisition artifact exceeds canonical digest resource limits"
                )
            }
        }
    }
}

impl Error for MycelialAcquisitionDomainError {}

fn validate_identifier(
    value: &str,
    field: &'static str,
    maximum_bytes: usize,
) -> Result<(), MycelialAcquisitionDomainError> {
    if value.trim().is_empty() || value.len() > maximum_bytes || value.chars().any(char::is_control)
    {
        return Err(MycelialAcquisitionDomainError::InvalidIdentifier(field));
    }
    Ok(())
}

fn reason_is_valid(value: &str) -> bool {
    !value.trim().is_empty()
        && value.len() <= MAX_DOMAIN_REASON_BYTES
        && !value.chars().any(char::is_control)
}

fn validate_digest(value: &str, field: &'static str) -> Result<(), MycelialAcquisitionDomainError> {
    if value.len() != DIGEST_HEX_BYTES
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(MycelialAcquisitionDomainError::InvalidDigest(field));
    }
    Ok(())
}

fn hash_string(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn hash_unit_key(hasher: &mut blake3::Hasher, unit: MycelialAcquisitionUnitKey) {
    hasher.update(&unit.tile_x.to_le_bytes());
    hasher.update(&unit.tile_y.to_le_bytes());
    hasher.update(&unit.z_plane.to_le_bytes());
    hasher.update(&unit.channel.to_le_bytes());
    hasher.update(&unit.timepoint.to_le_bytes());
}

fn axis_tag(axis: MycelialAcquisitionAxisName) -> u8 {
    match axis {
        MycelialAcquisitionAxisName::X => 0,
        MycelialAcquisitionAxisName::Y => 1,
        MycelialAcquisitionAxisName::Z => 2,
        MycelialAcquisitionAxisName::Channel => 3,
        MycelialAcquisitionAxisName::Time => 4,
    }
}

fn rights_status_tag(status: MycelialSourceRightsStatus) -> u8 {
    match status {
        MycelialSourceRightsStatus::VerifiedForProcessing => 0,
        MycelialSourceRightsStatus::AuthorPermissionRequired => 1,
        MycelialSourceRightsStatus::Restricted => 2,
        MycelialSourceRightsStatus::Unknown => 3,
    }
}

fn extraction_state_tag(state: MycelialExtractionUnitState) -> u8 {
    match state {
        MycelialExtractionUnitState::Processed => 0,
        MycelialExtractionUnitState::UnavailableOrCorrupted => 1,
        MycelialExtractionUnitState::Unmeasurable => 2,
        MycelialExtractionUnitState::ProcessingFailed => 3,
        MycelialExtractionUnitState::Pending => 4,
    }
}

impl MycelialAcquisitionDomainSpec {
    pub fn expected_unit_count(&self) -> Result<usize, MycelialAcquisitionDomainError> {
        if self.width_pixels == 0
            || self.height_pixels == 0
            || self.z_planes == 0
            || self.channels == 0
            || self.timepoints == 0
            || self.tile_width_pixels == 0
            || self.tile_height_pixels == 0
        {
            return Err(MycelialAcquisitionDomainError::InvalidDomainDimensions);
        }

        let tile_columns = self.width_pixels.div_ceil(self.tile_width_pixels) as usize;
        let tile_rows = self.height_pixels.div_ceil(self.tile_height_pixels) as usize;
        let count = tile_columns
            .checked_mul(tile_rows)
            .and_then(|value| value.checked_mul(self.z_planes as usize))
            .and_then(|value| value.checked_mul(self.channels as usize))
            .and_then(|value| value.checked_mul(self.timepoints as usize))
            .ok_or(MycelialAcquisitionDomainError::DomainResourceLimitExceeded(
                usize::MAX,
            ))?;

        if count > MAX_MYCELIAL_ACQUISITION_DOMAIN_UNITS {
            return Err(MycelialAcquisitionDomainError::DomainResourceLimitExceeded(
                count,
            ));
        }
        Ok(count)
    }

    pub fn expected_unit_keys(
        &self,
    ) -> Result<Vec<MycelialAcquisitionUnitKey>, MycelialAcquisitionDomainError> {
        let count = self.expected_unit_count()?;
        let tile_columns = self.width_pixels.div_ceil(self.tile_width_pixels);
        let tile_rows = self.height_pixels.div_ceil(self.tile_height_pixels);
        let mut units = Vec::with_capacity(count);

        for tile_x in 0..tile_columns {
            for tile_y in 0..tile_rows {
                for z_plane in 0..self.z_planes {
                    for channel in 0..self.channels {
                        for timepoint in 0..self.timepoints {
                            units.push(MycelialAcquisitionUnitKey {
                                tile_x,
                                tile_y,
                                z_plane,
                                channel,
                                timepoint,
                            });
                        }
                    }
                }
            }
        }
        units.sort_unstable();
        Ok(units)
    }
}

impl MycelialAcquisitionDomainManifest {
    /// Validate the raw-source identity, axis dimensions and explicitly excluded units.
    ///
    /// This proves completeness only relative to this declared manifest. An upstream
    /// instrument/acquisition attestation is still required to qualify the truth and
    /// completeness of the raw artifact metadata and declared experiment scope.
    pub fn validate(&self) -> Result<(), MycelialAcquisitionDomainError> {
        if self.schema_version != MYCELIAL_ACQUISITION_DOMAIN_SCHEMA_VERSION {
            return Err(MycelialAcquisitionDomainError::InvalidSchemaVersion(
                self.schema_version,
            ));
        }

        validate_identifier(
            &self.source_artifact_id,
            "source_artifact_id",
            MAX_DOMAIN_IDENTIFIER_BYTES,
        )?;
        validate_digest(&self.source_artifact_digest, "source_artifact_digest")?;
        validate_digest(&self.rights_evidence_digest, "rights_evidence_digest")?;
        if self.source_rights_status != MycelialSourceRightsStatus::VerifiedForProcessing {
            return Err(MycelialAcquisitionDomainError::SourceRightsNotCleared(
                self.source_rights_status,
            ));
        }
        validate_identifier(
            &self.source_format,
            "source_format",
            MAX_DOMAIN_VERSION_BYTES,
        )?;
        validate_identifier(
            &self.acquisition_session_id,
            "acquisition_session_id",
            MAX_DOMAIN_IDENTIFIER_BYTES,
        )?;
        validate_identifier(
            &self.biological_replicate_id,
            "biological_replicate_id",
            MAX_DOMAIN_IDENTIFIER_BYTES,
        )?;
        validate_identifier(
            &self.coordinate_frame_id,
            "coordinate_frame_id",
            MAX_DOMAIN_IDENTIFIER_BYTES,
        )?;
        validate_digest(&self.declared_scope_digest, "declared_scope_digest")?;
        validate_digest(
            &self.instrument_metadata_digest,
            "instrument_metadata_digest",
        )?;
        validate_digest(&self.spatial_transform_digest, "spatial_transform_digest")?;
        validate_digest(&self.calibration_digest, "calibration_digest")?;
        validate_identifier(
            &self.manifest_producer_id,
            "manifest_producer_id",
            MAX_DOMAIN_IDENTIFIER_BYTES,
        )?;
        validate_identifier(
            &self.manifest_producer_version,
            "manifest_producer_version",
            MAX_DOMAIN_VERSION_BYTES,
        )?;
        validate_digest(&self.manifest_producer_digest, "manifest_producer_digest")?;

        self.domain.expected_unit_count()?;

        if self.axes.len() != 5 {
            return Err(MycelialAcquisitionDomainError::InvalidAxisLayout);
        }
        let mut seen_axes = BTreeSet::new();
        for axis in &self.axes {
            if !seen_axes.insert(axis.name) {
                return Err(MycelialAcquisitionDomainError::DuplicateAxis(axis.name));
            }
            validate_identifier(&axis.unit, "axis_unit", MAX_DOMAIN_VERSION_BYTES)?;
            if axis.length == 0
                || !axis.origin.is_finite()
                || axis
                    .physical_scale
                    .is_some_and(|scale| !scale.is_finite() || scale <= 0.0)
            {
                return Err(MycelialAcquisitionDomainError::InvalidAxisMetadata(
                    "length or physical_scale",
                ));
            }
            let expected_length = match axis.name {
                MycelialAcquisitionAxisName::X => self.domain.width_pixels,
                MycelialAcquisitionAxisName::Y => self.domain.height_pixels,
                MycelialAcquisitionAxisName::Z => self.domain.z_planes,
                MycelialAcquisitionAxisName::Channel => self.domain.channels,
                MycelialAcquisitionAxisName::Time => self.domain.timepoints,
            };
            if axis.length != expected_length {
                return Err(MycelialAcquisitionDomainError::AxisDimensionMismatch(
                    axis.name,
                ));
            }
        }
        if seen_axes.len() != 5 {
            return Err(MycelialAcquisitionDomainError::InvalidAxisLayout);
        }

        let expected_units = self.domain.expected_unit_keys()?;
        if self.scope_exclusions.len() >= expected_units.len() {
            return Err(MycelialAcquisitionDomainError::EmptyDeclaredScope);
        }
        for pair in self.scope_exclusions.windows(2) {
            if pair[0].unit >= pair[1].unit {
                return Err(MycelialAcquisitionDomainError::NonCanonicalExclusionOrder);
            }
        }
        for exclusion in &self.scope_exclusions {
            if expected_units.binary_search(&exclusion.unit).is_err() {
                return Err(MycelialAcquisitionDomainError::ExcludedUnitOutsideDomain);
            }
            if !reason_is_valid(&exclusion.reason) {
                return Err(MycelialAcquisitionDomainError::InvalidExclusionReason);
            }
            validate_digest(&exclusion.scope_decision_digest, "scope_decision_digest")?;
        }
        Ok(())
    }

    pub fn in_scope_unit_keys(
        &self,
    ) -> Result<Vec<MycelialAcquisitionUnitKey>, MycelialAcquisitionDomainError> {
        self.validate()?;
        let excluded: BTreeSet<_> = self
            .scope_exclusions
            .iter()
            .map(|exclusion| exclusion.unit)
            .collect();
        Ok(self
            .domain
            .expected_unit_keys()?
            .into_iter()
            .filter(|unit| !excluded.contains(unit))
            .collect())
    }

    /// Hash the exact manifest fields independently from semantic validation.
    /// Invalid/tampered manifests still receive a distinct digest when bounded.
    pub fn canonical_digest_hex(&self) -> Result<String, MycelialAcquisitionDomainError> {
        if self.axes.len() > 5
            || self.scope_exclusions.len() > MAX_MYCELIAL_ACQUISITION_DOMAIN_UNITS
        {
            return Err(MycelialAcquisitionDomainError::DigestResourceLimitExceeded);
        }
        let mut string_bytes = 0usize;
        let mut account = |value: &str| {
            string_bytes = string_bytes.saturating_add(value.len());
            string_bytes <= MAX_CANONICAL_DOMAIN_STRING_BYTES
        };
        for value in [
            self.source_artifact_id.as_str(),
            self.source_artifact_digest.as_str(),
            self.rights_evidence_digest.as_str(),
            self.source_format.as_str(),
            self.acquisition_session_id.as_str(),
            self.biological_replicate_id.as_str(),
            self.coordinate_frame_id.as_str(),
            self.declared_scope_digest.as_str(),
            self.instrument_metadata_digest.as_str(),
            self.spatial_transform_digest.as_str(),
            self.calibration_digest.as_str(),
            self.manifest_producer_id.as_str(),
            self.manifest_producer_version.as_str(),
            self.manifest_producer_digest.as_str(),
        ] {
            if !account(value) {
                return Err(MycelialAcquisitionDomainError::DigestResourceLimitExceeded);
            }
        }
        for axis in &self.axes {
            if !account(&axis.unit) {
                return Err(MycelialAcquisitionDomainError::DigestResourceLimitExceeded);
            }
        }
        for exclusion in &self.scope_exclusions {
            if !account(&exclusion.reason) || !account(&exclusion.scope_decision_digest) {
                return Err(MycelialAcquisitionDomainError::DigestResourceLimitExceeded);
            }
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-mycelial-acquisition-domain-manifest-v1");
        hasher.update(&self.schema_version.to_le_bytes());
        hasher.update(&[rights_status_tag(self.source_rights_status)]);
        for value in [
            self.source_artifact_id.as_str(),
            self.source_artifact_digest.as_str(),
            self.rights_evidence_digest.as_str(),
            self.source_format.as_str(),
            self.acquisition_session_id.as_str(),
            self.biological_replicate_id.as_str(),
            self.coordinate_frame_id.as_str(),
            self.declared_scope_digest.as_str(),
            self.instrument_metadata_digest.as_str(),
        ] {
            hash_string(&mut hasher, value);
        }
        for value in [
            self.domain.width_pixels,
            self.domain.height_pixels,
            self.domain.z_planes,
            self.domain.channels,
            self.domain.timepoints,
            self.domain.tile_width_pixels,
            self.domain.tile_height_pixels,
        ] {
            hasher.update(&value.to_le_bytes());
        }
        hasher.update(&(self.axes.len() as u64).to_le_bytes());
        for axis in &self.axes {
            hasher.update(&[axis_tag(axis.name)]);
            hasher.update(&axis.length.to_le_bytes());
            hash_string(&mut hasher, &axis.unit);
            hasher.update(&axis.origin.to_bits().to_le_bytes());
            match axis.physical_scale {
                Some(scale) => {
                    hasher.update(&[1]);
                    hasher.update(&scale.to_bits().to_le_bytes());
                }
                None => {
                    hasher.update(&[0]);
                }
            }
        }
        for value in [
            self.spatial_transform_digest.as_str(),
            self.calibration_digest.as_str(),
            self.manifest_producer_id.as_str(),
            self.manifest_producer_version.as_str(),
            self.manifest_producer_digest.as_str(),
        ] {
            hash_string(&mut hasher, value);
        }
        hasher.update(&(self.scope_exclusions.len() as u64).to_le_bytes());
        for exclusion in &self.scope_exclusions {
            hash_unit_key(&mut hasher, exclusion.unit);
            hash_string(&mut hasher, &exclusion.reason);
            hash_string(&mut hasher, &exclusion.scope_decision_digest);
        }
        Ok(hasher.finalize().to_hex().to_string())
    }
}

impl MycelialExtractionCoverageReceipt {
    fn validate_unit_outcome(
        outcome: &MycelialExtractionUnitOutcome,
    ) -> Result<(), MycelialAcquisitionDomainError> {
        validate_digest(&outcome.source_unit_digest, "source_unit_digest")?;
        match outcome.state {
            MycelialExtractionUnitState::Processed => {
                let Some(digest) = outcome.output_observation_inventory_digest.as_deref() else {
                    return Err(MycelialAcquisitionDomainError::InvalidUnitDisposition {
                        reason: "processed unit requires an output observation inventory digest",
                    });
                };
                validate_digest(digest, "output_observation_inventory_digest")?;
                if outcome.output_observation_count.is_none()
                    || outcome.disposition_reason.is_some()
                {
                    return Err(MycelialAcquisitionDomainError::InvalidUnitDisposition {
                        reason: "processed unit requires a count and cannot carry a failure reason",
                    });
                }
            }
            MycelialExtractionUnitState::UnavailableOrCorrupted
            | MycelialExtractionUnitState::Unmeasurable
            | MycelialExtractionUnitState::ProcessingFailed
            | MycelialExtractionUnitState::Pending => {
                if outcome.output_observation_inventory_digest.is_some()
                    || outcome.output_observation_count.is_some()
                {
                    return Err(MycelialAcquisitionDomainError::InvalidUnitDisposition {
                        reason: "non-processed unit cannot claim an output inventory",
                    });
                }
                let Some(reason) = outcome.disposition_reason.as_deref() else {
                    return Err(MycelialAcquisitionDomainError::InvalidUnitDisposition {
                        reason: "non-processed unit requires an explicit disposition reason",
                    });
                };
                if !reason_is_valid(reason) {
                    return Err(MycelialAcquisitionDomainError::InvalidDispositionReason);
                }
            }
        }
        Ok(())
    }

    /// Validate exact coverage of every in-scope acquisition unit.
    ///
    /// A processed unit only attests that the named extractor produced a bound output
    /// inventory for that source region. It does not establish biological recall or
    /// ground-truth correctness; those require independent annotations/assays.
    pub fn validate_against(
        &self,
        manifest: &MycelialAcquisitionDomainManifest,
    ) -> Result<(), MycelialAcquisitionDomainError> {
        manifest.validate()?;
        if self.schema_version != MYCELIAL_EXTRACTION_RECEIPT_SCHEMA_VERSION {
            return Err(MycelialAcquisitionDomainError::InvalidSchemaVersion(
                self.schema_version,
            ));
        }
        if self.acquisition_domain_manifest_digest != manifest.canonical_digest_hex()? {
            return Err(MycelialAcquisitionDomainError::ManifestDigestMismatch);
        }
        validate_identifier(
            &self.extractor_id,
            "extractor_id",
            MAX_DOMAIN_IDENTIFIER_BYTES,
        )?;
        validate_identifier(
            &self.extractor_version,
            "extractor_version",
            MAX_DOMAIN_VERSION_BYTES,
        )?;
        validate_digest(&self.extractor_config_digest, "extractor_config_digest")?;

        let expected = manifest.in_scope_unit_keys()?;
        if self.units.len() != expected.len()
            || self.units.len() > MAX_MYCELIAL_ACQUISITION_DOMAIN_UNITS
        {
            return Err(MycelialAcquisitionDomainError::ReceiptUnitInventoryMismatch);
        }
        for (outcome, expected_unit) in self.units.iter().zip(&expected) {
            if outcome.unit != *expected_unit {
                return Err(MycelialAcquisitionDomainError::ReceiptUnitInventoryMismatch);
            }
            Self::validate_unit_outcome(outcome)?;
        }
        Ok(())
    }

    /// Reconcile the extraction receipt against the complete observation ledger.
    ///
    /// This checks that every observation is assigned to an in-scope acquisition unit,
    /// that source/replicate/coordinate bindings match, and that the canonical digest
    /// and count of IDs for each unit exactly match that unit's extraction output.
    /// It requires a complete extraction; partial receipts cannot certify a complete
    /// observation inventory.
    pub fn validate_observation_ledger(
        &self,
        manifest: &MycelialAcquisitionDomainManifest,
        ledger: &MycelialObservationCoverageLedger,
    ) -> Result<(), MycelialAcquisitionDomainError> {
        let summary = self.summarize(manifest)?;
        if summary.status == MycelialExtractionCoverageStatus::Partial {
            return Err(
                MycelialAcquisitionDomainError::IncompleteExtractionBlocksObservationInventory,
            );
        }
        ledger
            .validate()
            .map_err(|_| MycelialAcquisitionDomainError::ObservationLedgerInvalid)?;
        if ledger.source_dataset_id != manifest.source_artifact_id
            || ledger.source_dataset_digest != manifest.source_artifact_digest
            || ledger.coordinate_frame_id != manifest.coordinate_frame_id
            || ledger.spatial_transform_digest != manifest.spatial_transform_digest
            || ledger.calibration_digest != manifest.calibration_digest
            || ledger
                .records
                .iter()
                .any(|record| record.biological_replicate_id != manifest.biological_replicate_id)
        {
            return Err(MycelialAcquisitionDomainError::ObservationSourceBindingMismatch);
        }

        let in_scope = manifest.in_scope_unit_keys()?;
        let mut observations_by_unit: BTreeMap<MycelialAcquisitionUnitKey, Vec<String>> =
            BTreeMap::new();
        for record in &ledger.records {
            let Some(unit) = record.source_unit_key else {
                return Err(
                    MycelialAcquisitionDomainError::ObservationSourceUnitMissing(
                        record.observation_id.clone(),
                    ),
                );
            };
            if in_scope.binary_search(&unit).is_err() {
                return Err(
                    MycelialAcquisitionDomainError::ObservationSourceUnitOutsideScope(
                        record.observation_id.clone(),
                    ),
                );
            }
            observations_by_unit
                .entry(unit)
                .or_default()
                .push(record.observation_id.clone());
        }

        for outcome in &self.units {
            let ids = observations_by_unit
                .remove(&outcome.unit)
                .unwrap_or_default();
            let count = u64::try_from(ids.len()).map_err(|_| {
                MycelialAcquisitionDomainError::ObservationUnitInventoryCountMismatch(outcome.unit)
            })?;
            if outcome.output_observation_count != Some(count) {
                return Err(
                    MycelialAcquisitionDomainError::ObservationUnitInventoryCountMismatch(
                        outcome.unit,
                    ),
                );
            }
            let digest = mycelial_observation_id_inventory_digest(&ids);
            if outcome.output_observation_inventory_digest.as_deref() != Some(digest.as_str()) {
                return Err(
                    MycelialAcquisitionDomainError::ObservationUnitInventoryDigestMismatch(
                        outcome.unit,
                    ),
                );
            }
        }
        if !observations_by_unit.is_empty() {
            return Err(MycelialAcquisitionDomainError::ReceiptUnitInventoryMismatch);
        }
        Ok(())
    }

    /// Summarize only after validating the manifest binding and exact unit inventory.
    pub fn summarize(
        &self,
        manifest: &MycelialAcquisitionDomainManifest,
    ) -> Result<MycelialExtractionCoverageSummary, MycelialAcquisitionDomainError> {
        self.validate_against(manifest)?;
        let mut summary = MycelialExtractionCoverageSummary {
            status: MycelialExtractionCoverageStatus::Partial,
            total_domain_units: manifest.domain.expected_unit_count()?,
            in_scope_units: self.units.len(),
            excluded_units: manifest.scope_exclusions.len(),
            processed_units: 0,
            unavailable_or_corrupted_units: 0,
            unmeasurable_units: 0,
            processing_failed_units: 0,
            pending_units: 0,
        };
        for outcome in &self.units {
            match outcome.state {
                MycelialExtractionUnitState::Processed => summary.processed_units += 1,
                MycelialExtractionUnitState::UnavailableOrCorrupted => {
                    summary.unavailable_or_corrupted_units += 1;
                }
                MycelialExtractionUnitState::Unmeasurable => summary.unmeasurable_units += 1,
                MycelialExtractionUnitState::ProcessingFailed => {
                    summary.processing_failed_units += 1;
                }
                MycelialExtractionUnitState::Pending => summary.pending_units += 1,
            }
        }

        if summary.processed_units == summary.in_scope_units {
            summary.status = if summary.excluded_units == 0 {
                MycelialExtractionCoverageStatus::CompleteFullAcquisitionDomain
            } else {
                MycelialExtractionCoverageStatus::CompleteDeclaredScopeWithExclusions
            };
        }
        Ok(summary)
    }

    /// Hash exact receipt content independently from semantic validation.
    pub fn canonical_digest_hex(&self) -> Result<String, MycelialAcquisitionDomainError> {
        if self.units.len() > MAX_MYCELIAL_ACQUISITION_DOMAIN_UNITS {
            return Err(MycelialAcquisitionDomainError::DigestResourceLimitExceeded);
        }
        let mut string_bytes = self
            .acquisition_domain_manifest_digest
            .len()
            .saturating_add(self.extractor_id.len())
            .saturating_add(self.extractor_version.len())
            .saturating_add(self.extractor_config_digest.len());
        for outcome in &self.units {
            string_bytes = string_bytes.saturating_add(outcome.source_unit_digest.len());
            if let Some(value) = &outcome.output_observation_inventory_digest {
                string_bytes = string_bytes.saturating_add(value.len());
            }
            if let Some(value) = &outcome.disposition_reason {
                string_bytes = string_bytes.saturating_add(value.len());
            }
        }
        if string_bytes > MAX_CANONICAL_DOMAIN_STRING_BYTES {
            return Err(MycelialAcquisitionDomainError::DigestResourceLimitExceeded);
        }

        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-mycelial-extraction-coverage-receipt-v1");
        hasher.update(&self.schema_version.to_le_bytes());
        hash_string(&mut hasher, &self.acquisition_domain_manifest_digest);
        hash_string(&mut hasher, &self.extractor_id);
        hash_string(&mut hasher, &self.extractor_version);
        hash_string(&mut hasher, &self.extractor_config_digest);
        hasher.update(&(self.units.len() as u64).to_le_bytes());
        for outcome in &self.units {
            hash_unit_key(&mut hasher, outcome.unit);
            hash_string(&mut hasher, &outcome.source_unit_digest);
            hasher.update(&[extraction_state_tag(outcome.state)]);
            match outcome.output_observation_inventory_digest.as_deref() {
                Some(digest) => {
                    hasher.update(&[1]);
                    hash_string(&mut hasher, digest);
                }
                None => {
                    hasher.update(&[0]);
                }
            }
            match outcome.output_observation_count {
                Some(count) => {
                    hasher.update(&[1]);
                    hasher.update(&count.to_le_bytes());
                }
                None => {
                    hasher.update(&[0]);
                }
            }
            match outcome.disposition_reason.as_deref() {
                Some(reason) => {
                    hasher.update(&[1]);
                    hash_string(&mut hasher, reason);
                }
                None => {
                    hasher.update(&[0]);
                }
            }
        }
        Ok(hasher.finalize().to_hex().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(character: char) -> String {
        std::iter::repeat_n(character, DIGEST_HEX_BYTES).collect()
    }

    fn manifest() -> MycelialAcquisitionDomainManifest {
        MycelialAcquisitionDomainManifest {
            schema_version: MYCELIAL_ACQUISITION_DOMAIN_SCHEMA_VERSION,
            source_artifact_id: "source-001".to_string(),
            source_artifact_digest: digest('a'),
            source_rights_status: MycelialSourceRightsStatus::VerifiedForProcessing,
            rights_evidence_digest: digest('b'),
            source_format: "ome_tiff".to_string(),
            acquisition_session_id: "session-001".to_string(),
            biological_replicate_id: "replicate-001".to_string(),
            coordinate_frame_id: "specimen-frame-v1".to_string(),
            declared_scope_digest: digest('6'),
            instrument_metadata_digest: digest('7'),
            axes: vec![
                MycelialAcquisitionAxis {
                    name: MycelialAcquisitionAxisName::X,
                    length: 8,
                    unit: "pixel".to_string(),
                    origin: 0.0,
                    physical_scale: Some(0.25),
                },
                MycelialAcquisitionAxis {
                    name: MycelialAcquisitionAxisName::Y,
                    length: 6,
                    unit: "pixel".to_string(),
                    origin: 0.0,
                    physical_scale: Some(0.25),
                },
                MycelialAcquisitionAxis {
                    name: MycelialAcquisitionAxisName::Z,
                    length: 2,
                    unit: "micrometer".to_string(),
                    origin: 0.0,
                    physical_scale: Some(0.8),
                },
                MycelialAcquisitionAxis {
                    name: MycelialAcquisitionAxisName::Channel,
                    length: 2,
                    unit: "channel_index".to_string(),
                    origin: 0.0,
                    physical_scale: None,
                },
                MycelialAcquisitionAxis {
                    name: MycelialAcquisitionAxisName::Time,
                    length: 2,
                    unit: "frame".to_string(),
                    origin: 0.0,
                    physical_scale: None,
                },
            ],
            domain: MycelialAcquisitionDomainSpec {
                width_pixels: 8,
                height_pixels: 6,
                z_planes: 2,
                channels: 2,
                timepoints: 2,
                tile_width_pixels: 4,
                tile_height_pixels: 3,
            },
            spatial_transform_digest: digest('b'),
            calibration_digest: digest('c'),
            manifest_producer_id: "ome_metadata_reader".to_string(),
            manifest_producer_version: "1.0.0".to_string(),
            manifest_producer_digest: digest('d'),
            scope_exclusions: Vec::new(),
        }
    }

    fn receipt(manifest: &MycelialAcquisitionDomainManifest) -> MycelialExtractionCoverageReceipt {
        let manifest_digest = manifest.canonical_digest_hex().unwrap();
        let units = manifest
            .in_scope_unit_keys()
            .unwrap()
            .into_iter()
            .enumerate()
            .map(|(index, unit)| MycelialExtractionUnitOutcome {
                unit,
                source_unit_digest: digest('8'),
                state: MycelialExtractionUnitState::Processed,
                output_observation_inventory_digest: Some(digest(
                    char::from_digit((index % 16) as u32, 16).unwrap(),
                )),
                output_observation_count: Some(index as u64),
                disposition_reason: None,
            })
            .collect();
        MycelialExtractionCoverageReceipt {
            schema_version: MYCELIAL_EXTRACTION_RECEIPT_SCHEMA_VERSION,
            acquisition_domain_manifest_digest: manifest_digest,
            extractor_id: "mycelial_observation_extractor".to_string(),
            extractor_version: "0.1.0".to_string(),
            extractor_config_digest: digest('e'),
            units,
        }
    }

    fn observation_ledger(
        manifest: &MycelialAcquisitionDomainManifest,
        first_unit: MycelialAcquisitionUnitKey,
        second_unit: MycelialAcquisitionUnitKey,
    ) -> MycelialObservationCoverageLedger {
        let ids = vec!["obs-001".to_string(), "obs-002".to_string()];
        MycelialObservationCoverageLedger {
            schema_version:
                crate::mycelial_observation_coverage::MYCELIAL_OBSERVATION_COVERAGE_SCHEMA_VERSION,
            source_dataset_id: manifest.source_artifact_id.clone(),
            source_dataset_digest: manifest.source_artifact_digest.clone(),
            observation_inventory_digest: mycelial_observation_id_inventory_digest(&ids),
            declared_observation_ids: ids,
            coordinate_frame_id: manifest.coordinate_frame_id.clone(),
            spatial_transform_digest: manifest.spatial_transform_digest.clone(),
            calibration_digest: manifest.calibration_digest.clone(),
            candidate_generator_id: "mycelial-candidate-generator".to_string(),
            candidate_generator_version: "0.1.0".to_string(),
            candidate_generator_digest: digest('e'),
            records: vec![
                MycelialObservationCoverageRecord {
                    observation_id: "obs-001".to_string(),
                    observation_digest: digest('1'),
                    source_unit_key: Some(first_unit),
                    acquisition_group_id: manifest.acquisition_session_id.clone(),
                    biological_replicate_id: manifest.biological_replicate_id.clone(),
                    time_window_id: "time-001".to_string(),
                    state: MycelialObservationCoverageState::CandidateEmitted,
                    candidate_ids: vec!["candidate-001".to_string()],
                    residual_kind: None,
                },
                MycelialObservationCoverageRecord {
                    observation_id: "obs-002".to_string(),
                    observation_digest: digest('2'),
                    source_unit_key: Some(second_unit),
                    acquisition_group_id: manifest.acquisition_session_id.clone(),
                    biological_replicate_id: manifest.biological_replicate_id.clone(),
                    time_window_id: "time-001".to_string(),
                    state: MycelialObservationCoverageState::NoCandidateGenerated,
                    candidate_ids: Vec::new(),
                    residual_kind: Some(MycelialObservationResidualKind::PossibleContactMissing),
                },
            ],
        }
    }

    fn bind_receipt_to_ledger_outputs(
        receipt: &mut MycelialExtractionCoverageReceipt,
        ledger: &MycelialObservationCoverageLedger,
    ) {
        for outcome in &mut receipt.units {
            let ids: Vec<String> = ledger
                .records
                .iter()
                .filter(|record| record.source_unit_key == Some(outcome.unit))
                .map(|record| record.observation_id.clone())
                .collect();
            outcome.output_observation_count = Some(ids.len() as u64);
            outcome.output_observation_inventory_digest =
                Some(mycelial_observation_id_inventory_digest(&ids));
        }
    }

    #[test]
    fn expected_unit_enumerator_matches_cartesian_domain() {
        let domain = manifest().domain;
        assert_eq!(domain.expected_unit_count().unwrap(), 32);
        assert_eq!(domain.expected_unit_keys().unwrap().len(), 32);
    }

    #[test]
    fn complete_receipt_validates_and_reports_full_domain() {
        let manifest = manifest();
        let receipt = receipt(&manifest);
        receipt.validate_against(&manifest).unwrap();
        let summary = receipt.summarize(&manifest).unwrap();
        assert_eq!(
            summary.status,
            MycelialExtractionCoverageStatus::CompleteFullAcquisitionDomain
        );
        assert_eq!(summary.total_domain_units, 32);
        assert_eq!(summary.processed_units, 32);
        assert_eq!(summary.pending_units, 0);
    }

    #[test]
    fn extraction_receipt_reconciles_per_unit_observation_inventories() {
        let manifest = manifest();
        let units = manifest.domain.expected_unit_keys().unwrap();
        let ledger = observation_ledger(&manifest, units[0], units[1]);
        let mut receipt = receipt(&manifest);
        bind_receipt_to_ledger_outputs(&mut receipt, &ledger);
        receipt
            .validate_observation_ledger(&manifest, &ledger)
            .unwrap();
    }

    #[test]
    fn extraction_reconciliation_rejects_per_unit_digest_tampering() {
        let manifest = manifest();
        let units = manifest.domain.expected_unit_keys().unwrap();
        let ledger = observation_ledger(&manifest, units[0], units[1]);
        let mut receipt = receipt(&manifest);
        bind_receipt_to_ledger_outputs(&mut receipt, &ledger);
        receipt.units[0].output_observation_inventory_digest = Some(digest('f'));
        assert_eq!(
            receipt.validate_observation_ledger(&manifest, &ledger),
            Err(MycelialAcquisitionDomainError::ObservationUnitInventoryDigestMismatch(units[0]))
        );
    }

    #[test]
    fn extraction_reconciliation_requires_source_unit_per_observation() {
        let manifest = manifest();
        let units = manifest.domain.expected_unit_keys().unwrap();
        let mut ledger = observation_ledger(&manifest, units[0], units[1]);
        ledger.records[0].source_unit_key = None;
        let mut receipt = receipt(&manifest);
        bind_receipt_to_ledger_outputs(&mut receipt, &ledger);
        assert_eq!(
            receipt.validate_observation_ledger(&manifest, &ledger),
            Err(
                MycelialAcquisitionDomainError::ObservationSourceUnitMissing("obs-001".to_string())
            )
        );
    }

    #[test]
    fn partial_extraction_cannot_certify_complete_observation_inventory() {
        let manifest = manifest();
        let units = manifest.domain.expected_unit_keys().unwrap();
        let ledger = observation_ledger(&manifest, units[0], units[1]);
        let mut receipt = receipt(&manifest);
        bind_receipt_to_ledger_outputs(&mut receipt, &ledger);
        receipt.units[0].state = MycelialExtractionUnitState::Pending;
        receipt.units[0].output_observation_inventory_digest = None;
        receipt.units[0].output_observation_count = None;
        receipt.units[0].disposition_reason = Some("pending source review".to_string());
        assert_eq!(
            receipt.validate_observation_ledger(&manifest, &ledger),
            Err(MycelialAcquisitionDomainError::IncompleteExtractionBlocksObservationInventory)
        );
    }

    #[test]
    fn omission_of_a_z_plane_is_detected() {
        let manifest = manifest();
        let mut receipt = receipt(&manifest);
        receipt.units.retain(|outcome| outcome.unit.z_plane == 0);
        assert_eq!(
            receipt.validate_against(&manifest),
            Err(MycelialAcquisitionDomainError::ReceiptUnitInventoryMismatch)
        );
    }

    #[test]
    fn omission_of_a_channel_is_detected() {
        let manifest = manifest();
        let mut receipt = receipt(&manifest);
        receipt.units.retain(|outcome| outcome.unit.channel == 0);
        assert_eq!(
            receipt.validate_against(&manifest),
            Err(MycelialAcquisitionDomainError::ReceiptUnitInventoryMismatch)
        );
    }

    #[test]
    fn omission_of_a_timepoint_is_detected() {
        let manifest = manifest();
        let mut receipt = receipt(&manifest);
        receipt.units.retain(|outcome| outcome.unit.timepoint == 0);
        assert_eq!(
            receipt.validate_against(&manifest),
            Err(MycelialAcquisitionDomainError::ReceiptUnitInventoryMismatch)
        );
    }

    #[test]
    fn omission_of_an_interior_tile_is_detected() {
        let manifest = manifest();
        let mut receipt = receipt(&manifest);
        receipt
            .units
            .retain(|outcome| outcome.unit.tile_x != 1 || outcome.unit.tile_y != 1);
        assert_eq!(
            receipt.validate_against(&manifest),
            Err(MycelialAcquisitionDomainError::ReceiptUnitInventoryMismatch)
        );
    }

    #[test]
    fn pending_unit_blocks_complete_status() {
        let manifest = manifest();
        let mut receipt = receipt(&manifest);
        receipt.units[0].state = MycelialExtractionUnitState::Pending;
        receipt.units[0].output_observation_inventory_digest = None;
        receipt.units[0].output_observation_count = None;
        receipt.units[0].disposition_reason = Some("awaiting extraction".to_string());
        let summary = receipt.summarize(&manifest).unwrap();
        assert_eq!(summary.status, MycelialExtractionCoverageStatus::Partial);
        assert_eq!(summary.pending_units, 1);
    }

    #[test]
    fn explicit_scope_exclusion_is_visible_and_not_called_full_domain() {
        let mut manifest = manifest();
        let excluded_unit = manifest.domain.expected_unit_keys().unwrap()[0];
        manifest
            .scope_exclusions
            .push(MycelialAcquisitionScopeExclusion {
                unit: excluded_unit,
                reason: "outside preregistered field of view".to_string(),
                scope_decision_digest: digest('f'),
            });
        let receipt = receipt(&manifest);
        let summary = receipt.summarize(&manifest).unwrap();
        assert_eq!(
            summary.status,
            MycelialExtractionCoverageStatus::CompleteDeclaredScopeWithExclusions
        );
        assert_eq!(summary.excluded_units, 1);
        assert_eq!(summary.in_scope_units, 31);
    }

    #[test]
    fn exclusion_outside_source_domain_is_rejected() {
        let mut manifest = manifest();
        manifest
            .scope_exclusions
            .push(MycelialAcquisitionScopeExclusion {
                unit: MycelialAcquisitionUnitKey {
                    tile_x: 99,
                    tile_y: 99,
                    z_plane: 0,
                    channel: 0,
                    timepoint: 0,
                },
                reason: "not acquired".to_string(),
                scope_decision_digest: digest('f'),
            });
        assert_eq!(
            manifest.validate(),
            Err(MycelialAcquisitionDomainError::ExcludedUnitOutsideDomain)
        );
    }

    #[test]
    fn all_units_cannot_be_excluded() {
        let mut manifest = manifest();
        manifest.scope_exclusions = manifest
            .domain
            .expected_unit_keys()
            .unwrap()
            .into_iter()
            .map(|unit| MycelialAcquisitionScopeExclusion {
                unit,
                reason: "excluded".to_string(),
                scope_decision_digest: digest('f'),
            })
            .collect();
        assert_eq!(
            manifest.validate(),
            Err(MycelialAcquisitionDomainError::EmptyDeclaredScope)
        );
    }

    #[test]
    fn source_rights_must_be_cleared_before_extraction() {
        let mut manifest = manifest();
        manifest.source_rights_status = MycelialSourceRightsStatus::AuthorPermissionRequired;
        assert_eq!(
            manifest.validate(),
            Err(MycelialAcquisitionDomainError::SourceRightsNotCleared(
                MycelialSourceRightsStatus::AuthorPermissionRequired
            ))
        );
    }

    #[test]
    fn rights_evidence_digest_is_required_and_committed() {
        let original = manifest();
        let original_digest = original.canonical_digest_hex().unwrap();
        let mut changed = original;
        changed.rights_evidence_digest = digest('9');
        assert_ne!(changed.canonical_digest_hex().unwrap(), original_digest);
        changed.rights_evidence_digest = "not-a-digest".to_string();
        assert!(matches!(
            changed.validate(),
            Err(MycelialAcquisitionDomainError::InvalidDigest(
                "rights_evidence_digest"
            ))
        ));
    }

    #[test]
    fn declared_scope_and_instrument_metadata_are_committed() {
        let original = manifest();
        let original_digest = original.canonical_digest_hex().unwrap();
        let mut changed = original.clone();
        changed.declared_scope_digest = digest('8');
        assert_ne!(changed.canonical_digest_hex().unwrap(), original_digest);
        let mut changed = original.clone();
        changed.instrument_metadata_digest = digest('9');
        assert_ne!(changed.canonical_digest_hex().unwrap(), original_digest);
    }

    #[test]
    fn source_rights_status_is_committed_by_manifest_digest() {
        let original = manifest();
        let original_digest = original.canonical_digest_hex().unwrap();
        let mut changed = original;
        changed.source_rights_status = MycelialSourceRightsStatus::Unknown;
        assert_ne!(changed.canonical_digest_hex().unwrap(), original_digest);
    }

    #[test]
    fn axis_dimension_mismatch_is_rejected() {
        let mut manifest = manifest();
        manifest.axes[2].length = 3;
        assert_eq!(
            manifest.validate(),
            Err(MycelialAcquisitionDomainError::AxisDimensionMismatch(
                MycelialAcquisitionAxisName::Z
            ))
        );
    }

    #[test]
    fn duplicate_axis_is_rejected() {
        let mut manifest = manifest();
        manifest.axes[4].name = MycelialAcquisitionAxisName::Channel;
        assert!(matches!(
            manifest.validate(),
            Err(MycelialAcquisitionDomainError::DuplicateAxis(
                MycelialAcquisitionAxisName::Channel
            ))
        ));
    }

    #[test]
    fn resource_limit_blocks_unbounded_domain_enumeration() {
        let domain = MycelialAcquisitionDomainSpec {
            width_pixels: u32::MAX,
            height_pixels: u32::MAX,
            z_planes: u32::MAX,
            channels: u32::MAX,
            timepoints: u32::MAX,
            tile_width_pixels: 1,
            tile_height_pixels: 1,
        };
        assert!(matches!(
            domain.expected_unit_count(),
            Err(MycelialAcquisitionDomainError::DomainResourceLimitExceeded(
                _
            ))
        ));
    }

    #[test]
    fn processed_unit_requires_output_digest_and_count() {
        let manifest = manifest();
        let mut receipt = receipt(&manifest);
        receipt.units[0].output_observation_inventory_digest = None;
        assert!(matches!(
            receipt.validate_against(&manifest),
            Err(MycelialAcquisitionDomainError::InvalidUnitDisposition { .. })
        ));
    }

    #[test]
    fn unavailable_unit_cannot_claim_zero_output_observations() {
        let manifest = manifest();
        let mut receipt = receipt(&manifest);
        receipt.units[0].state = MycelialExtractionUnitState::UnavailableOrCorrupted;
        receipt.units[0].output_observation_inventory_digest = None;
        receipt.units[0].output_observation_count = Some(0);
        receipt.units[0].disposition_reason = Some("corrupt tile".to_string());
        assert!(matches!(
            receipt.validate_against(&manifest),
            Err(MycelialAcquisitionDomainError::InvalidUnitDisposition { .. })
        ));
    }

    #[test]
    fn receipt_must_bind_exact_manifest_digest() {
        let manifest = manifest();
        let mut receipt = receipt(&manifest);
        receipt.acquisition_domain_manifest_digest = digest('0');
        assert_eq!(
            receipt.validate_against(&manifest),
            Err(MycelialAcquisitionDomainError::ManifestDigestMismatch)
        );
    }

    #[test]
    fn manifest_digest_changes_when_axis_origin_changes() {
        let original = manifest();
        let original_digest = original.canonical_digest_hex().unwrap();
        let mut changed = original;
        changed.axes[0].origin = 12.5;
        assert_ne!(changed.canonical_digest_hex().unwrap(), original_digest);
        changed.axes[0].origin = f64::NAN;
        assert!(matches!(
            changed.validate(),
            Err(MycelialAcquisitionDomainError::InvalidAxisMetadata(_))
        ));
    }

    #[test]
    fn source_unit_digest_is_committed_and_validated() {
        let manifest = manifest();
        let mut receipt = receipt(&manifest);
        let original_digest = receipt.canonical_digest_hex().unwrap();
        receipt.units[0].source_unit_digest = digest('7');
        assert_ne!(receipt.canonical_digest_hex().unwrap(), original_digest);
        receipt.units[0].source_unit_digest = "not-a-digest".to_string();
        assert!(matches!(
            receipt.validate_against(&manifest),
            Err(MycelialAcquisitionDomainError::InvalidDigest(
                "source_unit_digest"
            ))
        ));
    }

    #[test]
    fn manifest_digest_changes_when_calibration_changes() {
        let original = manifest();
        let original_digest = original.canonical_digest_hex().unwrap();
        let mut changed = original;
        changed.calibration_digest = digest('9');
        assert_ne!(changed.canonical_digest_hex().unwrap(), original_digest);
    }

    #[test]
    fn receipt_digest_changes_when_a_unit_disposition_changes() {
        let manifest = manifest();
        let original = receipt(&manifest);
        let original_digest = original.canonical_digest_hex().unwrap();
        let mut changed = original;
        changed.units[0].state = MycelialExtractionUnitState::Pending;
        assert_ne!(changed.canonical_digest_hex().unwrap(), original_digest);
    }

    #[test]
    fn deserialization_rejects_unknown_manifest_fields() {
        let mut value = serde_json::to_value(manifest()).unwrap();
        value["unbound_scan_complete"] = serde_json::json!(true);
        assert!(serde_json::from_value::<MycelialAcquisitionDomainManifest>(value).is_err());
    }

    #[test]
    fn deserialization_rejects_unknown_receipt_unit_fields() {
        let manifest = manifest();
        let mut value = serde_json::to_value(receipt(&manifest)).unwrap();
        value["units"][0]["unbound_geometry"] = serde_json::json!({"z": 1.0});
        assert!(serde_json::from_value::<MycelialExtractionCoverageReceipt>(value).is_err());
    }
}
