// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Fail-closed public-data manifest and as-of-decision validation for clean-airline
//! benchmarks. This module records source metadata and validates chronology/identity;
//! it does not download data, interpret publisher licenses, authenticate the publisher,
//! or prove that a dataset's semantic claims are correct.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use sha2::{Digest, Sha256};

const MAX_SOURCES: usize = 128;
const MAX_PARTITIONS: usize = 20_000;
const MAX_SPLIT_ASSIGNMENTS: usize = 10_000_000;

/// Whether the declared data files were actually obtained. Cataloguing a source is not
/// equivalent to having retrieved its files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AcquisitionStatus {
    Catalogued,
    Retrieved,
    AccessUnavailable,
    Excluded,
}

/// Review state for use rights. This state is metadata, not legal advice or a license.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LicenseStatus {
    VerifiedOpen,
    TermsReviewed,
    Restricted,
    Unknown,
}

/// One declared public or licensed dataset/table release.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetSource {
    /// Stable ID includes publisher, dataset/table, and release/schema generation.
    pub source_id: String,
    pub publisher: String,
    pub dataset_name: String,
    pub release_id: String,
    pub schema_or_methodology_version: String,
    pub canonical_url: String,
    /// Coverage is a half-open interval [start, end) in Unix seconds and UTC.
    pub coverage_start_epoch: i64,
    pub coverage_end_epoch: i64,
    pub catalogued_at_epoch: i64,
    pub dataset_downloaded_at_epoch: Option<i64>,
    pub acquisition_status: AcquisitionStatus,
    pub license_status: LicenseStatus,
}

/// One immutable file or partition in a retrieved source, with exact digest and byte
/// size. Start/end represent the temporal coverage claimed for this partition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetPartition {
    pub source_id: String,
    pub partition_id: String,
    pub start_epoch: i64,
    pub end_epoch: i64,
    pub downloaded_at_epoch: i64,
    pub byte_len: u64,
    pub content_digest: String,
}

/// The manifest captures source metadata and exact file partitions. It does not contain
/// downloaded file bytes; every partition digest must be calculated from those bytes
/// by the ingestion pipeline before the partition is marked retrieved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicDataManifest {
    pub manifest_id: String,
    pub manifest_version: String,
    pub created_at_epoch: i64,
    pub sources: Vec<DatasetSource>,
    pub partitions: Vec<DatasetPartition>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataContractError {
    MissingIdentifier(&'static str),
    DuplicateIdentifier(&'static str),
    InvalidUrl,
    InvalidCoverage(&'static str),
    InvalidTimestamp(&'static str),
    TooManySources,
    TooManyPartitions,
    UnknownPartitionSource { source_id: String },
    PartitionSourceNotRetrieved { source_id: String },
    MissingRetrievedPartition { source_id: String },
    UnexpectedPartitionForUnretrievedSource { source_id: String },
    PartitionOutOfSourceCoverage { partition_id: String },
    OverlappingSourcePartitions { source_id: String },
    InvalidDigest { field: &'static str },
    InvalidByteLength { partition_id: String },
    InvalidSplitWindow(&'static str),
    OverlappingSplitWindows,
    EventClusterLeakage {
        event_cluster_id: String,
        first_split: SplitLabel,
        conflicting_split: SplitLabel,
    },
    FutureInformationLeakage {
        feature_id: String,
        decision_at_epoch: i64,
        available_at_epoch: i64,
    },
    FutureObservedValue {
        feature_id: String,
        decision_at_epoch: i64,
        value_time_epoch: i64,
    },
    InvalidFeatureChronology { feature_id: String },
    TooManyAssignments,
    ArithmeticOverflow,
}

impl fmt::Display for DataContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingIdentifier(field) => write!(f, "missing identifier: {field}"),
            Self::DuplicateIdentifier(field) => write!(f, "duplicate identifier: {field}"),
            Self::InvalidUrl => write!(f, "canonical source URL must be a valid-looking HTTPS URL"),
            Self::InvalidCoverage(field) => write!(f, "invalid half-open coverage interval: {field}"),
            Self::InvalidTimestamp(field) => write!(f, "invalid or inconsistent timestamp: {field}"),
            Self::TooManySources => write!(f, "source manifest exceeds the source-count limit"),
            Self::TooManyPartitions => write!(f, "source manifest exceeds the partition-count limit"),
            Self::UnknownPartitionSource { source_id } => {
                write!(f, "partition references unknown source {source_id}")
            }
            Self::PartitionSourceNotRetrieved { source_id } => {
                write!(f, "partition source {source_id} is not marked retrieved")
            }
            Self::MissingRetrievedPartition { source_id } => {
                write!(f, "retrieved source {source_id} has no partition")
            }
            Self::UnexpectedPartitionForUnretrievedSource { source_id } => {
                write!(f, "unretrieved source {source_id} has file partitions")
            }
            Self::PartitionOutOfSourceCoverage { partition_id } => {
                write!(f, "partition {partition_id} is outside its source coverage")
            }
            Self::OverlappingSourcePartitions { source_id } => {
                write!(f, "temporal partitions overlap for source {source_id}")
            }
            Self::InvalidDigest { field } => write!(f, "invalid SHA-256 digest: {field}"),
            Self::InvalidByteLength { partition_id } => {
                write!(f, "partition {partition_id} has zero byte length")
            }
            Self::InvalidSplitWindow(field) => write!(f, "invalid temporal split window: {field}"),
            Self::OverlappingSplitWindows => write!(f, "temporal split windows overlap or are out of order"),
            Self::EventClusterLeakage {
                event_cluster_id,
                first_split,
                conflicting_split,
            } => write!(
                f,
                "event cluster {event_cluster_id} is split between {first_split:?} and {conflicting_split:?}"
            ),
            Self::FutureInformationLeakage {
                feature_id,
                decision_at_epoch,
                available_at_epoch,
            } => write!(
                f,
                "feature {feature_id} was available at {available_at_epoch}, after decision time {decision_at_epoch}"
            ),
            Self::FutureObservedValue {
                feature_id,
                decision_at_epoch,
                value_time_epoch,
            } => write!(
                f,
                "observed feature {feature_id} refers to future value time {value_time_epoch} at decision time {decision_at_epoch}"
            ),
            Self::InvalidFeatureChronology { feature_id } => {
                write!(f, "invalid chronology for feature {feature_id}")
            }
            Self::TooManyAssignments => write!(f, "event assignment count exceeds the declared limit"),
            Self::ArithmeticOverflow => write!(f, "canonical manifest arithmetic overflowed"),
        }
    }
}

impl std::error::Error for DataContractError {}

/// Validate source/partition completeness and calculate a canonical domain-separated
/// digest. Sources and partitions are sorted before encoding; input vector order does
/// not change identity. Any schema change requires a new version string.
pub fn canonical_manifest_digest(
    manifest: &PublicDataManifest,
) -> Result<String, DataContractError> {
    validate_manifest(manifest)?;

    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"symthaea.clean-airline.public-data-manifest.v1\0");
    put_string(&mut bytes, &manifest.manifest_id);
    put_string(&mut bytes, &manifest.manifest_version);
    put_i64(&mut bytes, manifest.created_at_epoch);

    let mut sources = manifest.sources.iter().collect::<Vec<_>>();
    sources.sort_by(|left, right| left.source_id.cmp(&right.source_id));
    put_u32(&mut bytes, usize_to_u32(sources.len())?);
    for source in sources {
        put_string(&mut bytes, &source.source_id);
        put_string(&mut bytes, &source.publisher);
        put_string(&mut bytes, &source.dataset_name);
        put_string(&mut bytes, &source.release_id);
        put_string(&mut bytes, &source.schema_or_methodology_version);
        put_string(&mut bytes, &source.canonical_url);
        put_i64(&mut bytes, source.coverage_start_epoch);
        put_i64(&mut bytes, source.coverage_end_epoch);
        put_i64(&mut bytes, source.catalogued_at_epoch);
        match source.dataset_downloaded_at_epoch {
            Some(value) => {
                bytes.push(1);
                put_i64(&mut bytes, value);
            }
            None => bytes.push(0),
        }
        bytes.push(encode_acquisition_status(source.acquisition_status));
        bytes.push(encode_license_status(source.license_status));
    }

    let mut partitions = manifest.partitions.iter().collect::<Vec<_>>();
    partitions.sort_by(|left, right| {
        left.source_id
            .cmp(&right.source_id)
            .then_with(|| left.start_epoch.cmp(&right.start_epoch))
            .then_with(|| left.partition_id.cmp(&right.partition_id))
    });
    put_u32(&mut bytes, usize_to_u32(partitions.len())?);
    for partition in partitions {
        put_string(&mut bytes, &partition.source_id);
        put_string(&mut bytes, &partition.partition_id);
        put_i64(&mut bytes, partition.start_epoch);
        put_i64(&mut bytes, partition.end_epoch);
        put_i64(&mut bytes, partition.downloaded_at_epoch);
        bytes.extend_from_slice(&partition.byte_len.to_le_bytes());
        put_string(&mut bytes, &partition.content_digest);
    }

    Ok(sha256_hex(&bytes))
}

pub fn validate_manifest(manifest: &PublicDataManifest) -> Result<(), DataContractError> {
    require_id(&manifest.manifest_id, "manifest ID")?;
    require_id(&manifest.manifest_version, "manifest version")?;
    if manifest.created_at_epoch < 0 {
        return Err(DataContractError::InvalidTimestamp("manifest creation time"));
    }
    if manifest.sources.is_empty() {
        return Err(DataContractError::MissingIdentifier("sources"));
    }
    if manifest.sources.len() > MAX_SOURCES {
        return Err(DataContractError::TooManySources);
    }
    if manifest.partitions.len() > MAX_PARTITIONS {
        return Err(DataContractError::TooManyPartitions);
    }

    let mut sources_by_id = BTreeMap::<&str, &DatasetSource>::new();
    for source in &manifest.sources {
        require_id(&source.source_id, "source ID")?;
        require_id(&source.publisher, "publisher")?;
        require_id(&source.dataset_name, "dataset name")?;
        require_id(&source.release_id, "release ID")?;
        require_id(&source.schema_or_methodology_version, "schema/methodology version")?;
        require_id(&source.canonical_url, "canonical URL")?;
        let url = source.canonical_url.as_str();
        if !url.starts_with("https://")
            || url.len() <= "https://".len()
            || url.chars().any(char::is_whitespace)
        {
            return Err(DataContractError::InvalidUrl);
        }
        if source.coverage_start_epoch < 0
            || source.coverage_end_epoch <= source.coverage_start_epoch
        {
            return Err(DataContractError::InvalidCoverage("source"));
        }
        if source.catalogued_at_epoch < 0 || source.catalogued_at_epoch > manifest.created_at_epoch {
            return Err(DataContractError::InvalidTimestamp("source catalogued time"));
        }
        if let Some(downloaded_at) = source.dataset_downloaded_at_epoch {
            if downloaded_at < source.catalogued_at_epoch || downloaded_at > manifest.created_at_epoch {
                return Err(DataContractError::InvalidTimestamp("dataset download time"));
            }
        }
        if sources_by_id.insert(source.source_id.as_str(), source).is_some() {
            return Err(DataContractError::DuplicateIdentifier("source ID"));
        }
    }

    let mut partitions_by_source: BTreeMap<&str, Vec<&DatasetPartition>> = BTreeMap::new();
    let mut partition_ids = BTreeSet::new();
    for partition in &manifest.partitions {
        require_id(&partition.source_id, "partition source ID")?;
        require_id(&partition.partition_id, "partition ID")?;
        if !partition_ids.insert(partition.partition_id.as_str()) {
            return Err(DataContractError::DuplicateIdentifier("partition ID"));
        }
        if partition.start_epoch < 0 || partition.end_epoch <= partition.start_epoch {
            return Err(DataContractError::InvalidCoverage("partition"));
        }
        if partition.downloaded_at_epoch < 0 || partition.downloaded_at_epoch > manifest.created_at_epoch {
            return Err(DataContractError::InvalidTimestamp("partition download time"));
        }
        if partition.byte_len == 0 {
            return Err(DataContractError::InvalidByteLength {
                partition_id: partition.partition_id.clone(),
            });
        }
        if !is_valid_sha256(&partition.content_digest) {
            return Err(DataContractError::InvalidDigest { field: "partition content" });
        }
        let source = sources_by_id
            .get(partition.source_id.as_str())
            .copied()
            .ok_or_else(|| DataContractError::UnknownPartitionSource {
                source_id: partition.source_id.clone(),
            })?;
        if source.acquisition_status != AcquisitionStatus::Retrieved {
            return Err(DataContractError::PartitionSourceNotRetrieved {
                source_id: source.source_id.clone(),
            });
        }
        if partition.start_epoch < source.coverage_start_epoch
            || partition.end_epoch > source.coverage_end_epoch
        {
            return Err(DataContractError::PartitionOutOfSourceCoverage {
                partition_id: partition.partition_id.clone(),
            });
        }
        if source
            .dataset_downloaded_at_epoch
            .is_none_or(|downloaded_at| downloaded_at > partition.downloaded_at_epoch)
        {
            return Err(DataContractError::InvalidTimestamp("partition predates declared dataset retrieval"));
        }
        partitions_by_source
            .entry(partition.source_id.as_str())
            .or_default()
            .push(partition);
    }

    for source in &manifest.sources {
        let partitions = partitions_by_source.get(source.source_id.as_str());
        match source.acquisition_status {
            AcquisitionStatus::Retrieved => {
                if source.dataset_downloaded_at_epoch.is_none() {
                    return Err(DataContractError::InvalidTimestamp("retrieved dataset time"));
                }
                let Some(partitions) = partitions else {
                    return Err(DataContractError::MissingRetrievedPartition {
                        source_id: source.source_id.clone(),
                    });
                };
                let mut ordered = partitions.clone();
                ordered.sort_by(|left, right| {
                    left.start_epoch
                        .cmp(&right.start_epoch)
                        .then_with(|| left.end_epoch.cmp(&right.end_epoch))
                });
                for pair in ordered.windows(2) {
                    if pair[1].start_epoch < pair[0].end_epoch {
                        return Err(DataContractError::OverlappingSourcePartitions {
                            source_id: source.source_id.clone(),
                        });
                    }
                }
            }
            AcquisitionStatus::Catalogued | AcquisitionStatus::AccessUnavailable | AcquisitionStatus::Excluded => {
                if source.dataset_downloaded_at_epoch.is_some() || partitions.is_some() {
                    return Err(DataContractError::UnexpectedPartitionForUnretrievedSource {
                        source_id: source.source_id.clone(),
                    });
                }
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SplitLabel {
    Train,
    Calibration,
    Validation,
    Test,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitWindow {
    /// Half-open interval [start, end) in Unix seconds UTC.
    pub start_epoch: i64,
    pub end_epoch: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemporalSplitContract {
    pub contract_id: String,
    pub train: SplitWindow,
    pub calibration: SplitWindow,
    pub validation: SplitWindow,
    pub test: SplitWindow,
}

impl TemporalSplitContract {
    pub fn validate(&self) -> Result<(), DataContractError> {
        require_id(&self.contract_id, "temporal split contract ID")?;
        let windows = [
            (SplitLabel::Train, self.train),
            (SplitLabel::Calibration, self.calibration),
            (SplitLabel::Validation, self.validation),
            (SplitLabel::Test, self.test),
        ];
        for (label, window) in windows {
            if window.start_epoch < 0 || window.end_epoch <= window.start_epoch {
                return Err(DataContractError::InvalidSplitWindow(split_name(label)));
            }
        }
        if self.train.end_epoch > self.calibration.start_epoch
            || self.calibration.end_epoch > self.validation.start_epoch
            || self.validation.end_epoch > self.test.start_epoch
        {
            return Err(DataContractError::OverlappingSplitWindows);
        }
        Ok(())
    }

    pub fn label_at(&self, timestamp_epoch: i64) -> Result<Option<SplitLabel>, DataContractError> {
        self.validate()?;
        if timestamp_epoch < 0 {
            return Err(DataContractError::InvalidTimestamp("split lookup time"));
        }
        for (label, window) in [
            (SplitLabel::Train, self.train),
            (SplitLabel::Calibration, self.calibration),
            (SplitLabel::Validation, self.validation),
            (SplitLabel::Test, self.test),
        ] {
            if timestamp_epoch >= window.start_epoch && timestamp_epoch < window.end_epoch {
                return Ok(Some(label));
            }
        }
        Ok(None)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventClusterAssignment {
    pub event_cluster_id: String,
    pub split: SplitLabel,
}

/// Every flight/disruption cluster must live in exactly one partition even if it
/// contains many row-level assignments.
pub fn validate_event_cluster_splits(
    assignments: &[EventClusterAssignment],
) -> Result<(), DataContractError> {
    if assignments.len() > MAX_SPLIT_ASSIGNMENTS {
        return Err(DataContractError::TooManyAssignments);
    }
    let mut owner_by_cluster = BTreeMap::<&str, SplitLabel>::new();
    for assignment in assignments {
        require_id(&assignment.event_cluster_id, "event cluster ID")?;
        match owner_by_cluster.get(assignment.event_cluster_id.as_str()) {
            Some(owner) if *owner != assignment.split => {
                return Err(DataContractError::EventClusterLeakage {
                    event_cluster_id: assignment.event_cluster_id.clone(),
                    first_split: *owner,
                    conflicting_split: assignment.split,
                });
            }
            None => {
                owner_by_cluster.insert(assignment.event_cluster_id.as_str(), assignment.split);
            }
            _ => {}
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureValueKind {
    Observed,
    Forecast,
}

/// As-of-decision metadata for a feature used by an optimizer/policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureAvailability {
    pub feature_id: String,
    pub source_id: String,
    pub source_digest: String,
    pub value_kind: FeatureValueKind,
    /// When the observation/forecast was produced.
    pub information_time_epoch: i64,
    /// When the policy could actually access the value.
    pub available_at_epoch: i64,
    /// Time the value describes. For observed values this must not be future time.
    pub value_time_epoch: i64,
}

/// Validate that a feature was available by the decision time and that observed values
/// do not describe the future. A forecast may target future time if it was issued and
/// received by the decision time.
pub fn validate_as_of_feature(
    feature: &FeatureAvailability,
    decision_at_epoch: i64,
) -> Result<(), DataContractError> {
    require_id(&feature.feature_id, "feature ID")?;
    require_id(&feature.source_id, "feature source ID")?;
    if !is_valid_sha256(&feature.source_digest) {
        return Err(DataContractError::InvalidDigest { field: "feature source" });
    }
    if decision_at_epoch < 0
        || feature.information_time_epoch < 0
        || feature.available_at_epoch < 0
        || feature.value_time_epoch < 0
        || feature.information_time_epoch > feature.available_at_epoch
    {
        return Err(DataContractError::InvalidFeatureChronology {
            feature_id: feature.feature_id.clone(),
        });
    }
    if feature.available_at_epoch > decision_at_epoch {
        return Err(DataContractError::FutureInformationLeakage {
            feature_id: feature.feature_id.clone(),
            decision_at_epoch,
            available_at_epoch: feature.available_at_epoch,
        });
    }
    if feature.value_kind == FeatureValueKind::Observed
        && feature.value_time_epoch > decision_at_epoch
    {
        return Err(DataContractError::FutureObservedValue {
            feature_id: feature.feature_id.clone(),
            decision_at_epoch,
            value_time_epoch: feature.value_time_epoch,
        });
    }
    Ok(())
}

fn is_valid_sha256(value: &str) -> bool {
    let Some(hex) = value.strip_prefix("sha256:") else {
        return false;
    };
    hex.len() == 64
        && hex.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_id(value: &str, field: &'static str) -> Result<(), DataContractError> {
    if value.trim().is_empty() {
        return Err(DataContractError::MissingIdentifier(field));
    }
    Ok(())
}

fn split_name(label: SplitLabel) -> &'static str {
    match label {
        SplitLabel::Train => "train",
        SplitLabel::Calibration => "calibration",
        SplitLabel::Validation => "validation",
        SplitLabel::Test => "test",
    }
}

fn encode_acquisition_status(status: AcquisitionStatus) -> u8 {
    match status {
        AcquisitionStatus::Catalogued => 0,
        AcquisitionStatus::Retrieved => 1,
        AcquisitionStatus::AccessUnavailable => 2,
        AcquisitionStatus::Excluded => 3,
    }
}

fn encode_license_status(status: LicenseStatus) -> u8 {
    match status {
        LicenseStatus::VerifiedOpen => 0,
        LicenseStatus::TermsReviewed => 1,
        LicenseStatus::Restricted => 2,
        LicenseStatus::Unknown => 3,
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut result = String::with_capacity(7 + digest.len() * 2);
    result.push_str("sha256:");
    for byte in digest.iter() {
        use std::fmt::Write as _;
        write!(&mut result, "{byte:02x}").expect("writing to String cannot fail");
    }
    result
}

fn usize_to_u32(value: usize) -> Result<u32, DataContractError> {
    u32::try_from(value).map_err(|_| DataContractError::ArithmeticOverflow)
}

fn put_i64(bytes: &mut Vec<u8>, value: i64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn put_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn put_string(bytes: &mut Vec<u8>, value: &str) {
    put_u64(bytes, value.len() as u64);
    bytes.extend_from_slice(value.as_bytes());
}
