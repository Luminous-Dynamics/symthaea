//! Provenance and licensing contracts for real temporal sensor datasets.
//!
//! The manifest contains metadata and file digests only; dataset binaries and row-level
//! participant information remain outside the public repository. Source-family split
//! assignments are fixed before windowing so overlapping windows from one subject, machine,
//! speaker or recording family cannot cross evaluation partitions.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::error::Error;
use std::fmt;

use super::digest::CanonicalHasher;

const SCHEMA: &str = "symthaea.temporal.real-dataset-manifest";
const VERSION: &str = "0.1.0";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DatasetManifestError {
    pub field: String,
    pub message: String,
}

impl DatasetManifestError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self { field: field.into(), message: message.into() }
    }
}

impl fmt::Display for DatasetManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}
impl Error for DatasetManifestError {}

fn valid_digest(value: &[u8; 32]) -> bool {
    value.iter().any(|byte| *byte != 0)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(u8)]
pub enum RealDatasetId {
    Harth = 1,
    SpikingHeidelbergDigits = 2,
    SpikingSpeechCommands = 3,
    Mimii = 4,
    Custom = 255,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(u8)]
pub enum DatasetLicense {
    CcBy4 = 1,
    CcBySa4 = 2,
    CcByNcNd4 = 3,
    CustomReviewed = 255,
}

impl DatasetLicense {
    pub fn spdx_id(self) -> &'static str {
        match self {
            Self::CcBy4 => "CC-BY-4.0",
            Self::CcBySa4 => "CC-BY-SA-4.0",
            Self::CcByNcNd4 => "CC-BY-NC-ND-4.0",
            Self::CustomReviewed => "LicenseRef-Custom-Reviewed",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(u8)]
pub enum DatasetUsageIntent {
    CommercialResearchOrDevelopment = 1,
    NonCommercialResearch = 2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(u8)]
pub enum DatasetTimestampRepresentation {
    Iso8601Rfc3339 = 1,
    UnixNanoseconds = 2,
    UnixMicroseconds = 3,
    SampleIndex = 4,
    EventRelativeMicroseconds = 5,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(u8)]
pub enum DatasetSplit {
    Training = 0,
    Development = 1,
    Calibration = 2,
    Confirmatory = 3,
}

impl DatasetSplit {
    const ALL: [Self; 4] = [
        Self::Training,
        Self::Development,
        Self::Calibration,
        Self::Confirmatory,
    ];
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DatasetFileEntry {
    /// Relative path inside the pinned source archive. No absolute paths or traversal.
    pub relative_path: String,
    pub byte_length: u64,
    pub sha256: [u8; 32],
    /// Subject, speaker, machine model, recording session or other preregistered group.
    pub source_family_id: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DatasetSourceFamilySplit {
    pub source_family_id: u64,
    pub split: DatasetSplit,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DatasetLabel {
    pub label_id: i64,
    pub label_name: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RealTemporalDatasetManifest {
    pub schema: String,
    pub version: String,
    pub dataset_id: RealDatasetId,
    pub dataset_version: String,
    pub doi: String,
    pub source_uri: String,
    pub archive_sha256: [u8; 32],
    pub license: DatasetLicense,
    pub license_uri: String,
    pub usage_intent: DatasetUsageIntent,
    /// Explicit acknowledgement of CC BY-SA obligations for distributed adapted material.
    pub share_alike_obligations_acknowledged: bool,
    /// Whether adapted dataset material will be redistributed as part of this work.
    pub adapted_dataset_material_distributed: bool,
    /// Required for CustomReviewed licenses; digest of counsel/maintainer review record.
    pub license_review_sha256: Option<[u8; 32]>,
    pub timestamp_representation: DatasetTimestampRepresentation,
    /// Sample rate in milli-Hz when a fixed nominal rate exists (e.g. 50 Hz = 50_000).
    pub nominal_sample_rate_millihz: Option<u64>,
    pub timestamp_column: String,
    pub grouping_column: String,
    pub channel_names: Vec<String>,
    pub label_map: Vec<DatasetLabel>,
    pub schema_sha256: [u8; 32],
    pub preprocessing_sha256: [u8; 32],
    pub files: Vec<DatasetFileEntry>,
    pub source_family_splits: Vec<DatasetSourceFamilySplit>,
    pub manifest_sha256: [u8; 32],
}

fn hash_str(hasher: &mut CanonicalHasher, value: &str) {
    hasher.bytes(value.as_bytes());
}

pub fn compute_real_temporal_dataset_manifest_sha256(
    manifest: &RealTemporalDatasetManifest,
) -> [u8; 32] {
    let mut hasher = CanonicalHasher::new(b"symthaea.temporal.real-dataset-manifest.v0.1");
    hash_str(&mut hasher, &manifest.schema);
    hash_str(&mut hasher, &manifest.version);
    hasher.u8(manifest.dataset_id as u8);
    hash_str(&mut hasher, &manifest.dataset_version);
    hash_str(&mut hasher, &manifest.doi);
    hash_str(&mut hasher, &manifest.source_uri);
    hasher.bytes(&manifest.archive_sha256);
    hasher.u8(manifest.license as u8);
    hash_str(&mut hasher, &manifest.license_uri);
    hasher.u8(manifest.usage_intent as u8);
    hasher.u8(u8::from(manifest.share_alike_obligations_acknowledged));
    hasher.u8(u8::from(manifest.adapted_dataset_material_distributed));
    match manifest.license_review_sha256 {
        Some(digest) => {
            hasher.u8(1);
            hasher.bytes(&digest);
        }
        None => hasher.u8(0),
    }
    hasher.u8(manifest.timestamp_representation as u8);
    match manifest.nominal_sample_rate_millihz {
        Some(rate) => {
            hasher.u8(1);
            hasher.u64(rate);
        }
        None => hasher.u8(0),
    }
    hash_str(&mut hasher, &manifest.timestamp_column);
    hash_str(&mut hasher, &manifest.grouping_column);
    hasher.usize(manifest.channel_names.len());
    for channel in &manifest.channel_names {
        hash_str(&mut hasher, channel);
    }
    hasher.usize(manifest.label_map.len());
    for label in &manifest.label_map {
        hasher.u64(label.label_id as u64);
        hash_str(&mut hasher, &label.label_name);
    }
    hasher.bytes(&manifest.schema_sha256);
    hasher.bytes(&manifest.preprocessing_sha256);
    hasher.usize(manifest.files.len());
    for file in &manifest.files {
        hash_str(&mut hasher, &file.relative_path);
        hasher.u64(file.byte_length);
        hasher.bytes(&file.sha256);
        hasher.u64(file.source_family_id);
    }
    hasher.usize(manifest.source_family_splits.len());
    for assignment in &manifest.source_family_splits {
        hasher.u64(assignment.source_family_id);
        hasher.u8(assignment.split as u8);
    }
    hasher.finish()
}

fn expected_license(dataset_id: RealDatasetId) -> Option<DatasetLicense> {
    match dataset_id {
        RealDatasetId::Harth
        | RealDatasetId::SpikingHeidelbergDigits
        | RealDatasetId::SpikingSpeechCommands => Some(DatasetLicense::CcBy4),
        RealDatasetId::Mimii => Some(DatasetLicense::CcBySa4),
        RealDatasetId::Custom => None,
    }
}

fn validate_license_policy(
    manifest: &RealTemporalDatasetManifest,
) -> Result<(), DatasetManifestError> {
    if let Some(expected) = expected_license(manifest.dataset_id) {
        if manifest.license != expected {
            return Err(DatasetManifestError::new(
                "license",
                format!("dataset source policy expects {:?}, not {:?}", expected, manifest.license),
            ));
        }
    }
    match manifest.license {
        DatasetLicense::CcBy4 => {
            if manifest.license_uri != "https://creativecommons.org/licenses/by/4.0/" {
                return Err(DatasetManifestError::new("license_uri", "CC BY 4.0 URI mismatch"));
            }
        }
        DatasetLicense::CcBySa4 => {
            if manifest.license_uri != "https://creativecommons.org/licenses/by-sa/4.0/" {
                return Err(DatasetManifestError::new("license_uri", "CC BY-SA 4.0 URI mismatch"));
            }
            if manifest.adapted_dataset_material_distributed
                && !manifest.share_alike_obligations_acknowledged
            {
                return Err(DatasetManifestError::new(
                    "share_alike_obligations_acknowledged",
                    "redistributed adapted dataset material requires explicit share-alike acknowledgement",
                ));
            }
        }
        DatasetLicense::CcByNcNd4 => {
            if manifest.license_uri != "https://creativecommons.org/licenses/by-nc-nd/4.0/" {
                return Err(DatasetManifestError::new("license_uri", "CC BY-NC-ND 4.0 URI mismatch"));
            }
            if manifest.usage_intent != DatasetUsageIntent::NonCommercialResearch {
                return Err(DatasetManifestError::new(
                    "usage_intent",
                    "CC BY-NC-ND material is not admitted to commercial product/development use",
                ));
            }
            if manifest.adapted_dataset_material_distributed {
                return Err(DatasetManifestError::new(
                    "adapted_dataset_material_distributed",
                    "CC BY-NC-ND does not permit distribution of adapted material",
                ));
            }
        }
        DatasetLicense::CustomReviewed => {
            let digest = manifest.license_review_sha256.as_ref().ok_or_else(|| {
                DatasetManifestError::new("license_review_sha256", "custom license requires a pinned review record")
            })?;
            if !valid_digest(digest) {
                return Err(DatasetManifestError::new("license_review_sha256", "must be non-zero SHA-256"));
            }
        }
    }
    Ok(())
}

fn validate_relative_path(path: &str) -> bool {
    if path.trim().is_empty()
        || path.starts_with('/')
        || path.starts_with('\\\\')
        || path.contains('\\\\')
        || path.contains(':')
        || path.contains('\\0')
    {
        return false;
    }
    // Validate separators independent of the host OS. In particular, a Linux parser must
    // reject Windows drive/UNC/backslash traversal before the archive is extracted elsewhere.
    path.split('/')
        .all(|component| !component.is_empty() && component != "." && component != "..")
}

pub fn validate_real_temporal_dataset_manifest(
    manifest: &RealTemporalDatasetManifest,
) -> Result<(), DatasetManifestError> {
    if manifest.schema != SCHEMA || manifest.version != VERSION {
        return Err(DatasetManifestError::new("schema_version", "unsupported manifest schema/version"));
    }
    for (field, value) in [
        ("dataset_version", manifest.dataset_version.as_str()),
        ("doi", manifest.doi.as_str()),
        ("source_uri", manifest.source_uri.as_str()),
        ("license_uri", manifest.license_uri.as_str()),
        ("timestamp_column", manifest.timestamp_column.as_str()),
        ("grouping_column", manifest.grouping_column.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(DatasetManifestError::new(field, "must be non-empty"));
        }
    }
    if !manifest.source_uri.starts_with("https://") || !manifest.license_uri.starts_with("https://") {
        return Err(DatasetManifestError::new("source_uri", "source and license must use HTTPS"));
    }
    validate_license_policy(manifest)?;
    for (field, digest) in [
        ("archive_sha256", &manifest.archive_sha256),
        ("schema_sha256", &manifest.schema_sha256),
        ("preprocessing_sha256", &manifest.preprocessing_sha256),
    ] {
        if !valid_digest(digest) {
            return Err(DatasetManifestError::new(field, "must be non-zero SHA-256"));
        }
    }
    match (
        manifest.timestamp_representation,
        manifest.nominal_sample_rate_millihz,
    ) {
        (DatasetTimestampRepresentation::SampleIndex, Some(rate)) if rate > 0 => {}
        (DatasetTimestampRepresentation::SampleIndex, _) => {
            return Err(DatasetManifestError::new("nominal_sample_rate_millihz", "sample-index timestamps require a positive nominal sample rate"));
        }
        (_, Some(0)) => {
            return Err(DatasetManifestError::new("nominal_sample_rate_millihz", "must be positive when supplied"));
        }
        _ => {}
    }
    if manifest.channel_names.is_empty()
        || manifest.channel_names.iter().any(|name| name.trim().is_empty())
        || manifest.channel_names.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(DatasetManifestError::new("channel_names", "must be non-empty, unique and sorted"));
    }
    if manifest.label_map.is_empty()
        || manifest.label_map.iter().any(|label| label.label_name.trim().is_empty())
        || manifest.label_map.windows(2).any(|pair| pair[0].label_id >= pair[1].label_id)
    {
        return Err(DatasetManifestError::new("label_map", "must be non-empty, unique and sorted by label ID"));
    }
    if manifest.files.is_empty()
        || manifest.files.windows(2).any(|pair| pair[0].relative_path >= pair[1].relative_path)
    {
        return Err(DatasetManifestError::new("files", "file inventory must be non-empty, unique and sorted by relative path"));
    }
    let mut file_families = BTreeSet::new();
    for (index, file) in manifest.files.iter().enumerate() {
        if !validate_relative_path(&file.relative_path)
            || file.byte_length == 0
            || !valid_digest(&file.sha256)
        {
            return Err(DatasetManifestError::new(format!("files[{index}]"), "path, length or digest is invalid"));
        }
        file_families.insert(file.source_family_id);
    }

    if manifest.source_family_splits.is_empty()
        || manifest.source_family_splits.windows(2).any(|pair| {
            pair[0].source_family_id >= pair[1].source_family_id
        })
    {
        return Err(DatasetManifestError::new("source_family_splits", "must be non-empty, unique and sorted by source family"));
    }
    let mut assignments = HashMap::new();
    let mut split_coverage = BTreeSet::new();
    for (index, assignment) in manifest.source_family_splits.iter().enumerate() {
        if !file_families.contains(&assignment.source_family_id)
            || assignments.insert(assignment.source_family_id, assignment.split).is_some()
        {
            return Err(DatasetManifestError::new(format!("source_family_splits[{index}]"), "source family is missing from file inventory or assigned more than once"));
        }
        split_coverage.insert(assignment.split);
    }
    for source_family_id in &file_families {
        if !assignments.contains_key(source_family_id) {
            return Err(DatasetManifestError::new(
                "source_family_splits",
                "every file's source family must be assigned before windowing",
            ));
        }
    }
    for split in [
        DatasetSplit::Training,
        DatasetSplit::Development,
        DatasetSplit::Calibration,
        DatasetSplit::Confirmatory,
    ] {
        if !split_coverage.contains(&split) {
            return Err(DatasetManifestError::new(
                "source_family_splits",
                format!("missing required {split:?} source-family partition"),
            ));
        }
    }

    if compute_real_temporal_dataset_manifest_sha256(manifest) != manifest.manifest_sha256 {
        return Err(DatasetManifestError::new("manifest_sha256", "does not match canonical manifest"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(value: u8) -> [u8; 32] {
        [value; 32]
    }

    fn harth_manifest() -> RealTemporalDatasetManifest {
        let files: Vec<_> = (0..4).map(|index| DatasetFileEntry {
            relative_path: format!("subject_{index:02}.csv"),
            byte_length: 1024 + index,
            sha256: hash(index as u8 + 1),
            source_family_id: index as u64 + 1,
        }).collect();
        let splits = vec![
            DatasetSourceFamilySplit { source_family_id: 1, split: DatasetSplit::Training },
            DatasetSourceFamilySplit { source_family_id: 2, split: DatasetSplit::Development },
            DatasetSourceFamilySplit { source_family_id: 3, split: DatasetSplit::Calibration },
            DatasetSourceFamilySplit { source_family_id: 4, split: DatasetSplit::Confirmatory },
        ];
        let mut manifest = RealTemporalDatasetManifest {
            schema: SCHEMA.into(),
            version: VERSION.into(),
            dataset_id: RealDatasetId::Harth,
            dataset_version: "UCI HARTH dataset-779".into(),
            doi: "10.24432/C5NC90".into(),
            source_uri: "https://archive.ics.uci.edu/dataset/779/harth".into(),
            archive_sha256: hash(11),
            license: DatasetLicense::CcBy4,
            license_uri: "https://creativecommons.org/licenses/by/4.0/".into(),
            usage_intent: DatasetUsageIntent::CommercialResearchOrDevelopment,
            share_alike_obligations_acknowledged: false,
            adapted_dataset_material_distributed: false,
            license_review_sha256: None,
            timestamp_representation: DatasetTimestampRepresentation::Iso8601Rfc3339,
            nominal_sample_rate_millihz: Some(50_000),
            timestamp_column: "timestamp".into(),
            grouping_column: "participant_id".into(),
            channel_names: vec![
                "back_x".into(), "back_y".into(), "back_z".into(),
                "thigh_x".into(), "thigh_y".into(), "thigh_z".into(),
            ],
            label_map: vec![
                DatasetLabel { label_id: 1, label_name: "walking".into() },
                DatasetLabel { label_id: 2, label_name: "running".into() },
            ],
            schema_sha256: hash(12),
            preprocessing_sha256: hash(13),
            files,
            source_family_splits: splits,
            manifest_sha256: [0; 32],
        };
        manifest.manifest_sha256 = compute_real_temporal_dataset_manifest_sha256(&manifest);
        manifest
    }

    #[test]
    fn harth_manifest_binds_archive_schema_preprocessing_and_group_splits() {
        let manifest = harth_manifest();
        validate_real_temporal_dataset_manifest(&manifest).unwrap();
    }

    #[test]
    fn mimii_share_alike_requires_explicit_acknowledgement_when_derivatives_are_distributed() {
        let mut manifest = harth_manifest();
        manifest.dataset_id = RealDatasetId::Mimii;
        manifest.dataset_version = "MIMII public 1.0".into();
        manifest.doi = "10.5281/zenodo.3384388".into();
        manifest.source_uri = "https://zenodo.org/records/3384388".into();
        manifest.license = DatasetLicense::CcBySa4;
        manifest.license_uri = "https://creativecommons.org/licenses/by-sa/4.0/".into();
        manifest.timestamp_representation = DatasetTimestampRepresentation::SampleIndex;
        manifest.nominal_sample_rate_millihz = Some(16_000_000);
        manifest.timestamp_column = "sample_index".into();
        manifest.channel_names = (0..8).map(|index| format!("microphone_{index}")).collect();
        manifest.label_map = vec![
            DatasetLabel { label_id: 0, label_name: "normal".into() },
            DatasetLabel { label_id: 1, label_name: "anomaly".into() },
        ];
        manifest.adapted_dataset_material_distributed = true;
        manifest.manifest_sha256 = compute_real_temporal_dataset_manifest_sha256(&manifest);
        assert!(validate_real_temporal_dataset_manifest(&manifest).is_err());

        manifest.share_alike_obligations_acknowledged = true;
        manifest.manifest_sha256 = compute_real_temporal_dataset_manifest_sha256(&manifest);
        validate_real_temporal_dataset_manifest(&manifest).unwrap();
    }

    #[test]
    fn noncommercial_no_derivatives_dataset_cannot_enter_commercial_use() {
        let mut manifest = harth_manifest();
        manifest.dataset_id = RealDatasetId::Custom;
        manifest.license = DatasetLicense::CcByNcNd4;
        manifest.license_uri = "https://creativecommons.org/licenses/by-nc-nd/4.0/".into();
        manifest.usage_intent = DatasetUsageIntent::CommercialResearchOrDevelopment;
        manifest.manifest_sha256 = compute_real_temporal_dataset_manifest_sha256(&manifest);
        assert!(validate_real_temporal_dataset_manifest(&manifest).is_err());

        manifest.usage_intent = DatasetUsageIntent::NonCommercialResearch;
        manifest.adapted_dataset_material_distributed = true;
        manifest.manifest_sha256 = compute_real_temporal_dataset_manifest_sha256(&manifest);
        assert!(validate_real_temporal_dataset_manifest(&manifest).is_err());
    }

    #[test]
    fn rejects_windows_and_unix_path_traversal_independent_of_host_os() {
        for invalid_path in [
            "../labels.csv",
            "subdir/../../labels.csv",
            "\\\\server\\share\\dataset.csv",
            "C:/dataset/subject.csv",
            "nested\\..\\labels.csv",
        ] {
            let mut manifest = harth_manifest();
            manifest.files[0].relative_path = invalid_path.into();
            manifest.manifest_sha256 = compute_real_temporal_dataset_manifest_sha256(&manifest);
            assert!(validate_real_temporal_dataset_manifest(&manifest).is_err(), "accepted {invalid_path:?}");
        }
    }

    #[test]
    fn rejects_source_family_overlap_and_unassigned_inventory_files() {
        let mut manifest = harth_manifest();
        manifest.source_family_splits[3].source_family_id = 1;
        manifest.manifest_sha256 = compute_real_temporal_dataset_manifest_sha256(&manifest);
        assert!(validate_real_temporal_dataset_manifest(&manifest).is_err());

        let mut manifest = harth_manifest();
        manifest.files[3].source_family_id = 9;
        manifest.manifest_sha256 = compute_real_temporal_dataset_manifest_sha256(&manifest);
        assert!(validate_real_temporal_dataset_manifest(&manifest).is_err());
    }

    #[test]
    fn rejects_path_traversal_duplicate_files_and_missing_sample_rate() {
        let mut manifest = harth_manifest();
        manifest.files[0].relative_path = "../labels.csv".into();
        manifest.manifest_sha256 = compute_real_temporal_dataset_manifest_sha256(&manifest);
        assert!(validate_real_temporal_dataset_manifest(&manifest).is_err());

        let mut manifest = harth_manifest();
        manifest.timestamp_representation = DatasetTimestampRepresentation::SampleIndex;
        manifest.nominal_sample_rate_millihz = None;
        manifest.manifest_sha256 = compute_real_temporal_dataset_manifest_sha256(&manifest);
        assert!(validate_real_temporal_dataset_manifest(&manifest).is_err());
    }
}
