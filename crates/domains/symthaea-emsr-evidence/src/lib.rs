//! Fail-closed evidence contracts for defensive EMSR research.
//!
//! This crate validates research inputs and execution receipts. It does not
//! implement an RF detector, infer hostile intent, authorize operational
//! actions, or qualify a safety-critical system.
//!
//! Design fixtures and measured run receipts are deliberately separate:
//! frozen catalogues/splits describe *what should be evaluated*; receipts
//! describe *what actually ran* against exact source and code identities.

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const CATALOG_SCHEMA: &str = "emsr.dataset-catalog";
const CATALOG_VERSION: &str = "0.1.0";
const SPLIT_SCHEMA: &str = "emsr.split-manifest";
const SPLIT_VERSION: &str = "0.1.0";
const RECEIPT_SCHEMA: &str = "emsr.run-receipt";
const RECEIPT_VERSION: &str = "0.1.0";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationError {
    pub path: String,
    pub message: String,
}

impl ValidationError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for ValidationError {}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn is_lower_hex(value: &str, expected_len: usize) -> bool {
    value.len() == expected_len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// SHA-256 of the exact byte sequence supplied by the caller.
///
/// Hash raw source bytes, not a parsed/re-serialized JSON representation.
/// This avoids silently changing line endings, whitespace, or encoding.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        // Writing into a String is infallible.
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

/// Verify an exact-byte SHA-256 digest.
pub fn verify_sha256(
    bytes: &[u8],
    expected: &str,
    path: &str,
) -> Result<(), ValidationError> {
    if !is_lower_hex(expected, 64) {
        return Err(ValidationError::new(
            path,
            "expected a lowercase 64-character SHA-256 digest",
        ));
    }
    let actual = sha256_hex(bytes);
    if actual != expected {
        return Err(ValidationError::new(
            path,
            format!("SHA-256 mismatch: expected {expected}, computed {actual}"),
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetKind {
    RecordedReceiver,
    RawIf,
    RawIq,
    PublishedMethod,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RightsState {
    Pending,
    Reviewed,
    Blocked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UsePurpose {
    Research,
    Commercial,
    Redistribution,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IngestionState {
    NotImported,
    MetadataOnly,
    Imported,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileDigest {
    pub path: String,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetRights {
    pub state: RightsState,
    pub permitted_uses: Vec<UsePurpose>,
    pub source_terms_uri: Option<String>,
    pub reviewed_by: Option<String>,
    pub reviewed_at: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetEntry {
    pub id: String,
    pub kind: DatasetKind,
    pub version: String,
    pub source_uri: String,
    pub rights: DatasetRights,
    pub intended_use: UsePurpose,
    pub ingestion: IngestionState,
    pub payload_files: Vec<FileDigest>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetCatalog {
    pub schema: String,
    pub version: String,
    pub state: String,
    pub entries: Vec<DatasetEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogSummary {
    pub entry_ids: Vec<String>,
    pub not_imported: usize,
    pub metadata_only: usize,
    pub imported: usize,
}

pub fn validate_dataset_catalog(input: &str) -> Result<CatalogSummary, ValidationError> {
    let catalog: DatasetCatalog = serde_json::from_str(input)
        .map_err(|error| ValidationError::new("$", format!("invalid catalogue JSON: {error}")))?;

    if catalog.schema != CATALOG_SCHEMA {
        return Err(ValidationError::new(
            "$.schema",
            format!("expected {CATALOG_SCHEMA:?}"),
        ));
    }
    if catalog.version != CATALOG_VERSION {
        return Err(ValidationError::new(
            "$.version",
            format!("unsupported catalogue version {:?}", catalog.version),
        ));
    }
    if catalog.state != "FROZEN_DESIGN" {
        return Err(ValidationError::new(
            "$.state",
            "frozen design catalogues must use FROZEN_DESIGN; measurements belong in run receipts",
        ));
    }
    if catalog.entries.is_empty() {
        return Err(ValidationError::new("$.entries", "catalogue must contain at least one entry"));
    }

    let mut ids = HashSet::new();
    let mut summary = CatalogSummary {
        entry_ids: Vec::with_capacity(catalog.entries.len()),
        not_imported: 0,
        metadata_only: 0,
        imported: 0,
    };

    for (index, entry) in catalog.entries.iter().enumerate() {
        let path = format!("$.entries[{index}]");
        if !nonempty(&entry.id) || !ids.insert(entry.id.as_str()) {
            return Err(ValidationError::new(
                format!("{path}.id"),
                "dataset IDs must be non-empty and unique",
            ));
        }
        if !nonempty(&entry.version)
            || entry.version.eq_ignore_ascii_case("latest")
            || entry.version.eq_ignore_ascii_case("unknown")
            || entry.version.eq_ignore_ascii_case("tbd")
        {
            return Err(ValidationError::new(
                format!("{path}.version"),
                "dataset version must be pinned, not latest/unknown/TBD",
            ));
        }
        if !entry.source_uri.starts_with("https://") {
            return Err(ValidationError::new(
                format!("{path}.source_uri"),
                "source URI must be HTTPS",
            ));
        }

        let rights = &entry.rights;
        if rights.state == RightsState::Blocked && entry.ingestion != IngestionState::NotImported {
            return Err(ValidationError::new(
                format!("{path}.ingestion"),
                "blocked source rights prohibit payload ingestion",
            ));
        }
        if rights.state == RightsState::Pending {
            if !rights.permitted_uses.is_empty() {
                return Err(ValidationError::new(
                    format!("{path}.rights.permitted_uses"),
                    "pending rights review cannot declare permitted uses",
                ));
            }
            if entry.ingestion == IngestionState::Imported {
                return Err(ValidationError::new(
                    format!("{path}.ingestion"),
                    "payload cannot be imported while source-level rights are pending",
                ));
            }
        }
        if rights.state == RightsState::Reviewed {
            if rights.source_terms_uri.as_deref().is_none_or(|uri| !uri.starts_with("https://")) {
                return Err(ValidationError::new(
                    format!("{path}.rights.source_terms_uri"),
                    "reviewed rights require an HTTPS source-terms reference",
                ));
            }
            if rights.reviewed_by.as_deref().is_none_or(|value| !nonempty(value))
                || rights.reviewed_at.as_deref().is_none_or(|value| !nonempty(value))
            {
                return Err(ValidationError::new(
                    format!("{path}.rights"),
                    "reviewed rights require reviewer identity and review timestamp",
                ));
            }
        }
        if entry.ingestion == IngestionState::Imported
            && (rights.state != RightsState::Reviewed
                || !rights.permitted_uses.contains(&entry.intended_use))
        {
            return Err(ValidationError::new(
                format!("{path}.rights.permitted_uses"),
                "import requires reviewed rights permitting the entry's exact intended use",
            ));
        }

        match entry.ingestion {
            IngestionState::NotImported | IngestionState::MetadataOnly => {
                if !entry.payload_files.is_empty() {
                    return Err(ValidationError::new(
                        format!("{path}.payload_files"),
                        "non-imported entries must not claim imported payload file digests",
                    ));
                }
            }
            IngestionState::Imported => {
                if entry.payload_files.is_empty() {
                    return Err(ValidationError::new(
                        format!("{path}.payload_files"),
                        "imported entries require per-file SHA-256 digests",
                    ));
                }
                let mut file_paths = HashSet::new();
                for (file_index, file) in entry.payload_files.iter().enumerate() {
                    let file_path = format!("{path}.payload_files[{file_index}]");
                    if !nonempty(&file.path) || !file_paths.insert(file.path.as_str()) {
                        return Err(ValidationError::new(
                            format!("{file_path}.path"),
                            "payload paths must be non-empty and unique within the dataset",
                        ));
                    }
                    if !is_lower_hex(&file.sha256, 64) {
                        return Err(ValidationError::new(
                            format!("{file_path}.sha256"),
                            "expected a lowercase 64-character SHA-256 digest",
                        ));
                    }
                }
            }
        }

        match entry.ingestion {
            IngestionState::NotImported => summary.not_imported += 1,
            IngestionState::MetadataOnly => summary.metadata_only += 1,
            IngestionState::Imported => summary.imported += 1,
        }
        summary.entry_ids.push(entry.id.clone());
    }

    summary.entry_ids.sort();
    Ok(summary)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataSplit {
    Train,
    Validation,
    Test,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SplitDimension {
    EventGroup,
    RecordingGroup,
    ReceiverGroup,
    CampaignGroup,
    LocationGroup,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplitRecord {
    pub record_id: String,
    pub split: DataSplit,
    pub event_group: String,
    pub recording_group: String,
    pub receiver_group: Option<String>,
    pub campaign_group: Option<String>,
    pub location_group: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplitManifest {
    pub schema: String,
    pub version: String,
    pub dataset_id: String,
    pub dataset_version: String,
    pub held_out_dimensions: Vec<SplitDimension>,
    pub records: Vec<SplitRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitSummary {
    pub record_count: usize,
    pub train_count: usize,
    pub validation_count: usize,
    pub test_count: usize,
    pub held_out_dimensions: Vec<SplitDimension>,
}

fn dimension_value<'a>(record: &'a SplitRecord, dimension: SplitDimension) -> Option<&'a str> {
    match dimension {
        SplitDimension::EventGroup => Some(&record.event_group),
        SplitDimension::RecordingGroup => Some(&record.recording_group),
        SplitDimension::ReceiverGroup => record.receiver_group.as_deref(),
        SplitDimension::CampaignGroup => record.campaign_group.as_deref(),
        SplitDimension::LocationGroup => record.location_group.as_deref(),
    }
}

pub fn validate_split_manifest(input: &str) -> Result<SplitSummary, ValidationError> {
    let manifest: SplitManifest = serde_json::from_str(input)
        .map_err(|error| ValidationError::new("$", format!("invalid split JSON: {error}")))?;

    if manifest.schema != SPLIT_SCHEMA {
        return Err(ValidationError::new("$.schema", format!("expected {SPLIT_SCHEMA:?}")));
    }
    if manifest.version != SPLIT_VERSION {
        return Err(ValidationError::new(
            "$.version",
            format!("unsupported split version {:?}", manifest.version),
        ));
    }
    if !nonempty(&manifest.dataset_id) || !nonempty(&manifest.dataset_version) {
        return Err(ValidationError::new(
            "$.dataset_id",
            "split manifest must bind a dataset ID and pinned version",
        ));
    }
    if manifest.records.is_empty() {
        return Err(ValidationError::new("$.records", "split manifest must contain records"));
    }

    let mut dimensions = vec![SplitDimension::EventGroup, SplitDimension::RecordingGroup];
    let mut declared = HashSet::new();
    for (index, dimension) in manifest.held_out_dimensions.iter().enumerate() {
        if !declared.insert(*dimension) {
            return Err(ValidationError::new(
                format!("$.held_out_dimensions[{index}]"),
                "held-out dimensions must be unique",
            ));
        }
        if !dimensions.contains(dimension) {
            dimensions.push(*dimension);
        }
    }

    let mut record_ids = HashSet::new();
    let mut assignments: HashMap<(SplitDimension, String), DataSplit> = HashMap::new();
    let mut summary = SplitSummary {
        record_count: manifest.records.len(),
        train_count: 0,
        validation_count: 0,
        test_count: 0,
        held_out_dimensions: dimensions.clone(),
    };

    for (index, record) in manifest.records.iter().enumerate() {
        let path = format!("$.records[{index}]");
        if !nonempty(&record.record_id) || !record_ids.insert(record.record_id.as_str()) {
            return Err(ValidationError::new(
                format!("{path}.record_id"),
                "record IDs must be non-empty and unique",
            ));
        }

        match record.split {
            DataSplit::Train => summary.train_count += 1,
            DataSplit::Validation => summary.validation_count += 1,
            DataSplit::Test => summary.test_count += 1,
        }

        for dimension in &dimensions {
            let group = dimension_value(record, *dimension).filter(|value| nonempty(value));
            let Some(group) = group else {
                return Err(ValidationError::new(
                    format!("{path}.{dimension:?}"),
                    "required grouping value is missing or empty",
                ));
            };
            let key = (*dimension, group.to_owned());
            if let Some(previous_split) = assignments.get(&key) {
                if *previous_split != record.split {
                    return Err(ValidationError::new(
                        format!("{path}.{dimension:?}"),
                        format!(
                            "group {group:?} crosses {:?} and {:?} partitions",
                            previous_split, record.split
                        ),
                    ));
                }
            } else {
                assignments.insert(key, record.split);
            }
        }
    }

    if summary.train_count == 0 || summary.validation_count == 0 || summary.test_count == 0 {
        return Err(ValidationError::new(
            "$.records",
            "train, validation, and test partitions must all be non-empty",
        ));
    }

    Ok(summary)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunState {
    Complete,
    Incomplete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaseOutcome {
    Pass,
    Fail,
    Indeterminate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationState {
    Pending,
    Verified,
    Failed,
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaseResult {
    pub case_id: String,
    pub outcome: CaseOutcome,
    pub reason_code: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndependentVerification {
    pub state: VerificationState,
    pub verifier_id: Option<String>,
    pub verifier_sha256: Option<String>,
    pub report_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunReceipt {
    pub schema: String,
    pub version: String,
    pub run_id: String,
    pub state: RunState,
    pub git_commit: String,
    pub dataset_catalog_sha256: String,
    pub dataset_payload_sha256: Vec<String>,
    pub split_manifest_sha256: String,
    pub code_sha256: String,
    pub model_sha256: Option<String>,
    pub configuration_sha256: String,
    pub toolchain: String,
    pub environment_sha256: String,
    pub seed: Option<u64>,
    pub cases: Vec<CaseResult>,
    pub independent_verification: IndependentVerification,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReceiptSummary {
    pub run_id: String,
    pub state: RunState,
    pub pass_count: usize,
    pub fail_count: usize,
    pub indeterminate_count: usize,
    pub verification_state: VerificationState,
}

/// Validate receipt structure and provenance completeness.
///
/// This validates the receipt contract only. It does not certify the scientific
/// validity of labels, the correctness of a detector, or operational safety.
pub fn validate_run_receipt(input: &str) -> Result<ReceiptSummary, ValidationError> {
    let receipt: RunReceipt = serde_json::from_str(input)
        .map_err(|error| ValidationError::new("$", format!("invalid run receipt JSON: {error}")))?;

    if receipt.schema != RECEIPT_SCHEMA {
        return Err(ValidationError::new("$.schema", format!("expected {RECEIPT_SCHEMA:?}")));
    }
    if receipt.version != RECEIPT_VERSION {
        return Err(ValidationError::new(
            "$.version",
            format!("unsupported receipt version {:?}", receipt.version),
        ));
    }
    if !nonempty(&receipt.run_id) {
        return Err(ValidationError::new("$.run_id", "run ID must be non-empty"));
    }
    if !is_lower_hex(&receipt.git_commit, 40) && !is_lower_hex(&receipt.git_commit, 64) {
        return Err(ValidationError::new(
            "$.git_commit",
            "expected a lowercase 40-character Git SHA-1 or 64-character Git SHA-256 object ID",
        ));
    }

    for (path, digest) in [
        ("$.dataset_catalog_sha256", receipt.dataset_catalog_sha256.as_str()),
        ("$.split_manifest_sha256", receipt.split_manifest_sha256.as_str()),
        ("$.code_sha256", receipt.code_sha256.as_str()),
        ("$.configuration_sha256", receipt.configuration_sha256.as_str()),
        ("$.environment_sha256", receipt.environment_sha256.as_str()),
    ] {
        if !is_lower_hex(digest, 64) {
            return Err(ValidationError::new(
                path,
                "expected a lowercase 64-character SHA-256 digest",
            ));
        }
    }
    if receipt.dataset_payload_sha256.is_empty() {
        return Err(ValidationError::new(
            "$.dataset_payload_sha256",
            "at least one exact dataset payload digest is required",
        ));
    }
    let mut payload_digests = HashSet::new();
    for (index, digest) in receipt.dataset_payload_sha256.iter().enumerate() {
        if !is_lower_hex(digest, 64) || !payload_digests.insert(digest.as_str()) {
            return Err(ValidationError::new(
                format!("$.dataset_payload_sha256[{index}]"),
                "payload digests must be valid, lowercase, unique SHA-256 values",
            ));
        }
    }
    if receipt.model_sha256.as_deref().is_some_and(|value| !is_lower_hex(value, 64)) {
        return Err(ValidationError::new(
            "$.model_sha256",
            "model digest must be a lowercase 64-character SHA-256 value when present",
        ));
    }
    if !nonempty(&receipt.toolchain) {
        return Err(ValidationError::new("$.toolchain", "toolchain identity is required"));
    }
    if receipt.cases.is_empty() {
        return Err(ValidationError::new("$.cases", "run receipt must contain case outcomes"));
    }

    let mut case_ids = HashSet::new();
    let mut summary = ReceiptSummary {
        run_id: receipt.run_id.clone(),
        state: receipt.state,
        pass_count: 0,
        fail_count: 0,
        indeterminate_count: 0,
        verification_state: receipt.independent_verification.state,
    };

    for (index, case) in receipt.cases.iter().enumerate() {
        if !nonempty(&case.case_id) || !case_ids.insert(case.case_id.as_str()) {
            return Err(ValidationError::new(
                format!("$.cases[{index}].case_id"),
                "case IDs must be non-empty and unique",
            ));
        }
        match case.outcome {
            CaseOutcome::Pass => summary.pass_count += 1,
            CaseOutcome::Fail => summary.fail_count += 1,
            CaseOutcome::Indeterminate => summary.indeterminate_count += 1,
        }
    }

    let verification = &receipt.independent_verification;
    match verification.state {
        VerificationState::Verified => {
            if receipt.state != RunState::Complete {
                return Err(ValidationError::new(
                    "$.independent_verification.state",
                    "an incomplete run cannot claim verified reconstruction",
                ));
            }
            if verification.verifier_id.as_deref().is_none_or(|value| !nonempty(value)) {
                return Err(ValidationError::new(
                    "$.independent_verification.verifier_id",
                    "verified reconstruction requires an independent verifier identity",
                ));
            }
            for (path, value) in [
                ("$.independent_verification.verifier_sha256", verification.verifier_sha256.as_deref()),
                ("$.independent_verification.report_sha256", verification.report_sha256.as_deref()),
            ] {
                if value.is_none_or(|digest| !is_lower_hex(digest, 64)) {
                    return Err(ValidationError::new(
                        path,
                        "verified reconstruction requires a lowercase SHA-256 digest",
                    ));
                }
            }
        }
        VerificationState::Pending | VerificationState::Failed | VerificationState::Indeterminate => {
            if verification.verifier_id.is_some()
                || verification.verifier_sha256.is_some()
                || verification.report_sha256.is_some()
            {
                return Err(ValidationError::new(
                    "$.independent_verification",
                    "non-verified states must not carry a partial verification attestation",
                ));
            }
        }
    }

    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample_catalog(ingestion: &str, rights_state: &str, permitted_uses: Vec<&str>) -> String {
        let payload_files: Vec<serde_json::Value> = if ingestion == "imported" {
            vec![json!({
                "path": "receiver.sbf",
                "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            })]
        } else {
            Vec::new()
        };

        json!({
            "schema": CATALOG_SCHEMA,
            "version": CATALOG_VERSION,
            "state": "FROZEN_DESIGN",
            "entries": [{
                "id": "JT-2025",
                "kind": "recorded_receiver",
                "version": "release-2026-09-30",
                "source_uri": "https://example.org/dataset",
                "rights": {
                    "state": rights_state,
                    "permitted_uses": permitted_uses,
                    "source_terms_uri": if rights_state == "reviewed" { Some("https://example.org/terms") } else { None::<&str> },
                    "reviewed_by": if rights_state == "reviewed" { Some("reviewer-1") } else { None::<&str> },
                    "reviewed_at": if rights_state == "reviewed" { Some("2026-11-06T00:00:00Z") } else { None::<&str> }
                },
                "intended_use": "research",
                "ingestion": ingestion,
                "payload_files": payload_files
            }]
        }).to_string()
    }

    #[test]
    fn accepts_discovery_only_catalogue_with_pending_rights() {
        let input = sample_catalog("not_imported", "pending", vec![]);
        let summary = validate_dataset_catalog(&input).expect("discovery-only catalogue is valid");
        assert_eq!(summary.not_imported, 1);
        assert_eq!(summary.imported, 0);
    }

    #[test]
    fn refuses_payload_import_while_rights_are_pending() {
        let input = sample_catalog("imported", "pending", vec![]);
        let error = validate_dataset_catalog(&input).expect_err("pending rights must fail closed");
        assert!(error.to_string().contains("pending"));
    }

    #[test]
    fn refuses_import_for_unpermitted_intended_use() {
        let input = sample_catalog("imported", "reviewed", vec!["redistribution"]);
        let error = validate_dataset_catalog(&input).expect_err("research use is not permitted");
        assert!(error.to_string().contains("exact intended use"));
    }

    #[test]
    fn rejects_event_or_recording_group_leakage() {
        let input = json!({
            "schema": SPLIT_SCHEMA,
            "version": SPLIT_VERSION,
            "dataset_id": "JT-2025",
            "dataset_version": "release-2026-09-30",
            "held_out_dimensions": ["receiver_group"],
            "records": [
                {"record_id":"r1","split":"train","event_group":"event-a","recording_group":"recording-a","receiver_group":"rx-a","campaign_group":"campaign-a","location_group":"site-a"},
                {"record_id":"r2","split":"test","event_group":"event-a","recording_group":"recording-b","receiver_group":"rx-b","campaign_group":"campaign-b","location_group":"site-b"},
                {"record_id":"r3","split":"validation","event_group":"event-c","recording_group":"recording-c","receiver_group":"rx-c","campaign_group":"campaign-c","location_group":"site-c"}
            ]
        }).to_string();
        let error = validate_split_manifest(&input).expect_err("same event must not cross partitions");
        assert!(error.to_string().contains("crosses"));
    }

    #[test]
    fn accepts_disjoint_event_recording_and_receiver_groups() {
        let input = json!({
            "schema": SPLIT_SCHEMA,
            "version": SPLIT_VERSION,
            "dataset_id": "JT-2025",
            "dataset_version": "release-2026-09-30",
            "held_out_dimensions": ["receiver_group","campaign_group"],
            "records": [
                {"record_id":"r1","split":"train","event_group":"event-a","recording_group":"recording-a","receiver_group":"rx-a","campaign_group":"campaign-a","location_group":"site-a"},
                {"record_id":"r2","split":"validation","event_group":"event-b","recording_group":"recording-b","receiver_group":"rx-b","campaign_group":"campaign-b","location_group":"site-b"},
                {"record_id":"r3","split":"test","event_group":"event-c","recording_group":"recording-c","receiver_group":"rx-c","campaign_group":"campaign-c","location_group":"site-c"}
            ]
        }).to_string();
        let summary = validate_split_manifest(&input).expect("disjoint split manifest is valid");
        assert_eq!(summary.record_count, 3);
        assert_eq!(summary.test_count, 1);
    }

    #[test]
    fn computes_and_verifies_sha256_over_exact_bytes() {
        let digest = sha256_hex(b"abc");
        assert_eq!(
            digest,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        verify_sha256(b"abc", &digest, "$.digest").expect("matching digest");
        assert!(verify_sha256(b"abd", &digest, "$.digest").is_err());
    }

    #[test]
    fn verified_receipt_requires_verifier_identity_and_artifacts() {
        let input = json!({
            "schema": RECEIPT_SCHEMA,
            "version": RECEIPT_VERSION,
            "run_id": "run-1",
            "state": "complete",
            "git_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "dataset_catalog_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "dataset_payload_sha256": ["cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"],
            "split_manifest_sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            "code_sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
            "model_sha256": null,
            "configuration_sha256": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            "toolchain": "rustc-1.96.0",
            "environment_sha256": "abababababababababababababababababababababababababababababababab",
            "seed": 7,
            "cases": [{"case_id":"case-1","outcome":"pass","reason_code":null}],
            "independent_verification": {
                "state":"verified",
                "verifier_id":null,
                "verifier_sha256":null,
                "report_sha256":null
            }
        }).to_string();
        let error = validate_run_receipt(&input).expect_err("verified state needs verifier evidence");
        assert!(error.to_string().contains("verifier identity"));
    }

    #[test]
    fn valid_receipt_is_not_a_safety_or_performance_qualification() {
        let input = json!({
            "schema": RECEIPT_SCHEMA,
            "version": RECEIPT_VERSION,
            "run_id": "run-2",
            "state": "complete",
            "git_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "dataset_catalog_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "dataset_payload_sha256": ["cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"],
            "split_manifest_sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            "code_sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
            "model_sha256": null,
            "configuration_sha256": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            "toolchain": "rustc-1.96.0",
            "environment_sha256": "abababababababababababababababababababababababababababababababab",
            "seed": 7,
            "cases": [{"case_id":"case-1","outcome":"pass","reason_code":null}],
            "independent_verification": {
                "state":"verified",
                "verifier_id":"independent-verifier-1",
                "verifier_sha256":"cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd",
                "report_sha256":"dededededededededededededededededededededededededededededededede"
            }
        }).to_string();
        let summary = validate_run_receipt(&input).expect("complete receipt contract");
        assert_eq!(summary.pass_count, 1);
        assert_eq!(summary.verification_state, VerificationState::Verified);
    }
}
