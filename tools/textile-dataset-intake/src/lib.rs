// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Offline, fail-closed verification of rights-reviewed textile dataset imports.
//!
//! This crate never downloads data. It verifies a local manifest, enforces the
//! declared use permission, checks path containment, file size and SHA-256, and
//! emits a deterministic receipt bound to the exact manifest and local files.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt::{Display, Formatter};
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Component, Path, PathBuf};

pub const MANIFEST_SCHEMA_VERSION: &str = "textile-data-intake-v0.1";
pub const RECEIPT_SCHEMA_VERSION: &str = "textile-data-intake-receipt-v0.1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntakeError(pub String);

impl Display for IntakeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for IntakeError {}

pub type Result<T> = std::result::Result<T, IntakeError>;

fn fail<T>(message: impl Into<String>) -> Result<T> {
    Err(IntakeError(message.into()))
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DatasetSource {
    pub title: String,
    pub publisher: String,
    pub landing_page: String,
    pub doi: Option<String>,
    pub version: String,
    pub retrieved_at_utc: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RightsStatus {
    Pending,
    Reviewed,
    Rejected,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RightsReview {
    pub status: RightsStatus,
    pub license_url: String,
    pub license_text_sha256: Option<String>,
    pub reviewed_by: Option<String>,
    pub reviewed_at_utc: Option<String>,
    pub permitted_uses: Vec<String>,
    pub attribution: String,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DatasetFile {
    /// Path/key as named by the upstream dataset.
    pub source_path: String,
    /// Relative path beneath the local dataset root; absolute paths are forbidden.
    pub local_path: String,
    /// SHA-256 supplied by the source manifest or sidecar, lowercase hexadecimal.
    pub sha256: String,
    pub size_bytes: u64,
    /// Examples: nir_absorbance, nir_reflectance, sample_metadata, calibration, image.
    pub role: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DatasetManifest {
    pub schema_version: String,
    pub dataset_id: String,
    pub dataset_version: String,
    pub source: DatasetSource,
    pub rights: RightsReview,
    pub files: Vec<DatasetFile>,
    pub claim_ceiling: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct VerifiedFile {
    pub source_path: String,
    pub local_path: String,
    pub sha256: String,
    pub size_bytes: u64,
    pub role: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct IntakeReceipt {
    pub receipt_schema_version: String,
    pub manifest_schema_version: String,
    pub dataset_id: String,
    pub dataset_version: String,
    pub source_doi: Option<String>,
    pub operation: String,
    pub manifest_sha256: String,
    pub verified_files: Vec<VerifiedFile>,
    pub claim_ceiling: String,
}

/// Hashes a byte slice and returns canonical lowercase SHA-256 hex.
pub fn sha256_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    hex_lower(&digest)
}

/// Hashes a file by streaming it; the whole file is never loaded into memory.
pub fn sha256_file(path: &Path) -> Result<String> {
    let file = File::open(path)
        .map_err(|error| IntakeError(format!("cannot open {}: {error}", path.display())))?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];

    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|error| IntakeError(format!("cannot read {}: {error}", path.display())))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }

    Ok(hex_lower(&hasher.finalize()))
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn validate_manifest_shape(manifest: &DatasetManifest) -> Result<()> {
    if manifest.schema_version != MANIFEST_SCHEMA_VERSION {
        return fail(format!(
            "unsupported manifest schema version {:?}",
            manifest.schema_version
        ));
    }
    for (name, value) in [
        ("dataset_id", manifest.dataset_id.as_str()),
        ("dataset_version", manifest.dataset_version.as_str()),
        ("source.title", manifest.source.title.as_str()),
        ("source.publisher", manifest.source.publisher.as_str()),
        ("source.landing_page", manifest.source.landing_page.as_str()),
        ("source.version", manifest.source.version.as_str()),
        ("source.retrieved_at_utc", manifest.source.retrieved_at_utc.as_str()),
        ("rights.license_url", manifest.rights.license_url.as_str()),
        ("rights.attribution", manifest.rights.attribution.as_str()),
        ("claim_ceiling", manifest.claim_ceiling.as_str()),
    ] {
        if !nonempty(value) {
            return fail(format!("{name} must be non-empty"));
        }
    }
    if !(manifest.source.landing_page.starts_with("https://")
        || manifest.source.landing_page.starts_with("http://"))
    {
        return fail("source.landing_page must be an HTTP(S) URL");
    }
    if !(manifest.rights.license_url.starts_with("https://")
        || manifest.rights.license_url.starts_with("http://"))
    {
        return fail("rights.license_url must be an HTTP(S) URL");
    }
    if manifest.files.is_empty() {
        return fail("manifest has no files; an empty source inventory cannot be qualified");
    }

    if manifest.rights.status == RightsStatus::Reviewed {
        if manifest
            .rights
            .reviewed_by
            .as_deref()
            .is_none_or(|value| !nonempty(value))
            || manifest
                .rights
                .reviewed_at_utc
                .as_deref()
                .is_none_or(|value| !nonempty(value))
        {
            return fail("reviewed rights status requires reviewer and review timestamp");
        }
        if manifest
            .rights
            .license_text_sha256
            .as_deref()
            .is_none_or(|value| !valid_sha256(value))
        {
            return fail("reviewed rights status requires the license-text SHA-256");
        }
    }
    if manifest.rights.status == RightsStatus::Pending
        && (manifest.rights.reviewed_by.is_some() || manifest.rights.reviewed_at_utc.is_some())
    {
        return fail("pending rights review cannot contain reviewer completion fields");
    }

    let mut source_paths = BTreeSet::new();
    let mut local_paths = BTreeSet::new();
    for (index, entry) in manifest.files.iter().enumerate() {
        if !nonempty(&entry.source_path) || !nonempty(&entry.local_path) || !nonempty(&entry.role) {
            return fail(format!("files[{index}] source_path, local_path and role must be non-empty"));
        }
        if entry.source_path.contains('\0') || entry.local_path.contains('\0') {
            return fail(format!("files[{index}] contains a NUL byte in a path"));
        }
        if !valid_sha256(&entry.sha256) {
            return fail(format!("files[{index}] sha256 must be 64 lowercase hexadecimal characters"));
        }
        if entry.size_bytes == 0 {
            return fail(format!("files[{index}] size_bytes must be positive"));
        }
        if !source_paths.insert(entry.source_path.as_str()) {
            return fail(format!("duplicate source_path {:?}", entry.source_path));
        }
        if !local_paths.insert(entry.local_path.as_str()) {
            return fail(format!("duplicate local_path {:?}", entry.local_path));
        }
        validate_relative_path(&entry.local_path)?;
    }
    Ok(())
}

fn validate_relative_path(value: &str) -> Result<()> {
    if value.contains('\\') {
        return fail(format!("local_path must use portable forward slashes: {value:?}"));
    }
    let path = Path::new(value);
    if path.as_os_str().is_empty() || path.is_absolute() {
        return fail(format!("local_path must be a non-empty relative path: {value:?}"));
    }
    for component in path.components() {
        match component {
            Component::Normal(_) => {}
            _ => return fail(format!("local_path contains a forbidden path component: {value:?}")),
        }
    }
    Ok(())
}

fn resolve_under_root(root: &Path, relative: &str) -> Result<PathBuf> {
    validate_relative_path(relative)?;
    let candidate = root.join(relative);
    let canonical = fs::canonicalize(&candidate).map_err(|error| {
        IntakeError(format!("cannot resolve local file {}: {error}", candidate.display()))
    })?;
    if !canonical.starts_with(root) {
        return fail(format!("local_path escapes dataset root through a symlink: {relative:?}"));
    }
    let metadata = fs::metadata(&canonical)
        .map_err(|error| IntakeError(format!("cannot inspect {}: {error}", canonical.display())))?;
    if !metadata.is_file() {
        return fail(format!("local_path is not a regular file: {relative:?}"));
    }
    Ok(canonical)
}

/// Validates rights, path containment, file lengths and hashes, then emits a
/// deterministic receipt bound to the exact manifest bytes and verified files.
pub fn build_receipt(
    manifest_path: &Path,
    dataset_root: &Path,
    operation: &str,
) -> Result<IntakeReceipt> {
    if !nonempty(operation) {
        return fail("operation must be non-empty");
    }
    let raw_manifest = fs::read(manifest_path).map_err(|error| {
        IntakeError(format!("cannot read manifest {}: {error}", manifest_path.display()))
    })?;
    let manifest: DatasetManifest = serde_json::from_slice(&raw_manifest)
        .map_err(|error| IntakeError(format!("manifest JSON is invalid: {error}")))?;
    validate_manifest_shape(&manifest)?;

    match manifest.rights.status {
        RightsStatus::Pending => {
            return fail("rights status is pending; no data qualification or benchmark receipt is allowed")
        }
        RightsStatus::Rejected => {
            return fail("rights status is rejected; this dataset cannot be used for the requested operation")
        }
        RightsStatus::Reviewed => {}
    }
    if !manifest
        .rights
        .permitted_uses
        .iter()
        .any(|permitted| permitted == operation)
    {
        return fail(format!(
            "operation {operation:?} is not present in the reviewed permitted_uses list"
        ));
    }

    let root = fs::canonicalize(dataset_root).map_err(|error| {
        IntakeError(format!("cannot resolve dataset root {}: {error}", dataset_root.display()))
    })?;
    if !root.is_dir() {
        return fail("dataset root is not a directory");
    }

    let mut verified_files = Vec::with_capacity(manifest.files.len());
    for entry in &manifest.files {
        let path = resolve_under_root(&root, &entry.local_path)?;
        let metadata = fs::metadata(&path)
            .map_err(|error| IntakeError(format!("cannot inspect {}: {error}", path.display())))?;
        if metadata.len() != entry.size_bytes {
            return fail(format!(
                "size mismatch for {}: manifest={} actual={}",
                entry.local_path,
                entry.size_bytes,
                metadata.len()
            ));
        }
        let actual = sha256_file(&path)?;
        if actual != entry.sha256 {
            return fail(format!(
                "SHA-256 mismatch for {}: manifest={} actual={actual}",
                entry.local_path, entry.sha256
            ));
        }
        verified_files.push(VerifiedFile {
            source_path: entry.source_path.clone(),
            local_path: entry.local_path.clone(),
            sha256: actual,
            size_bytes: metadata.len(),
            role: entry.role.clone(),
        });
    }

    Ok(IntakeReceipt {
        receipt_schema_version: RECEIPT_SCHEMA_VERSION.to_owned(),
        manifest_schema_version: manifest.schema_version,
        dataset_id: manifest.dataset_id,
        dataset_version: manifest.dataset_version,
        source_doi: manifest.source.doi,
        operation: operation.to_owned(),
        manifest_sha256: sha256_bytes(&raw_manifest),
        verified_files,
        claim_ceiling: manifest.claim_ceiling,
    })
}

/// Reconstructs the expected receipt and checks an existing receipt against it.
pub fn verify_receipt(
    manifest_path: &Path,
    dataset_root: &Path,
    receipt_path: &Path,
) -> Result<IntakeReceipt> {
    let raw_receipt = fs::read(receipt_path).map_err(|error| {
        IntakeError(format!("cannot read receipt {}: {error}", receipt_path.display()))
    })?;
    let supplied: IntakeReceipt = serde_json::from_slice(&raw_receipt)
        .map_err(|error| IntakeError(format!("receipt JSON is invalid: {error}")))?;
    let expected = build_receipt(manifest_path, dataset_root, &supplied.operation)?;
    if supplied != expected {
        return fail("receipt does not match current manifest bytes or local source files");
    }
    Ok(expected)
}

/// Writes a receipt atomically using a sibling temporary file and rename.
pub fn write_receipt(path: &Path, receipt: &IntakeReceipt) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .map_err(|error| IntakeError(format!("cannot create receipt directory: {error}")))?;
    let tmp = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec_pretty(receipt)
        .map_err(|error| IntakeError(format!("cannot serialize receipt: {error}")))?;
    fs::write(&tmp, bytes)
        .map_err(|error| IntakeError(format!("cannot write temporary receipt: {error}")))?;
    fs::rename(&tmp, path)
        .map_err(|error| IntakeError(format!("cannot commit receipt file: {error}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_ID: AtomicU64 = AtomicU64::new(1);

    struct Fixture {
        root: PathBuf,
        manifest: PathBuf,
        receipt: PathBuf,
        outside: PathBuf,
    }

    impl Fixture {
        fn new() -> Self {
            let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
            let base = std::env::temp_dir().join(format!(
                "textile-dataset-intake-{}-{id}",
                std::process::id()
            ));
            let root = base.join("dataset");
            fs::create_dir_all(&root).unwrap();
            let file_bytes = b"specimen_id,wavelength_nm,absorbance\nS-001,900,0.42\n";
            fs::write(root.join("spectra.csv"), file_bytes).unwrap();
            let rights_text = b"NIST-compatible terms fixture for unit testing only";
            let manifest = base.join("manifest.json");
            let receipt = base.join("receipt.json");
            let outside = base.join("outside.csv");
            fs::write(&outside, b"outside root").unwrap();

            let manifest_value = DatasetManifest {
                schema_version: MANIFEST_SCHEMA_VERSION.to_owned(),
                dataset_id: "synthetic-test-dataset".to_owned(),
                dataset_version: "fixture-1".to_owned(),
                source: DatasetSource {
                    title: "Synthetic intake fixture".to_owned(),
                    publisher: "Unit test".to_owned(),
                    landing_page: "https://example.invalid/dataset".to_owned(),
                    doi: None,
                    version: "fixture-1".to_owned(),
                    retrieved_at_utc: "2026-11-13T00:00:00Z".to_owned(),
                },
                rights: RightsReview {
                    status: RightsStatus::Reviewed,
                    license_url: "https://example.invalid/license".to_owned(),
                    license_text_sha256: Some(sha256_bytes(rights_text)),
                    reviewed_by: Some("unit-test".to_owned()),
                    reviewed_at_utc: Some("2026-11-13T00:00:00Z".to_owned()),
                    permitted_uses: vec!["benchmark_reproduction".to_owned()],
                    attribution: "Synthetic test fixture; not a real dataset".to_owned(),
                    notes: "Test terms only".to_owned(),
                },
                files: vec![DatasetFile {
                    source_path: "Spectroscopy/Source_A/spectra.csv".to_owned(),
                    local_path: "spectra.csv".to_owned(),
                    sha256: sha256_bytes(file_bytes),
                    size_bytes: file_bytes.len() as u64,
                    role: "nir_absorbance".to_owned(),
                }],
                claim_ceiling: "synthetic_fixture_integrity_only".to_owned(),
            };
            fs::write(&manifest, serde_json::to_vec_pretty(&manifest_value).unwrap()).unwrap();

            Self {
                root,
                manifest,
                receipt,
                outside,
            }
        }

        fn manifest_value(&self) -> DatasetManifest {
            serde_json::from_slice(&fs::read(&self.manifest).unwrap()).unwrap()
        }

        fn write_manifest(&self, value: &DatasetManifest) {
            fs::write(&self.manifest, serde_json::to_vec_pretty(value).unwrap()).unwrap();
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            if let Some(base) = self.root.parent() {
                let _ = fs::remove_dir_all(base);
            }
        }
    }

    #[test]
    fn reviewed_manifest_and_matching_files_produce_deterministic_receipt() {
        let fixture = Fixture::new();
        let first = build_receipt(&fixture.manifest, &fixture.root, "benchmark_reproduction").unwrap();
        let second = build_receipt(&fixture.manifest, &fixture.root, "benchmark_reproduction").unwrap();
        assert_eq!(first, second);
        assert_eq!(first.verified_files.len(), 1);
        assert_eq!(first.verified_files[0].sha256, sha256_file(&fixture.root.join("spectra.csv")).unwrap());
        assert_eq!(first.manifest_sha256, sha256_bytes(&fs::read(&fixture.manifest).unwrap()));
    }

    #[test]
    fn pending_or_rejected_rights_never_produce_a_receipt() {
        let fixture = Fixture::new();
        let mut manifest = fixture.manifest_value();
        manifest.rights.status = RightsStatus::Pending;
        manifest.rights.reviewed_by = None;
        manifest.rights.reviewed_at_utc = None;
        fixture.write_manifest(&manifest);
        assert!(build_receipt(&fixture.manifest, &fixture.root, "benchmark_reproduction").is_err());

        manifest.rights.status = RightsStatus::Rejected;
        manifest.rights.reviewed_by = Some("reviewer".to_owned());
        manifest.rights.reviewed_at_utc = Some("2026-11-13T00:00:00Z".to_owned());
        fixture.write_manifest(&manifest);
        assert!(build_receipt(&fixture.manifest, &fixture.root, "benchmark_reproduction").is_err());
    }

    #[test]
    fn unapproved_operation_is_denied() {
        let fixture = Fixture::new();
        assert!(build_receipt(&fixture.manifest, &fixture.root, "redistribution").is_err());
    }

    #[test]
    fn changed_file_bytes_fail_checksum_verification() {
        let fixture = Fixture::new();
        fs::write(fixture.root.join("spectra.csv"), b"tampered").unwrap();
        assert!(build_receipt(&fixture.manifest, &fixture.root, "benchmark_reproduction").is_err());
    }

    #[test]
    fn changed_file_length_fails_before_hash_acceptance() {
        let fixture = Fixture::new();
        fs::write(fixture.root.join("spectra.csv"), b"changed and longer").unwrap();
        let error = build_receipt(&fixture.manifest, &fixture.root, "benchmark_reproduction").unwrap_err();
        assert!(error.to_string().contains("size mismatch"));
    }

    #[test]
    fn parent_traversal_and_absolute_paths_are_rejected() {
        let fixture = Fixture::new();
        let mut manifest = fixture.manifest_value();
        manifest.files[0].local_path = "../outside.csv".to_owned();
        fixture.write_manifest(&manifest);
        assert!(build_receipt(&fixture.manifest, &fixture.root, "benchmark_reproduction").is_err());

        manifest.files[0].local_path = fixture.outside.display().to_string();
        fixture.write_manifest(&manifest);
        assert!(build_receipt(&fixture.manifest, &fixture.root, "benchmark_reproduction").is_err());
    }

    #[test]
    fn duplicate_source_or_local_paths_are_rejected() {
        let fixture = Fixture::new();
        let mut manifest = fixture.manifest_value();
        let mut duplicate = manifest.files[0].clone();
        duplicate.source_path = "different-source-name.csv".to_owned();
        manifest.files.push(duplicate);
        fixture.write_manifest(&manifest);
        assert!(build_receipt(&fixture.manifest, &fixture.root, "benchmark_reproduction").is_err());
    }

    #[test]
    fn altered_manifest_invalidates_existing_receipt() {
        let fixture = Fixture::new();
        let receipt = build_receipt(&fixture.manifest, &fixture.root, "benchmark_reproduction").unwrap();
        write_receipt(&fixture.receipt, &receipt).unwrap();
        let mut manifest = fixture.manifest_value();
        manifest.claim_ceiling = "unjustified_promotion".to_owned();
        fixture.write_manifest(&manifest);
        assert!(verify_receipt(&fixture.manifest, &fixture.root, &fixture.receipt).is_err());
    }

    #[test]
    fn valid_receipt_round_trip_verifies() {
        let fixture = Fixture::new();
        let receipt = build_receipt(&fixture.manifest, &fixture.root, "benchmark_reproduction").unwrap();
        write_receipt(&fixture.receipt, &receipt).unwrap();
        assert_eq!(
            verify_receipt(&fixture.manifest, &fixture.root, &fixture.receipt).unwrap(),
            receipt
        );
    }

    #[test]
    fn published_schema_and_manifest_template_are_parseable_and_fail_closed() {
        const SCHEMA: &str = include_str!("../manifest.schema.json");
        const TEMPLATE: &str = include_str!("../manifest.template.json");
        let schema: serde_json::Value = serde_json::from_str(SCHEMA).unwrap();
        let template: DatasetManifest = serde_json::from_str(TEMPLATE).unwrap();

        assert_eq!(
            schema["properties"]["schema_version"]["const"],
            MANIFEST_SCHEMA_VERSION
        );
        assert_eq!(template.schema_version, MANIFEST_SCHEMA_VERSION);
        assert_eq!(template.dataset_id, "NIST-NIR-SORT");
        assert_eq!(template.rights.status, RightsStatus::Pending);
        assert!(template.files.is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn symlink_escape_from_dataset_root_is_rejected() {
        use std::os::unix::fs::symlink;
        let fixture = Fixture::new();
        symlink(&fixture.outside, fixture.root.join("escape.csv")).unwrap();
        let mut manifest = fixture.manifest_value();
        manifest.files[0].local_path = "escape.csv".to_owned();
        manifest.files[0].size_bytes = fs::metadata(&fixture.outside).unwrap().len();
        manifest.files[0].sha256 = sha256_file(&fixture.outside).unwrap();
        fixture.write_manifest(&manifest);
        assert!(build_receipt(&fixture.manifest, &fixture.root, "benchmark_reproduction").is_err());
    }
}
