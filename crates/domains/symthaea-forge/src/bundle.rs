// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Completion manifest for persistent Forge output bundles.
//!
//! A directory containing some Forge files is not, by itself, a completed result. The manifest is
//! written last and content-addresses the exact canonical file set for a winner, no-winner, or
//! aborted search. New v2 manifests require raw proposal evidence for every terminal outcome while
//! legacy v1 manifests remain verifiable as historical evidence. An aborted run may preserve a
//! previously valid survivor, but partial survivor triplets are rejected.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use symthaea_algorithms::ContentId;
use thiserror::Error;

pub const MANIFEST_FILE: &str = "bundle-manifest.json";
pub const TRACE_FILE: &str = "search-trace.json";
pub const OBSERVATIONS_FILE: &str = "observations.json";
pub const RAW_PROPOSALS_FILE: &str = "raw-proposals.json";
pub const ABORT_FILE: &str = "abort.json";
pub const CANDIDATE_FILE: &str = "candidate.rs";
pub const CERTIFICATE_FILE: &str = "certificate.json";
pub const REPORT_FILE: &str = "report.md";
pub const LEGACY_BUNDLE_FORMAT_VERSION: u32 = 1;
pub const CURRENT_BUNDLE_FORMAT_VERSION: u32 = 2;

const LEGACY_WINNER_FILES: &[&str] = &[
    TRACE_FILE,
    OBSERVATIONS_FILE,
    CANDIDATE_FILE,
    CERTIFICATE_FILE,
    REPORT_FILE,
];
const LEGACY_NO_WINNER_FILES: &[&str] = &[TRACE_FILE, OBSERVATIONS_FILE];
const LEGACY_ABORT_BASE_FILES: &[&str] = &[TRACE_FILE, OBSERVATIONS_FILE, ABORT_FILE];

const V2_WINNER_FILES: &[&str] = &[
    TRACE_FILE,
    OBSERVATIONS_FILE,
    RAW_PROPOSALS_FILE,
    CANDIDATE_FILE,
    CERTIFICATE_FILE,
    REPORT_FILE,
];
const V2_NO_WINNER_FILES: &[&str] = &[TRACE_FILE, OBSERVATIONS_FILE, RAW_PROPOSALS_FILE];
const V2_ABORT_BASE_FILES: &[&str] = &[
    TRACE_FILE,
    OBSERVATIONS_FILE,
    RAW_PROPOSALS_FILE,
    ABORT_FILE,
];
const SURVIVOR_FILES: &[&str] = &[CANDIDATE_FILE, CERTIFICATE_FILE, REPORT_FILE];

fn legacy_bundle_format_version() -> u32 {
    LEGACY_BUNDLE_FORMAT_VERSION
}

#[derive(Debug, Error)]
pub enum BundleError {
    #[error("Forge bundle path is not a regular file: {0}")]
    MissingFile(String),
    #[error("Forge bundle file set is not canonical for this outcome and format version")]
    UnexpectedFileSet,
    #[error("unsupported Forge bundle manifest format version: {0}")]
    UnsupportedFormatVersion(u32),
    #[error("Forge bundle file changed after manifest creation: {0}")]
    FileMismatch(String),
    #[error("Forge bundle manifest identity does not match its canonical fields")]
    ManifestIdentityMismatch,
    #[error("I/O error for Forge bundle file `{path}`: {detail}")]
    Io { path: String, detail: String },
    #[error("Forge bundle manifest serialization failed: {0}")]
    Serialization(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ForgeBundleOutcome {
    Winner,
    NoWinner,
    Aborted,
}

impl ForgeBundleOutcome {
    fn tag(self) -> &'static [u8] {
        match self {
            Self::Winner => b"winner",
            Self::NoWinner => b"no-winner",
            Self::Aborted => b"aborted",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForgeBundleFile {
    pub name: String,
    pub byte_len: u64,
    pub content_id: ContentId,
}

impl ForgeBundleFile {
    fn observe(root: &Path, name: &str) -> Result<Self, BundleError> {
        let path = root.join(name);
        if !path.is_file() {
            return Err(BundleError::MissingFile(name.to_string()));
        }
        let bytes = fs::read(&path).map_err(|error| BundleError::Io {
            path: path.display().to_string(),
            detail: error.to_string(),
        })?;
        let byte_len = u64::try_from(bytes.len())
            .map_err(|_| BundleError::FileMismatch(name.into()))?;
        let content_id = ContentId::derive(
            "symthaea.forge-output-file.v1",
            [name.as_bytes(), bytes.as_slice()],
        );
        Ok(Self {
            name: name.to_string(),
            byte_len,
            content_id,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForgeBundleManifest {
    #[serde(default = "legacy_bundle_format_version")]
    pub format_version: u32,
    pub id: ContentId,
    pub outcome: ForgeBundleOutcome,
    pub files: Vec<ForgeBundleFile>,
}

impl ForgeBundleManifest {
    /// Observe a new proposal-aware v2 bundle. Legacy v1 manifests are read-only historical forms.
    pub fn observe(root: &Path, outcome: ForgeBundleOutcome) -> Result<Self, BundleError> {
        Self::observe_version(root, outcome, CURRENT_BUNDLE_FORMAT_VERSION)
    }

    fn observe_version(
        root: &Path,
        outcome: ForgeBundleOutcome,
        format_version: u32,
    ) -> Result<Self, BundleError> {
        let names = canonical_file_names_at(root, outcome, format_version)?;
        let mut files = names
            .iter()
            .map(|name| ForgeBundleFile::observe(root, name))
            .collect::<Result<Vec<_>, _>>()?;
        files.sort_by(|a, b| a.name.cmp(&b.name));
        let id = derive_manifest_id(format_version, outcome, &files)?;
        Ok(Self {
            format_version,
            id,
            outcome,
            files,
        })
    }

    pub fn validate(&self) -> Result<(), BundleError> {
        validate_format_version(self.format_version)?;
        let observed_names: Vec<&str> = self.files.iter().map(|file| file.name.as_str()).collect();
        if !is_canonical_name_set(self.format_version, self.outcome, &observed_names)? {
            return Err(BundleError::UnexpectedFileSet);
        }
        if self.files.windows(2).any(|pair| pair[0].name >= pair[1].name) {
            return Err(BundleError::UnexpectedFileSet);
        }
        if derive_manifest_id(self.format_version, self.outcome, &self.files)? != self.id {
            return Err(BundleError::ManifestIdentityMismatch);
        }
        Ok(())
    }

    /// Re-read every file according to the manifest's own historical format version.
    pub fn validate_at(&self, root: &Path) -> Result<(), BundleError> {
        self.validate()?;
        let observed = Self::observe_version(root, self.outcome, self.format_version)?;
        if observed == *self {
            Ok(())
        } else {
            let changed = self
                .files
                .iter()
                .zip(observed.files.iter())
                .find_map(|(expected, actual)| (expected != actual).then(|| expected.name.clone()))
                .unwrap_or_else(|| "file-set".to_string());
            Err(BundleError::FileMismatch(changed))
        }
    }

    pub fn to_json_pretty(&self) -> Result<String, BundleError> {
        serde_json::to_string_pretty(self)
            .map_err(|error| BundleError::Serialization(error.to_string()))
    }
}

fn validate_format_version(format_version: u32) -> Result<(), BundleError> {
    match format_version {
        LEGACY_BUNDLE_FORMAT_VERSION | CURRENT_BUNDLE_FORMAT_VERSION => Ok(()),
        other => Err(BundleError::UnsupportedFormatVersion(other)),
    }
}

fn base_files(
    format_version: u32,
    outcome: ForgeBundleOutcome,
) -> Result<&'static [&'static str], BundleError> {
    validate_format_version(format_version)?;
    Ok(match (format_version, outcome) {
        (LEGACY_BUNDLE_FORMAT_VERSION, ForgeBundleOutcome::Winner) => LEGACY_WINNER_FILES,
        (LEGACY_BUNDLE_FORMAT_VERSION, ForgeBundleOutcome::NoWinner) => LEGACY_NO_WINNER_FILES,
        (LEGACY_BUNDLE_FORMAT_VERSION, ForgeBundleOutcome::Aborted) => LEGACY_ABORT_BASE_FILES,
        (CURRENT_BUNDLE_FORMAT_VERSION, ForgeBundleOutcome::Winner) => V2_WINNER_FILES,
        (CURRENT_BUNDLE_FORMAT_VERSION, ForgeBundleOutcome::NoWinner) => V2_NO_WINNER_FILES,
        (CURRENT_BUNDLE_FORMAT_VERSION, ForgeBundleOutcome::Aborted) => V2_ABORT_BASE_FILES,
        _ => unreachable!("validated bundle format version must be known"),
    })
}

fn canonical_file_names_at(
    root: &Path,
    outcome: ForgeBundleOutcome,
    format_version: u32,
) -> Result<Vec<&'static str>, BundleError> {
    let base = base_files(format_version, outcome)?;
    if outcome != ForgeBundleOutcome::Aborted {
        return Ok(base.to_vec());
    }

    let survivor_presence: Vec<bool> = SURVIVOR_FILES
        .iter()
        .map(|name| root.join(name).is_file())
        .collect();
    let all_survivor = survivor_presence.iter().all(|present| *present);
    let no_survivor = survivor_presence.iter().all(|present| !*present);
    if !all_survivor && !no_survivor {
        return Err(BundleError::UnexpectedFileSet);
    }
    let mut files = base.to_vec();
    if all_survivor {
        files.extend_from_slice(SURVIVOR_FILES);
    }
    Ok(files)
}

fn is_canonical_name_set(
    format_version: u32,
    outcome: ForgeBundleOutcome,
    names: &[&str],
) -> Result<bool, BundleError> {
    let mut observed = names.to_vec();
    observed.sort_unstable();
    observed.dedup();
    if observed.len() != names.len() {
        return Ok(false);
    }

    let matches = |expected: &[&str]| {
        let mut expected = expected.to_vec();
        expected.sort_unstable();
        observed == expected
    };
    let base = base_files(format_version, outcome)?;
    if outcome != ForgeBundleOutcome::Aborted {
        return Ok(matches(base));
    }
    let mut with_survivor = base.to_vec();
    with_survivor.extend_from_slice(SURVIVOR_FILES);
    Ok(matches(base) || matches(&with_survivor))
}

fn derive_manifest_id(
    format_version: u32,
    outcome: ForgeBundleOutcome,
    files: &[ForgeBundleFile],
) -> Result<ContentId, BundleError> {
    validate_format_version(format_version)?;
    let mut parts = vec![
        outcome.tag().to_vec(),
        (files.len() as u64).to_be_bytes().to_vec(),
    ];
    let domain = match format_version {
        LEGACY_BUNDLE_FORMAT_VERSION => "symthaea.forge-output-manifest.v1",
        CURRENT_BUNDLE_FORMAT_VERSION => {
            parts.insert(0, format_version.to_be_bytes().to_vec());
            "symthaea.forge-output-manifest.v2"
        }
        _ => unreachable!("validated bundle format version must be known"),
    };
    for file in files {
        parts.push(file.name.as_bytes().to_vec());
        parts.push(file.byte_len.to_be_bytes().to_vec());
        parts.push(file.content_id.as_str().as_bytes().to_vec());
    }
    Ok(ContentId::derive(domain, parts.iter().map(Vec::as_slice)))
}

pub fn read_completed_manifest(root: &Path) -> Result<ForgeBundleManifest, BundleError> {
    let path: PathBuf = root.join(MANIFEST_FILE);
    let bytes = fs::read(&path).map_err(|error| BundleError::Io {
        path: path.display().to_string(),
        detail: error.to_string(),
    })?;
    let manifest: ForgeBundleManifest = serde_json::from_slice(&bytes)
        .map_err(|error| BundleError::Serialization(error.to_string()))?;
    manifest.validate_at(root)?;
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "symthaea-forge-bundle-{label}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        root
    }

    fn write_legacy_core(root: &Path) {
        fs::write(root.join(TRACE_FILE), b"[]").unwrap();
        fs::write(root.join(OBSERVATIONS_FILE), b"[]").unwrap();
    }

    fn write_v2_core(root: &Path) {
        write_legacy_core(root);
        fs::write(root.join(RAW_PROPOSALS_FILE), b"{}").unwrap();
    }

    fn write_survivor_triplet(root: &Path) {
        fs::write(root.join(CANDIDATE_FILE), b"fn f() {}").unwrap();
        fs::write(root.join(CERTIFICATE_FILE), b"{}").unwrap();
        fs::write(root.join(REPORT_FILE), b"report").unwrap();
    }

    #[test]
    fn no_winner_v2_manifest_requires_trace_observations_and_raw_proposals() {
        let root = temp_dir("no-winner-v2");
        write_v2_core(&root);
        let manifest = ForgeBundleManifest::observe(&root, ForgeBundleOutcome::NoWinner).unwrap();
        assert_eq!(manifest.format_version, CURRENT_BUNDLE_FORMAT_VERSION);
        assert_eq!(manifest.files.len(), 3);
        assert!(manifest.validate_at(&root).is_ok());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn missing_raw_proposals_prevents_new_v2_manifest() {
        let root = temp_dir("missing-raw");
        write_legacy_core(&root);
        assert!(matches!(
            ForgeBundleManifest::observe(&root, ForgeBundleOutcome::NoWinner),
            Err(BundleError::MissingFile(name)) if name == RAW_PROPOSALS_FILE
        ));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn legacy_manifest_without_version_field_remains_verifiable() {
        let root = temp_dir("legacy");
        write_legacy_core(&root);
        let legacy = ForgeBundleManifest::observe_version(
            &root,
            ForgeBundleOutcome::NoWinner,
            LEGACY_BUNDLE_FORMAT_VERSION,
        )
        .unwrap();
        let mut value = serde_json::to_value(&legacy).unwrap();
        value.as_object_mut().unwrap().remove("format_version");
        fs::write(
            root.join(MANIFEST_FILE),
            serde_json::to_vec_pretty(&value).unwrap(),
        )
        .unwrap();

        let reloaded = read_completed_manifest(&root).unwrap();
        assert_eq!(reloaded.format_version, LEGACY_BUNDLE_FORMAT_VERSION);
        assert_eq!(reloaded.id, legacy.id);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn winner_v2_manifest_binds_all_required_files() {
        let root = temp_dir("winner");
        write_v2_core(&root);
        write_survivor_triplet(&root);
        let manifest = ForgeBundleManifest::observe(&root, ForgeBundleOutcome::Winner).unwrap();
        assert_eq!(manifest.files.len(), 6);
        assert!(manifest.validate_at(&root).is_ok());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn aborted_v2_manifest_without_survivor_is_canonical() {
        let root = temp_dir("aborted");
        write_v2_core(&root);
        fs::write(root.join(ABORT_FILE), b"{}").unwrap();
        let manifest = ForgeBundleManifest::observe(&root, ForgeBundleOutcome::Aborted).unwrap();
        assert_eq!(manifest.files.len(), 4);
        assert!(manifest.validate_at(&root).is_ok());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn aborted_v2_manifest_may_preserve_complete_survivor_triplet() {
        let root = temp_dir("aborted-survivor");
        write_v2_core(&root);
        fs::write(root.join(ABORT_FILE), b"{}").unwrap();
        write_survivor_triplet(&root);
        let manifest = ForgeBundleManifest::observe(&root, ForgeBundleOutcome::Aborted).unwrap();
        assert_eq!(manifest.files.len(), 7);
        assert!(manifest.validate_at(&root).is_ok());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn aborted_v2_manifest_rejects_partial_survivor_triplet() {
        let root = temp_dir("aborted-partial");
        write_v2_core(&root);
        fs::write(root.join(ABORT_FILE), b"{}").unwrap();
        fs::write(root.join(CANDIDATE_FILE), b"fn f() {}").unwrap();
        assert!(matches!(
            ForgeBundleManifest::observe(&root, ForgeBundleOutcome::Aborted),
            Err(BundleError::UnexpectedFileSet)
        ));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn post_manifest_raw_proposal_change_is_detected() {
        let root = temp_dir("mutation");
        write_v2_core(&root);
        let manifest = ForgeBundleManifest::observe(&root, ForgeBundleOutcome::NoWinner).unwrap();
        fs::write(root.join(RAW_PROPOSALS_FILE), b"changed").unwrap();
        assert!(matches!(
            manifest.validate_at(&root),
            Err(BundleError::FileMismatch(_))
        ));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn unknown_manifest_format_version_fails_closed() {
        let root = temp_dir("unknown-version");
        write_v2_core(&root);
        let mut manifest = ForgeBundleManifest::observe(&root, ForgeBundleOutcome::NoWinner).unwrap();
        manifest.format_version = 99;
        assert!(matches!(
            manifest.validate(),
            Err(BundleError::UnsupportedFormatVersion(99))
        ));
        let _ = fs::remove_dir_all(root);
    }
}
