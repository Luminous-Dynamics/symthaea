// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Completion manifest for persistent Forge output bundles.
//!
//! A directory containing some Forge files is not, by itself, a completed result. The manifest is
//! written last by the CLI and content-addresses the exact required files for either a winner or a
//! no-winner outcome. Consumers should require and validate it before treating a bundle as complete.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use symthaea_algorithms::ContentId;
use thiserror::Error;

pub const MANIFEST_FILE: &str = "bundle-manifest.json";
pub const TRACE_FILE: &str = "search-trace.json";
pub const OBSERVATIONS_FILE: &str = "observations.json";
pub const CANDIDATE_FILE: &str = "candidate.rs";
pub const CERTIFICATE_FILE: &str = "certificate.json";
pub const REPORT_FILE: &str = "report.md";

#[derive(Debug, Error)]
pub enum BundleError {
    #[error("Forge bundle path is not a regular file: {0}")]
    MissingFile(String),
    #[error("Forge bundle file name is not one of the canonical names for this outcome")]
    UnexpectedFileSet,
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
}

impl ForgeBundleOutcome {
    fn tag(self) -> &'static [u8] {
        match self {
            Self::Winner => b"winner",
            Self::NoWinner => b"no-winner",
        }
    }

    pub fn required_files(self) -> &'static [&'static str] {
        match self {
            Self::Winner => &[
                TRACE_FILE,
                OBSERVATIONS_FILE,
                CANDIDATE_FILE,
                CERTIFICATE_FILE,
                REPORT_FILE,
            ],
            Self::NoWinner => &[TRACE_FILE, OBSERVATIONS_FILE],
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
    pub id: ContentId,
    pub outcome: ForgeBundleOutcome,
    pub files: Vec<ForgeBundleFile>,
}

impl ForgeBundleManifest {
    /// Observe the already-written result files and construct the terminal completion manifest.
    pub fn observe(root: &Path, outcome: ForgeBundleOutcome) -> Result<Self, BundleError> {
        let mut files = outcome
            .required_files()
            .iter()
            .map(|name| ForgeBundleFile::observe(root, name))
            .collect::<Result<Vec<_>, _>>()?;
        files.sort_by(|a, b| a.name.cmp(&b.name));
        let id = derive_manifest_id(outcome, &files);
        Ok(Self { id, outcome, files })
    }

    pub fn validate(&self) -> Result<(), BundleError> {
        let mut expected_names: Vec<&str> = self.outcome.required_files().to_vec();
        expected_names.sort_unstable();
        let observed_names: Vec<&str> = self.files.iter().map(|file| file.name.as_str()).collect();
        if observed_names != expected_names {
            return Err(BundleError::UnexpectedFileSet);
        }
        if derive_manifest_id(self.outcome, &self.files) != self.id {
            return Err(BundleError::ManifestIdentityMismatch);
        }
        Ok(())
    }

    /// Re-read every required file and prove the persisted bundle still matches this manifest.
    pub fn validate_at(&self, root: &Path) -> Result<(), BundleError> {
        self.validate()?;
        let observed = Self::observe(root, self.outcome)?;
        if observed == *self {
            Ok(())
        } else {
            let changed = self
                .files
                .iter()
                .zip(observed.files.iter())
                .find_map(|(expected, actual)| (expected != actual).then(|| expected.name.clone()))
                .unwrap_or_else(|| "unknown".to_string());
            Err(BundleError::FileMismatch(changed))
        }
    }

    pub fn to_json_pretty(&self) -> Result<String, BundleError> {
        serde_json::to_string_pretty(self)
            .map_err(|error| BundleError::Serialization(error.to_string()))
    }
}

fn derive_manifest_id(outcome: ForgeBundleOutcome, files: &[ForgeBundleFile]) -> ContentId {
    let mut parts = vec![
        outcome.tag().to_vec(),
        (files.len() as u64).to_be_bytes().to_vec(),
    ];
    for file in files {
        parts.push(file.name.as_bytes().to_vec());
        parts.push(file.byte_len.to_be_bytes().to_vec());
        parts.push(file.content_id.as_str().as_bytes().to_vec());
    }
    ContentId::derive(
        "symthaea.forge-output-manifest.v1",
        parts.iter().map(Vec::as_slice),
    )
}

/// Read and validate a completed Forge bundle from disk.
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

    #[test]
    fn no_winner_manifest_requires_trace_and_observations() {
        let root = temp_dir("no-winner");
        fs::write(root.join(TRACE_FILE), b"[]").unwrap();
        fs::write(root.join(OBSERVATIONS_FILE), b"[]").unwrap();
        let manifest = ForgeBundleManifest::observe(&root, ForgeBundleOutcome::NoWinner).unwrap();
        assert_eq!(manifest.files.len(), 2);
        assert!(manifest.validate_at(&root).is_ok());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn winner_manifest_binds_all_required_files() {
        let root = temp_dir("winner");
        for (name, bytes) in [
            (TRACE_FILE, b"[]".as_slice()),
            (OBSERVATIONS_FILE, b"[]".as_slice()),
            (CANDIDATE_FILE, b"fn f() {}".as_slice()),
            (CERTIFICATE_FILE, b"{}".as_slice()),
            (REPORT_FILE, b"report".as_slice()),
        ] {
            fs::write(root.join(name), bytes).unwrap();
        }
        let manifest = ForgeBundleManifest::observe(&root, ForgeBundleOutcome::Winner).unwrap();
        assert_eq!(manifest.files.len(), 5);
        assert!(manifest.validate_at(&root).is_ok());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn post_manifest_file_change_is_detected() {
        let root = temp_dir("mutation");
        fs::write(root.join(TRACE_FILE), b"before").unwrap();
        fs::write(root.join(OBSERVATIONS_FILE), b"[]").unwrap();
        let manifest = ForgeBundleManifest::observe(&root, ForgeBundleOutcome::NoWinner).unwrap();
        fs::write(root.join(TRACE_FILE), b"after").unwrap();
        assert!(matches!(
            manifest.validate_at(&root),
            Err(BundleError::FileMismatch(_))
        ));
        let _ = fs::remove_dir_all(root);
    }
}
