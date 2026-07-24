// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical release artifact inventories for reproducible promotion.

use crate::crypto_digest::{Sha256, Sha256Digest, sha256};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const RELEASE_ARTIFACT_SET_SCHEMA: &str = "symthaea.fabrication.release-artifact-set.v1";
pub const MAX_RELEASE_ARTIFACTS: usize = 4096;
pub const MAX_ARTIFACT_PATH_BYTES: usize = 1024;
pub const MAX_MEDIA_TYPE_BYTES: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReleaseArtifact {
    pub path: String,
    pub media_type: String,
    pub byte_length: u64,
    pub sha256_digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReleaseArtifactSet {
    pub schema_version: String,
    pub source_tree_digest: Sha256Digest,
    pub artifacts: Vec<ReleaseArtifact>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArtifactSetError {
    UnsupportedSchema,
    EmptySet,
    TooManyArtifacts {
        actual: usize,
        maximum: usize,
    },
    UnsafePath(String),
    InvalidMediaType(String),
    DuplicatePath(String),
    NonCanonicalOrder,
    MissingArtifact(String),
    UnexpectedArtifact(String),
    LengthMismatch {
        path: String,
        expected: u64,
        actual: u64,
    },
    DigestMismatch(String),
    Encoding(String),
}

impl ReleaseArtifactSet {
    pub fn new(
        source_tree_digest: Sha256Digest,
        mut artifacts: Vec<ReleaseArtifact>,
    ) -> Result<Self, ArtifactSetError> {
        artifacts.sort_by(|left, right| left.path.cmp(&right.path));
        let set = Self {
            schema_version: RELEASE_ARTIFACT_SET_SCHEMA.into(),
            source_tree_digest,
            artifacts,
        };
        set.validate()?;
        Ok(set)
    }

    pub fn validate(&self) -> Result<(), ArtifactSetError> {
        if self.schema_version != RELEASE_ARTIFACT_SET_SCHEMA {
            return Err(ArtifactSetError::UnsupportedSchema);
        }
        if self.artifacts.is_empty() {
            return Err(ArtifactSetError::EmptySet);
        }
        if self.artifacts.len() > MAX_RELEASE_ARTIFACTS {
            return Err(ArtifactSetError::TooManyArtifacts {
                actual: self.artifacts.len(),
                maximum: MAX_RELEASE_ARTIFACTS,
            });
        }
        let mut paths = BTreeSet::new();
        let mut previous: Option<&str> = None;
        for artifact in &self.artifacts {
            validate_artifact_path(&artifact.path)?;
            validate_media_type(&artifact.media_type)?;
            if !paths.insert(artifact.path.clone()) {
                return Err(ArtifactSetError::DuplicatePath(artifact.path.clone()));
            }
            if previous.is_some_and(|path| path >= artifact.path.as_str()) {
                return Err(ArtifactSetError::NonCanonicalOrder);
            }
            previous = Some(&artifact.path);
        }
        Ok(())
    }
}

pub fn build_release_artifact_set(
    source_tree_digest: Sha256Digest,
    files: &BTreeMap<String, (String, Vec<u8>)>,
) -> Result<ReleaseArtifactSet, ArtifactSetError> {
    let artifacts = files
        .iter()
        .map(|(path, (media_type, bytes))| ReleaseArtifact {
            path: path.clone(),
            media_type: media_type.clone(),
            byte_length: bytes.len() as u64,
            sha256_digest: sha256(bytes),
        })
        .collect();
    ReleaseArtifactSet::new(source_tree_digest, artifacts)
}

pub fn verify_release_artifact_set(
    set: &ReleaseArtifactSet,
    files: &BTreeMap<String, Vec<u8>>,
) -> Result<(), Vec<ArtifactSetError>> {
    let mut errors = Vec::new();
    if let Err(error) = set.validate() {
        errors.push(error);
    }
    let expected_paths: BTreeSet<_> = set
        .artifacts
        .iter()
        .map(|artifact| artifact.path.as_str())
        .collect();
    for artifact in &set.artifacts {
        let Some(bytes) = files.get(&artifact.path) else {
            errors.push(ArtifactSetError::MissingArtifact(artifact.path.clone()));
            continue;
        };
        if bytes.len() as u64 != artifact.byte_length {
            errors.push(ArtifactSetError::LengthMismatch {
                path: artifact.path.clone(),
                expected: artifact.byte_length,
                actual: bytes.len() as u64,
            });
        }
        if sha256(bytes) != artifact.sha256_digest {
            errors.push(ArtifactSetError::DigestMismatch(artifact.path.clone()));
        }
    }
    for path in files.keys() {
        if !expected_paths.contains(path.as_str()) {
            errors.push(ArtifactSetError::UnexpectedArtifact(path.clone()));
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

pub fn digest_release_artifact_set(
    set: &ReleaseArtifactSet,
) -> Result<Sha256Digest, ArtifactSetError> {
    set.validate()?;
    let bytes =
        serde_json::to_vec(set).map_err(|error| ArtifactSetError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.release-artifact-set-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

fn validate_artifact_path(path: &str) -> Result<(), ArtifactSetError> {
    if path.trim().is_empty()
        || path != path.trim()
        || path.len() > MAX_ARTIFACT_PATH_BYTES
        || path.starts_with('/')
        || path.starts_with('\\')
        || path.contains('\\')
        || path
            .split('/')
            .any(|component| component.is_empty() || component == "." || component == "..")
        || path.chars().any(char::is_control)
    {
        return Err(ArtifactSetError::UnsafePath(path.to_string()));
    }
    Ok(())
}

fn validate_media_type(value: &str) -> Result<(), ArtifactSetError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > MAX_MEDIA_TYPE_BYTES
        || !value.contains('/')
        || value.chars().any(char::is_control)
    {
        return Err(ArtifactSetError::InvalidMediaType(value.to_string()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inventory_detects_substitution_and_extras() {
        let source = Sha256Digest([1; 32]);
        let mut build = BTreeMap::new();
        build.insert(
            "bin/gateway".into(),
            ("application/octet-stream".into(), b"one".to_vec()),
        );
        let set = build_release_artifact_set(source, &build).unwrap();
        let mut supplied = BTreeMap::new();
        supplied.insert("bin/gateway".into(), b"two".to_vec());
        supplied.insert("extra".into(), b"x".to_vec());
        let errors = verify_release_artifact_set(&set, &supplied).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| matches!(error, ArtifactSetError::DigestMismatch(_)))
        );
        assert!(
            errors
                .iter()
                .any(|error| matches!(error, ArtifactSetError::UnexpectedArtifact(_)))
        );
    }

    #[test]
    fn traversal_paths_are_rejected() {
        let artifact = ReleaseArtifact {
            path: "../escape".into(),
            media_type: "application/octet-stream".into(),
            byte_length: 1,
            sha256_digest: Sha256Digest([0; 32]),
        };
        assert!(matches!(
            ReleaseArtifactSet::new(Sha256Digest([1; 32]), vec![artifact]),
            Err(ArtifactSetError::UnsafePath(_))
        ));
    }
}
