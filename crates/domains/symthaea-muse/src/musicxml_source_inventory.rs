// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pinned source-only inventory for the OSSQ-OMR MuseScore/MusicXML history.
//!
//! This metadata snapshot is deliberately not a corpus byte-verification
//! receipt. Git blob IDs are SHA-1 object IDs (which include Git's blob
//! framing); they are not SHA-256 hashes of file bytes. A controlled checkout
//! must compute SHA-256 from the actual acquired bytes before MUSE-MIR-020 can
//! advance any sample to ByteVerified.

use crate::evidence_digest::canonical_json_sha256;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const OSSQ_SOURCE_SNAPSHOT_SCHEMA_V1: &str = "muse-ossq-omr-source-snapshot-v1";
pub const OSSQ_SOURCE_INVENTORY_CANONICALIZATION_V1: &str =
    "muse-ossq-source-inventory-sorted-score-records-v1";
pub const OSSQ_UPSTREAM_REPOSITORY_V1: &str = "MALerLab/ossq-omr";
pub const OSSQ_UPSTREAM_COMMIT_V1: &str =
    "7a17e45cddc0b7064fc3a179b62caeb57595e993";
pub const OSSQ_UPSTREAM_TREE_V1: &str =
    "81e829b3755fe169bc3005b8b5d1d8767605afbe";
pub const OSSQ_EXPECTED_SCORE_COUNT_V1: usize = 122;
pub const OSSQ_EXPECTED_COMPOSER_COUNT_V1: usize = 47;
pub const OSSQ_EXPECTED_WORK_COUNT_V1: usize = 116;
pub const OSSQ_EXPECTED_SOURCE_ARTIFACT_COUNT_V1: usize = 462;

/// The only stage represented by this snapshot. Byte verification is a later,
/// separate operation over bytes read from a controlled upstream checkout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OssqSourceSnapshotStatusV1 {
    SourceMetadataRecorded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OssqSourceArtifactRoleV1 {
    NativeMusescoreSource,
    NativeMusicxmlSource,
    CleanedMusicxmlDerivative,
    ScannedAlignmentMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OssqGitBlobReferenceV1 {
    pub path: String,
    /// Git object ID: SHA-1 over the framed Git blob object, not file SHA-256.
    pub git_blob_sha1: String,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OssqSourceArtifactV1 {
    pub role: OssqSourceArtifactRoleV1,
    pub path: String,
    pub git_blob_sha1: String,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OssqSourceScoreRecordV1 {
    pub score_id: String,
    pub title: String,
    pub score_relative_path: String,
    pub composer_group_id: String,
    pub composer_path: String,
    pub composer_name: String,
    /// Uses the upstream set ID as the shared work-group identity.
    pub work_group_id: String,
    pub work_title: String,
    pub source_artifacts: Vec<OssqSourceArtifactV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OssqUpstreamSnapshotV1 {
    pub repository: String,
    pub commit_sha1: String,
    pub tree_sha1: String,
    pub tree_entry_count: usize,
    pub tree_truncated: bool,
    pub source_commit_message: String,
    pub source_commit_parent_sha1: String,
    pub manifest_blobs: Vec<OssqGitBlobReferenceV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OssqInventoryScopeV1 {
    pub description: String,
    pub score_subtree_git_blob_count: usize,
    pub score_rows: usize,
    pub composer_rows: usize,
    pub set_work_rows: usize,
    pub source_artifact_count: usize,
    pub source_artifact_counts: BTreeMap<String, usize>,
    pub excluded_from_this_snapshot: Vec<String>,
    pub sha_semantics: String,
    pub raw_artifact_bytes_acquired: bool,
    pub raw_artifact_sha256_verified: bool,
    pub acquisition_note: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OssqCardinalityExpectationsV1 {
    pub unique_score_ids: usize,
    pub unique_composer_ids: usize,
    pub unique_work_set_ids: usize,
    pub required_per_score_roles: Vec<String>,
    pub optional_per_score_roles: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OssqSourceSnapshotV1 {
    pub schema: String,
    pub status: OssqSourceSnapshotStatusV1,
    pub upstream: OssqUpstreamSnapshotV1,
    pub inventory_scope: OssqInventoryScopeV1,
    pub cardinality_expectations: OssqCardinalityExpectationsV1,
    pub scores: Vec<OssqSourceScoreRecordV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OssqSourceInventoryIssueV1 {
    WrongSchema,
    WrongStatus,
    WrongRepository,
    WrongPinnedCommit,
    WrongPinnedTree,
    TruncatedUpstreamTree,
    MissingCommitMessage,
    InvalidGitSha1 { field: String },
    MissingManifestBlob { path: String },
    DuplicateManifestPath { path: String },
    UnsafeManifestPath { path: String },
    InvalidManifestSize { path: String },
    WrongManifestSet,
    BytesClaimedAcquired,
    Sha256ClaimedVerified,
    MissingShaSemantics,
    MissingAcquisitionNote,
    WrongExpectedCardinality { field: String, expected: usize, actual: usize },
    WrongSourceArtifactCount { expected: usize, actual: usize },
    WrongArtifactRoleCount { role: String, expected: usize, actual: usize },
    MissingExcludedScope { value: String },
    EmptyScoreId,
    DuplicateScoreId { score_id: String },
    EmptyScoreTitle { score_id: String },
    EmptyComposerId { score_id: String },
    EmptyComposerName { score_id: String },
    EmptyWorkId { score_id: String },
    EmptyWorkTitle { score_id: String },
    UnsafeScorePath { score_id: String },
    ScorePathComposerMismatch { score_id: String },
    DuplicateComposerPath { composer_path: String },
    DuplicateArtifactPath { path: String },
    UnsafeArtifactPath { path: String },
    InvalidArtifactSha1 { path: String },
    InvalidArtifactSize { path: String },
    WrongArtifactPathForRole { path: String, role: OssqSourceArtifactRoleV1 },
    MissingRequiredArtifactRole { score_id: String, role: OssqSourceArtifactRoleV1 },
    DuplicateArtifactRole { score_id: String, role: OssqSourceArtifactRoleV1 },
    UnknownArtifactCountRole { role: String },
    MissingManifestSha256, // reserved refusal if a caller tries to treat SHA-1 as SHA-256
}

fn is_git_sha1(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_safe_relative_path(value: &str) -> bool {
    if value.is_empty()
        || value.starts_with('/')
        || value.starts_with('\\')
        || value.contains('\\')
        || value.contains('\0')
    {
        return false;
    }
    value.split('/').all(|component| {
        !component.is_empty()
            && component != "."
            && component != ".."
            && !component.chars().any(char::is_control)
    })
}

fn expected_artifact_suffix(role: OssqSourceArtifactRoleV1, score_id: &str) -> String {
    match role {
        OssqSourceArtifactRoleV1::NativeMusescoreSource => format!("sq{score_id}.mscx"),
        OssqSourceArtifactRoleV1::NativeMusicxmlSource => format!("sq{score_id}.musicxml"),
        OssqSourceArtifactRoleV1::CleanedMusicxmlDerivative => {
            format!("sq{score_id}_cleaned.musicxml")
        }
        OssqSourceArtifactRoleV1::ScannedAlignmentMetadata => {
            format!("sq{score_id}_scanned.csv")
        }
    }
}

fn expected_role_key(role: OssqSourceArtifactRoleV1) -> &'static str {
    match role {
        OssqSourceArtifactRoleV1::NativeMusescoreSource => "mscx",
        OssqSourceArtifactRoleV1::NativeMusicxmlSource => "musicxml",
        OssqSourceArtifactRoleV1::CleanedMusicxmlDerivative => "cleaned_musicxml",
        OssqSourceArtifactRoleV1::ScannedAlignmentMetadata => "scanned_alignment_csv",
    }
}

pub fn validate_ossq_source_snapshot_v1(
    snapshot: &OssqSourceSnapshotV1,
) -> Vec<OssqSourceInventoryIssueV1> {
    let mut issues = Vec::new();
    if snapshot.schema != OSSQ_SOURCE_SNAPSHOT_SCHEMA_V1 {
        issues.push(OssqSourceInventoryIssueV1::WrongSchema);
    }
    if snapshot.status != OssqSourceSnapshotStatusV1::SourceMetadataRecorded {
        issues.push(OssqSourceInventoryIssueV1::WrongStatus);
    }
    let upstream = &snapshot.upstream;
    if upstream.repository != OSSQ_UPSTREAM_REPOSITORY_V1 {
        issues.push(OssqSourceInventoryIssueV1::WrongRepository);
    }
    if upstream.commit_sha1 != OSSQ_UPSTREAM_COMMIT_V1 {
        issues.push(OssqSourceInventoryIssueV1::WrongPinnedCommit);
    }
    if upstream.tree_sha1 != OSSQ_UPSTREAM_TREE_V1 {
        issues.push(OssqSourceInventoryIssueV1::WrongPinnedTree);
    }
    if upstream.tree_truncated {
        issues.push(OssqSourceInventoryIssueV1::TruncatedUpstreamTree);
    }
    if upstream.source_commit_message.trim().is_empty() {
        issues.push(OssqSourceInventoryIssueV1::MissingCommitMessage);
    }
    for (field, value) in [
        ("upstream.commit_sha1", upstream.commit_sha1.as_str()),
        ("upstream.tree_sha1", upstream.tree_sha1.as_str()),
        (
            "upstream.source_commit_parent_sha1",
            upstream.source_commit_parent_sha1.as_str(),
        ),
    ] {
        if !is_git_sha1(value) {
            issues.push(OssqSourceInventoryIssueV1::InvalidGitSha1 {
                field: field.into(),
            });
        }
    }

    const REQUIRED_MANIFESTS: [&str; 8] = [
        "README.md",
        "LICENSE.txt",
        "data/scores.tsv",
        "data/composers.tsv",
        "data/sets.tsv",
        "data/scores.yaml",
        "data/composers.yaml",
        "data/sets.yaml",
    ];
    let mut manifest_paths = BTreeSet::new();
    for manifest in &upstream.manifest_blobs {
        if !is_safe_relative_path(&manifest.path) {
            issues.push(OssqSourceInventoryIssueV1::UnsafeManifestPath {
                path: manifest.path.clone(),
            });
        }
        if !manifest_paths.insert(manifest.path.as_str()) {
            issues.push(OssqSourceInventoryIssueV1::DuplicateManifestPath {
                path: manifest.path.clone(),
            });
        }
        if !is_git_sha1(&manifest.git_blob_sha1) {
            issues.push(OssqSourceInventoryIssueV1::InvalidGitSha1 {
                field: format!("manifest:{}", manifest.path),
            });
        }
        if manifest.size_bytes == 0 {
            issues.push(OssqSourceInventoryIssueV1::InvalidManifestSize {
                path: manifest.path.clone(),
            });
        }
    }
    for required in REQUIRED_MANIFESTS {
        if !manifest_paths.contains(required) {
            issues.push(OssqSourceInventoryIssueV1::MissingManifestBlob {
                path: required.into(),
            });
        }
    }
    if manifest_paths.len() != REQUIRED_MANIFESTS.len() {
        issues.push(OssqSourceInventoryIssueV1::WrongManifestSet);
    }

    let scope = &snapshot.inventory_scope;
    if scope.raw_artifact_bytes_acquired {
        issues.push(OssqSourceInventoryIssueV1::BytesClaimedAcquired);
    }
    if scope.raw_artifact_sha256_verified {
        issues.push(OssqSourceInventoryIssueV1::Sha256ClaimedVerified);
    }
    if scope.sha_semantics.trim().is_empty() {
        issues.push(OssqSourceInventoryIssueV1::MissingShaSemantics);
    }
    if scope.acquisition_note.trim().is_empty() {
        issues.push(OssqSourceInventoryIssueV1::MissingAcquisitionNote);
    }
    if scope.score_rows != snapshot.cardinality_expectations.unique_score_ids {
        issues.push(OssqSourceInventoryIssueV1::WrongExpectedCardinality {
            field: "score_rows_vs_unique_score_ids".into(),
            expected: snapshot.cardinality_expectations.unique_score_ids,
            actual: scope.score_rows,
        });
    }
    if scope.composer_rows != snapshot.cardinality_expectations.unique_composer_ids {
        issues.push(OssqSourceInventoryIssueV1::WrongExpectedCardinality {
            field: "composer_rows_vs_unique_composer_ids".into(),
            expected: snapshot.cardinality_expectations.unique_composer_ids,
            actual: scope.composer_rows,
        });
    }
    if scope.set_work_rows != snapshot.cardinality_expectations.unique_work_set_ids {
        issues.push(OssqSourceInventoryIssueV1::WrongExpectedCardinality {
            field: "set_rows_vs_unique_work_set_ids".into(),
            expected: snapshot.cardinality_expectations.unique_work_set_ids,
            actual: scope.set_work_rows,
        });
    }
    if scope.score_rows != OSSQ_EXPECTED_SCORE_COUNT_V1 {
        issues.push(OssqSourceInventoryIssueV1::WrongExpectedCardinality {
            field: "pinned_score_rows".into(),
            expected: OSSQ_EXPECTED_SCORE_COUNT_V1,
            actual: scope.score_rows,
        });
    }
    if scope.composer_rows != OSSQ_EXPECTED_COMPOSER_COUNT_V1 {
        issues.push(OssqSourceInventoryIssueV1::WrongExpectedCardinality {
            field: "pinned_composer_rows".into(),
            expected: OSSQ_EXPECTED_COMPOSER_COUNT_V1,
            actual: scope.composer_rows,
        });
    }
    if scope.set_work_rows != OSSQ_EXPECTED_WORK_COUNT_V1 {
        issues.push(OssqSourceInventoryIssueV1::WrongExpectedCardinality {
            field: "pinned_set_rows".into(),
            expected: OSSQ_EXPECTED_WORK_COUNT_V1,
            actual: scope.set_work_rows,
        });
    }
    if scope.source_artifact_count != snapshot.scores.iter().map(|score| score.source_artifacts.len()).sum() {
        issues.push(OssqSourceInventoryIssueV1::WrongSourceArtifactCount {
            expected: scope.source_artifact_count,
            actual: snapshot.scores.iter().map(|score| score.source_artifacts.len()).sum(),
        });
    }
    for (key, expected) in [
        ("mscx", 122usize),
        ("musicxml", 122usize),
        ("cleaned_musicxml", 122usize),
        ("scanned_alignment_csv", 96usize),
    ] {
        let actual = scope.source_artifact_counts.get(key).copied().unwrap_or(0);
        if actual != expected {
            issues.push(OssqSourceInventoryIssueV1::WrongArtifactRoleCount {
                role: key.into(),
                expected,
                actual,
            });
        }
    }
    for key in scope.source_artifact_counts.keys() {
        if !["mscx", "musicxml", "cleaned_musicxml", "scanned_alignment_csv"].contains(&key.as_str()) {
            issues.push(OssqSourceInventoryIssueV1::UnknownArtifactCountRole {
                role: key.clone(),
            });
        }
    }
    for excluded in [
        "synthetic and scanned score-image files",
        "systemwise/partwise cropped images",
        "derived LMXE/PLMXE/RLMXE",
        "derived kern/eKern",
        "derived ABC/eABC",
        "official benchmark split membership",
        "acoustic audio and performance recordings",
    ] {
        if !scope.excluded_from_this_snapshot.iter().any(|value| value == excluded) {
            issues.push(OssqSourceInventoryIssueV1::MissingExcludedScope {
                value: excluded.into(),
            });
        }
    }

    let mut score_ids = BTreeSet::new();
    let mut composer_ids = BTreeSet::new();
    let mut composer_paths = BTreeSet::new();
    let mut work_ids = BTreeSet::new();
    let mut artifact_paths = BTreeSet::new();
    let mut artifact_counts = BTreeMap::<String, usize>::new();

    for score in &snapshot.scores {
        if score.score_id.trim().is_empty() {
            issues.push(OssqSourceInventoryIssueV1::EmptyScoreId);
        }
        if !score_ids.insert(score.score_id.as_str()) {
            issues.push(OssqSourceInventoryIssueV1::DuplicateScoreId {
                score_id: score.score_id.clone(),
            });
        }
        for (field, value) in [
            ("title", score.title.as_str()),
            ("composer_group_id", score.composer_group_id.as_str()),
            ("composer_name", score.composer_name.as_str()),
            ("work_group_id", score.work_group_id.as_str()),
            ("work_title", score.work_title.as_str()),
        ] {
            if value.trim().is_empty() {
                match field {
                    "title" => issues.push(OssqSourceInventoryIssueV1::EmptyScoreTitle {
                        score_id: score.score_id.clone(),
                    }),
                    "composer_group_id" => issues.push(OssqSourceInventoryIssueV1::EmptyComposerId {
                        score_id: score.score_id.clone(),
                    }),
                    "composer_name" => issues.push(OssqSourceInventoryIssueV1::EmptyComposerName {
                        score_id: score.score_id.clone(),
                    }),
                    "work_group_id" => issues.push(OssqSourceInventoryIssueV1::EmptyWorkId {
                        score_id: score.score_id.clone(),
                    }),
                    "work_title" => issues.push(OssqSourceInventoryIssueV1::EmptyWorkTitle {
                        score_id: score.score_id.clone(),
                    }),
                    _ => {}
                }
            }
        }
        if !is_safe_relative_path(&score.score_relative_path) {
            issues.push(OssqSourceInventoryIssueV1::UnsafeScorePath {
                score_id: score.score_id.clone(),
            });
        }
        if !score.score_relative_path.starts_with("scores/")
            || !score.score_relative_path.starts_with(&format!("scores/{}/", score.composer_path))
        {
            issues.push(OssqSourceInventoryIssueV1::ScorePathComposerMismatch {
                score_id: score.score_id.clone(),
            });
        }
        composer_ids.insert(score.composer_group_id.as_str());
        composer_paths.insert(score.composer_path.as_str());
        work_ids.insert(score.work_group_id.as_str());

        let mut role_counts = BTreeMap::<OssqSourceArtifactRoleV1, usize>::new();
        for artifact in &score.source_artifacts {
            if !is_safe_relative_path(&artifact.path) {
                issues.push(OssqSourceInventoryIssueV1::UnsafeArtifactPath {
                    path: artifact.path.clone(),
                });
            }
            if !artifact_paths.insert(artifact.path.as_str()) {
                issues.push(OssqSourceInventoryIssueV1::DuplicateArtifactPath {
                    path: artifact.path.clone(),
                });
            }
            if !is_git_sha1(&artifact.git_blob_sha1) {
                issues.push(OssqSourceInventoryIssueV1::InvalidArtifactSha1 {
                    path: artifact.path.clone(),
                });
            }
            if artifact.size_bytes == 0 {
                issues.push(OssqSourceInventoryIssueV1::InvalidArtifactSize {
                    path: artifact.path.clone(),
                });
            }
            let expected_name = expected_artifact_suffix(artifact.role, &score.score_id);
            let expected_path = format!("{}/{}", score.score_relative_path, expected_name);
            if artifact.path != expected_path {
                issues.push(OssqSourceInventoryIssueV1::WrongArtifactPathForRole {
                    path: artifact.path.clone(),
                    role: artifact.role,
                });
            }
            *role_counts.entry(artifact.role).or_default() += 1;
            *artifact_counts.entry(expected_role_key(artifact.role).into()).or_default() += 1;
        }
        for required_role in [
            OssqSourceArtifactRoleV1::NativeMusescoreSource,
            OssqSourceArtifactRoleV1::NativeMusicxmlSource,
        ] {
            match role_counts.get(&required_role).copied().unwrap_or(0) {
                0 => issues.push(OssqSourceInventoryIssueV1::MissingRequiredArtifactRole {
                    score_id: score.score_id.clone(),
                    role: required_role,
                }),
                1 => {}
                _ => issues.push(OssqSourceInventoryIssueV1::DuplicateArtifactRole {
                    score_id: score.score_id.clone(),
                    role: required_role,
                }),
            }
        }
        for optional_role in [
            OssqSourceArtifactRoleV1::CleanedMusicxmlDerivative,
            OssqSourceArtifactRoleV1::ScannedAlignmentMetadata,
        ] {
            if role_counts.get(&optional_role).copied().unwrap_or(0) > 1 {
                issues.push(OssqSourceInventoryIssueV1::DuplicateArtifactRole {
                    score_id: score.score_id.clone(),
                    role: optional_role,
                });
            }
        }
    }

    if score_ids.len() != OSSQ_EXPECTED_SCORE_COUNT_V1 {
        issues.push(OssqSourceInventoryIssueV1::WrongExpectedCardinality {
            field: "unique_score_ids".into(),
            expected: OSSQ_EXPECTED_SCORE_COUNT_V1,
            actual: score_ids.len(),
        });
    }
    if composer_ids.len() != OSSQ_EXPECTED_COMPOSER_COUNT_V1 {
        issues.push(OssqSourceInventoryIssueV1::WrongExpectedCardinality {
            field: "unique_composer_ids".into(),
            expected: OSSQ_EXPECTED_COMPOSER_COUNT_V1,
            actual: composer_ids.len(),
        });
    }
    if composer_paths.len() != OSSQ_EXPECTED_COMPOSER_COUNT_V1 {
        issues.push(OssqSourceInventoryIssueV1::WrongExpectedCardinality {
            field: "unique_composer_paths".into(),
            expected: OSSQ_EXPECTED_COMPOSER_COUNT_V1,
            actual: composer_paths.len(),
        });
    }
    if work_ids.len() != OSSQ_EXPECTED_WORK_COUNT_V1 {
        issues.push(OssqSourceInventoryIssueV1::WrongExpectedCardinality {
            field: "unique_work_set_ids".into(),
            expected: OSSQ_EXPECTED_WORK_COUNT_V1,
            actual: work_ids.len(),
        });
    }
    for (key, expected) in [
        ("mscx", 122usize),
        ("musicxml", 122usize),
        ("cleaned_musicxml", 122usize),
        ("scanned_alignment_csv", 96usize),
    ] {
        let actual = artifact_counts.get(key).copied().unwrap_or(0);
        if actual != expected {
            issues.push(OssqSourceInventoryIssueV1::WrongArtifactRoleCount {
                role: key.into(),
                expected,
                actual,
            });
        }
    }
    if artifact_paths.len() != OSSQ_EXPECTED_SOURCE_ARTIFACT_COUNT_V1 {
        issues.push(OssqSourceInventoryIssueV1::WrongSourceArtifactCount {
            expected: OSSQ_EXPECTED_SOURCE_ARTIFACT_COUNT_V1,
            actual: artifact_paths.len(),
        });
    }
    issues
}

/// Parse the checked-in source snapshot. The repository's upstream artifact
/// bytes are not embedded; this JSON contains only commit/tree/manifest and
/// Git blob metadata.
pub fn load_pinned_ossq_source_snapshot_v1(
) -> Result<OssqSourceSnapshotV1, serde_json::Error> {
    serde_json::from_str(include_str!(
        "../../../../docs/research/OSSQ_OMR_SOURCE_SNAPSHOT_V1.json"
    ))
}

/// Canonical commitment to the complete source metadata snapshot. This SHA-256
/// binds the local JSON structure; it does not replace upstream Git blob IDs or
/// verify any upstream file bytes.
pub fn ossq_source_snapshot_commitment_v1(
    snapshot: &OssqSourceSnapshotV1,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(snapshot)
}

pub fn validate_pinned_ossq_source_snapshot_v1(
) -> Result<(OssqSourceSnapshotV1, String), Vec<OssqSourceInventoryIssueV1>> {
    let snapshot = load_pinned_ossq_source_snapshot_v1().map_err(|_| {
        vec![OssqSourceInventoryIssueV1::WrongSchema]
    })?;
    let issues = validate_ossq_source_snapshot_v1(&snapshot);
    if !issues.is_empty() {
        return Err(issues);
    }
    let digest = ossq_source_snapshot_commitment_v1(&snapshot).map_err(|_| {
        vec![OssqSourceInventoryIssueV1::WrongSchema]
    })?;
    Ok((snapshot, digest))
}

/// Checks whether a source inventory is aligned to the registered OSSQ
/// source revision. It deliberately checks revision metadata only.
pub fn ossq_snapshot_matches_corpus_registration_v1(
    snapshot: &OssqSourceSnapshotV1,
    registration: &crate::musicxml_corpus_qualification::CorpusRegistrationV1,
) -> bool {
    registration.corpus_id == "ossq-omr"
        && registration.source_revision_id.as_deref()
            == Some(snapshot.upstream.commit_sha1.as_str())
        && registration.release_id == "ismir-2026-ossq-omr"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_snapshot_is_valid_and_remains_metadata_only() {
        let (snapshot, digest) = validate_pinned_ossq_source_snapshot_v1().unwrap();
        assert_eq!(snapshot.upstream.repository, OSSQ_UPSTREAM_REPOSITORY_V1);
        assert_eq!(snapshot.upstream.commit_sha1, OSSQ_UPSTREAM_COMMIT_V1);
        assert_eq!(snapshot.upstream.tree_sha1, OSSQ_UPSTREAM_TREE_V1);
        assert_eq!(snapshot.scores.len(), 122);
        assert_eq!(snapshot.inventory_scope.composer_rows, 47);
        assert_eq!(snapshot.inventory_scope.set_work_rows, 116);
        assert_eq!(snapshot.inventory_scope.source_artifact_count, 462);
        assert!(!snapshot.inventory_scope.raw_artifact_bytes_acquired);
        assert!(!snapshot.inventory_scope.raw_artifact_sha256_verified);
        assert_eq!(
            snapshot.status,
            OssqSourceSnapshotStatusV1::SourceMetadataRecorded
        );
        assert_eq!(digest.len(), 64);
    }

    #[test]
    fn source_snapshot_records_native_sources_without_claiming_derived_images() {
        let snapshot = load_pinned_ossq_source_snapshot_v1().unwrap();
        assert_eq!(
            snapshot.inventory_scope.source_artifact_counts.get("mscx"),
            Some(&122)
        );
        assert_eq!(
            snapshot.inventory_scope.source_artifact_counts.get("musicxml"),
            Some(&122)
        );
        assert_eq!(
            snapshot.inventory_scope.source_artifact_counts.get("cleaned_musicxml"),
            Some(&122)
        );
        assert_eq!(
            snapshot.inventory_scope.source_artifact_counts.get("scanned_alignment_csv"),
            Some(&96)
        );
        assert!(snapshot.inventory_scope.excluded_from_this_snapshot
            .iter()
            .any(|value| value == "synthetic and scanned score-image files"));
    }

    #[test]
    fn git_sha1_is_not_accepted_as_a_sha256_artifact_digest() {
        let sha1 = "a".repeat(40);
        let sha256 = "a".repeat(64);
        assert!(is_git_sha1(&sha1));
        assert!(!is_git_sha1(&sha256));
        assert_ne!(sha1.len(), sha256.len());
    }

    #[test]
    fn malformed_git_blob_id_is_rejected() {
        let mut snapshot = load_pinned_ossq_source_snapshot_v1().unwrap();
        snapshot.scores[0].source_artifacts[0].git_blob_sha1 = "g".repeat(40);
        assert!(validate_ossq_source_snapshot_v1(&snapshot)
            .iter()
            .any(|issue| matches!(issue, OssqSourceInventoryIssueV1::InvalidArtifactSha1 { .. })));
    }

    #[test]
    fn missing_required_native_source_is_rejected() {
        let mut snapshot = load_pinned_ossq_source_snapshot_v1().unwrap();
        snapshot.scores[0].source_artifacts.retain(|artifact| {
            artifact.role != OssqSourceArtifactRoleV1::NativeMusicxmlSource
        });
        assert!(validate_ossq_source_snapshot_v1(&snapshot)
            .iter()
            .any(|issue| matches!(
                issue,
                OssqSourceInventoryIssueV1::MissingRequiredArtifactRole {
                    role: OssqSourceArtifactRoleV1::NativeMusicxmlSource,
                    ..
                }
            )));
    }

    #[test]
    fn duplicate_score_ids_and_artifact_paths_are_rejected() {
        let mut snapshot = load_pinned_ossq_source_snapshot_v1().unwrap();
        snapshot.scores[1].score_id = snapshot.scores[0].score_id.clone();
        assert!(validate_ossq_source_snapshot_v1(&snapshot)
            .iter()
            .any(|issue| matches!(issue, OssqSourceInventoryIssueV1::DuplicateScoreId { .. })));

        let mut snapshot = load_pinned_ossq_source_snapshot_v1().unwrap();
        let duplicate_path = snapshot.scores[0].source_artifacts[0].path.clone();
        snapshot.scores[1].source_artifacts[0].path = duplicate_path;
        assert!(validate_ossq_source_snapshot_v1(&snapshot)
            .iter()
            .any(|issue| matches!(issue, OssqSourceInventoryIssueV1::DuplicateArtifactPath { .. })));
    }

    #[test]
    fn traversal_paths_and_wrong_snapshot_pins_are_rejected() {
        let mut snapshot = load_pinned_ossq_source_snapshot_v1().unwrap();
        snapshot.scores[0].source_artifacts[0].path =
            "scores/../../outside.mscx".into();
        snapshot.upstream.commit_sha1 = "0".repeat(40);
        let issues = validate_ossq_source_snapshot_v1(&snapshot);
        assert!(issues.iter().any(|issue| matches!(
            issue,
            OssqSourceInventoryIssueV1::UnsafeArtifactPath { .. }
        )));
        assert!(issues.contains(&OssqSourceInventoryIssueV1::WrongPinnedCommit));
    }

    #[test]
    fn metadata_only_snapshot_cannot_be_promoted_by_toggling_flags() {
        let mut snapshot = load_pinned_ossq_source_snapshot_v1().unwrap();
        snapshot.inventory_scope.raw_artifact_bytes_acquired = true;
        snapshot.inventory_scope.raw_artifact_sha256_verified = true;
        let issues = validate_ossq_source_snapshot_v1(&snapshot);
        assert!(issues.contains(&OssqSourceInventoryIssueV1::BytesClaimedAcquired));
        assert!(issues.contains(&OssqSourceInventoryIssueV1::Sha256ClaimedVerified));
    }

    #[test]
    fn snapshot_commitment_changes_when_upstream_source_identity_changes() {
        let mut snapshot = load_pinned_ossq_source_snapshot_v1().unwrap();
        let before = ossq_source_snapshot_commitment_v1(&snapshot).unwrap();
        snapshot.upstream.source_commit_parent_sha1 = "a".repeat(40);
        let after = ossq_source_snapshot_commitment_v1(&snapshot).unwrap();
        assert_ne!(before, after);
    }

    #[test]
    fn snapshot_registration_alignment_uses_exact_upstream_commit() {
        let snapshot = load_pinned_ossq_source_snapshot_v1().unwrap();
        let mut registration =
            crate::musicxml_corpus_qualification::candidate_musicxml_corpus_registry_v1()
                .into_iter()
                .find(|registration| registration.corpus_id == "ossq-omr")
                .unwrap();
        assert!(ossq_snapshot_matches_corpus_registration_v1(
            &snapshot,
            &registration
        ));
        registration.source_revision_id = Some("0".repeat(40));
        assert!(!ossq_snapshot_matches_corpus_registration_v1(
            &snapshot,
            &registration
        ));
    }
}
