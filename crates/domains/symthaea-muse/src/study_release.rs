// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Root commitment for a reproducible cognition-study release.

use crate::evidence_digest::{canonical_json_sha256, sha256_hex};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Component, Path};

pub const STUDY_RELEASE_PLAN_VERSION: &str = "symthaea-muse-study-release-plan-v1";
pub const STUDY_RELEASE_BUNDLE_VERSION: &str = "symthaea-muse-study-release-bundle-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum StudyReleaseRole {
    Readme,
    FrozenManifest,
    FrozenMethodology,
    ExternalPreregistrationReceipt,
    BlindedSchedule,
    ParticipantSchedule,
    ArtifactProductionPlan,
    ArtifactBundle,
    StructuralEvidence,
    PolicyBudgetEvidence,
    ParticipantEvidence,
    ConfirmatoryAnalysisPlan,
    ConfirmatoryAnalysisReport,
    IndependentAnalysisReport,
    TemporalAnalysisReport,
    SourceArchive,
    NixFlakeLock,
    ToolchainEvidence,
    BlindingCodebook,
    RandomizationKeyReveal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReleaseVisibility {
    PublicAtRegistration,
    PublicBeforeCollection,
    PublicAfterCollection,
    PublicAfterUnblinding,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StudyReleasePlanEntry {
    pub role: StudyReleaseRole,
    pub visibility: ReleaseVisibility,
    pub relative_path: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StudyReleasePlan {
    pub plan_version: String,
    pub source_revision: String,
    pub workspace_tree_sha256: String,
    pub execution_environment_sha256: String,
    pub external_registration_uri: String,
    pub entries: Vec<StudyReleasePlanEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StudyReleaseFileEvidence {
    pub role: StudyReleaseRole,
    pub visibility: ReleaseVisibility,
    pub relative_path: String,
    pub byte_count: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StudyReleaseBundle {
    pub bundle_version: String,
    pub release_plan_sha256: String,
    pub source_revision: String,
    pub workspace_tree_sha256: String,
    pub execution_environment_sha256: String,
    pub external_registration_uri: String,
    pub files: Vec<StudyReleaseFileEvidence>,
    pub bundle_sha256: String,
}

#[derive(Serialize)]
struct StudyReleaseCommitment<'a> {
    bundle_version: &'a str,
    release_plan_sha256: &'a str,
    source_revision: &'a str,
    workspace_tree_sha256: &'a str,
    execution_environment_sha256: &'a str,
    external_registration_uri: &'a str,
    files: &'a [StudyReleaseFileEvidence],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StudyReleaseIssue {
    WrongPlanVersion { found: String },
    WrongBundleVersion { found: String },
    EmptyField { field: String },
    InvalidDigest { field: String },
    MissingRequiredRole { role: StudyReleaseRole },
    DuplicateRole { role: StudyReleaseRole },
    DuplicatePath { relative_path: String },
    VisibilityViolation { role: StudyReleaseRole },
    UnsafeRelativePath { relative_path: String },
    MissingFile { relative_path: String },
    FileReadFailed { relative_path: String },
    SerializationFailed { field: String },
    BundleMismatch,
}

pub fn study_release_commitment(bundle: &StudyReleaseBundle) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&StudyReleaseCommitment {
        bundle_version: &bundle.bundle_version,
        release_plan_sha256: &bundle.release_plan_sha256,
        source_revision: &bundle.source_revision,
        workspace_tree_sha256: &bundle.workspace_tree_sha256,
        execution_environment_sha256: &bundle.execution_environment_sha256,
        external_registration_uri: &bundle.external_registration_uri,
        files: &bundle.files,
    })
}

pub fn seal_study_release(
    plan: &StudyReleasePlan,
    release_root: &Path,
) -> Result<StudyReleaseBundle, Vec<StudyReleaseIssue>> {
    let mut issues = validate_release_plan(plan);
    let mut seen_paths = BTreeSet::new();
    let mut files = Vec::with_capacity(plan.entries.len());
    for entry in &plan.entries {
        if !seen_paths.insert(entry.relative_path.as_str()) {
            issues.push(StudyReleaseIssue::DuplicatePath {
                relative_path: entry.relative_path.clone(),
            });
            continue;
        }
        let path = Path::new(&entry.relative_path);
        if !safe_relative_path(path) {
            issues.push(StudyReleaseIssue::UnsafeRelativePath {
                relative_path: entry.relative_path.clone(),
            });
            continue;
        }
        let full_path = release_root.join(path);
        if !full_path.is_file() {
            issues.push(StudyReleaseIssue::MissingFile {
                relative_path: entry.relative_path.clone(),
            });
            continue;
        }
        match fs::read(&full_path) {
            Ok(bytes) => files.push(StudyReleaseFileEvidence {
                role: entry.role,
                visibility: entry.visibility,
                relative_path: entry.relative_path.clone(),
                byte_count: bytes.len() as u64,
                sha256: sha256_hex(&bytes),
            }),
            Err(_) => issues.push(StudyReleaseIssue::FileReadFailed {
                relative_path: entry.relative_path.clone(),
            }),
        }
    }
    if !issues.is_empty() {
        return Err(issues);
    }
    files.sort_by(|left, right| {
        left.role
            .cmp(&right.role)
            .then_with(|| left.relative_path.cmp(&right.relative_path))
    });
    let mut bundle = StudyReleaseBundle {
        bundle_version: STUDY_RELEASE_BUNDLE_VERSION.into(),
        release_plan_sha256: canonical_json_sha256(plan).map_err(|_| {
            vec![StudyReleaseIssue::SerializationFailed {
                field: "release_plan".into(),
            }]
        })?,
        source_revision: plan.source_revision.clone(),
        workspace_tree_sha256: plan.workspace_tree_sha256.clone(),
        execution_environment_sha256: plan.execution_environment_sha256.clone(),
        external_registration_uri: plan.external_registration_uri.clone(),
        files,
        bundle_sha256: String::new(),
    };
    bundle.bundle_sha256 = study_release_commitment(&bundle).map_err(|_| {
        vec![StudyReleaseIssue::SerializationFailed {
            field: "release_bundle".into(),
        }]
    })?;
    Ok(bundle)
}

pub fn validate_study_release(
    plan: &StudyReleasePlan,
    bundle: &StudyReleaseBundle,
    release_root: &Path,
) -> Vec<StudyReleaseIssue> {
    if bundle.bundle_version != STUDY_RELEASE_BUNDLE_VERSION {
        return vec![StudyReleaseIssue::WrongBundleVersion {
            found: bundle.bundle_version.clone(),
        }];
    }
    match seal_study_release(plan, release_root) {
        Ok(expected) if expected == *bundle => Vec::new(),
        Ok(_) => vec![StudyReleaseIssue::BundleMismatch],
        Err(issues) => issues,
    }
}

fn validate_release_plan(plan: &StudyReleasePlan) -> Vec<StudyReleaseIssue> {
    let mut issues = Vec::new();
    if plan.plan_version != STUDY_RELEASE_PLAN_VERSION {
        issues.push(StudyReleaseIssue::WrongPlanVersion {
            found: plan.plan_version.clone(),
        });
    }
    for (field, value) in [
        ("source_revision", &plan.source_revision),
        ("external_registration_uri", &plan.external_registration_uri),
    ] {
        if value.trim().is_empty() {
            issues.push(StudyReleaseIssue::EmptyField {
                field: field.into(),
            });
        }
    }
    for (field, digest) in [
        ("workspace_tree_sha256", &plan.workspace_tree_sha256),
        (
            "execution_environment_sha256",
            &plan.execution_environment_sha256,
        ),
    ] {
        if !is_sha256(digest) {
            issues.push(StudyReleaseIssue::InvalidDigest {
                field: field.into(),
            });
        }
    }
    let mut roles = BTreeSet::new();
    for entry in &plan.entries {
        if !roles.insert(entry.role) {
            issues.push(StudyReleaseIssue::DuplicateRole { role: entry.role });
        }
        let visibility_valid = match entry.role {
            StudyReleaseRole::ExternalPreregistrationReceipt => {
                entry.visibility == ReleaseVisibility::PublicAtRegistration
            }
            StudyReleaseRole::FrozenManifest
            | StudyReleaseRole::FrozenMethodology
            | StudyReleaseRole::ConfirmatoryAnalysisPlan => matches!(
                entry.visibility,
                ReleaseVisibility::PublicAtRegistration | ReleaseVisibility::PublicBeforeCollection
            ),
            StudyReleaseRole::ParticipantEvidence
            | StudyReleaseRole::ConfirmatoryAnalysisReport
            | StudyReleaseRole::IndependentAnalysisReport
            | StudyReleaseRole::TemporalAnalysisReport => matches!(
                entry.visibility,
                ReleaseVisibility::PublicAfterCollection | ReleaseVisibility::PublicAfterUnblinding
            ),
            StudyReleaseRole::BlindingCodebook | StudyReleaseRole::RandomizationKeyReveal => {
                entry.visibility == ReleaseVisibility::PublicAfterUnblinding
            }
            _ => true,
        };
        if !visibility_valid {
            issues.push(StudyReleaseIssue::VisibilityViolation { role: entry.role });
        }
    }
    for role in [
        StudyReleaseRole::Readme,
        StudyReleaseRole::FrozenManifest,
        StudyReleaseRole::FrozenMethodology,
        StudyReleaseRole::ExternalPreregistrationReceipt,
        StudyReleaseRole::BlindedSchedule,
        StudyReleaseRole::ParticipantSchedule,
        StudyReleaseRole::ArtifactProductionPlan,
        StudyReleaseRole::ArtifactBundle,
        StudyReleaseRole::StructuralEvidence,
        StudyReleaseRole::PolicyBudgetEvidence,
        StudyReleaseRole::ParticipantEvidence,
        StudyReleaseRole::ConfirmatoryAnalysisPlan,
        StudyReleaseRole::ConfirmatoryAnalysisReport,
        StudyReleaseRole::IndependentAnalysisReport,
        StudyReleaseRole::SourceArchive,
        StudyReleaseRole::NixFlakeLock,
        StudyReleaseRole::ToolchainEvidence,
        StudyReleaseRole::BlindingCodebook,
        StudyReleaseRole::RandomizationKeyReveal,
    ] {
        if !roles.contains(&role) {
            issues.push(StudyReleaseIssue::MissingRequiredRole { role });
        }
    }
    issues
}

fn safe_relative_path(path: &Path) -> bool {
    !path.as_os_str().is_empty()
        && !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_) | Component::CurDir))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_commitment_excludes_only_root_digest() {
        let mut bundle = StudyReleaseBundle {
            bundle_version: STUDY_RELEASE_BUNDLE_VERSION.into(),
            release_plan_sha256: "a".repeat(64),
            source_revision: "revision".into(),
            workspace_tree_sha256: "b".repeat(64),
            execution_environment_sha256: "c".repeat(64),
            external_registration_uri: "https://example.invalid/registration".into(),
            files: Vec::new(),
            bundle_sha256: String::new(),
        };
        let digest = study_release_commitment(&bundle).unwrap();
        bundle.bundle_sha256 = digest.clone();
        assert_eq!(study_release_commitment(&bundle).unwrap(), digest);
    }
}
