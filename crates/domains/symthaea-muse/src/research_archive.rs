// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Long-term archival manifest with file-level fixity and recovery evidence.
//!
//! The archive is built from actual files on disk. It rejects secrets and raw
//! personal data, requires multiple independent custodians, and records a
//! completed recovery drill rather than assuming that deposited files remain usable.

use crate::evidence_digest::{canonical_json_sha256, sha256_hex};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Component, Path};

pub const RESEARCH_ARCHIVE_VERSION: &str = "symthaea-muse-research-archive-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ArchiveFileRole {
    SourceArchive,
    EnvironmentLock,
    FinalRelease,
    ReplicationProtocol,
    ReplicationSiteRegistry,
    ReplicationExecutions,
    ReplicationSynthesis,
    AnalysisCode,
    IndependentVerifier,
    Documentation,
    License,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArchiveFilePlan {
    pub role: ArchiveFileRole,
    pub relative_path: String,
    pub media_type: String,
    pub public: bool,
    pub contains_personal_data: bool,
    pub contains_secret_material: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArchivedFile {
    pub role: ArchiveFileRole,
    pub relative_path: String,
    pub media_type: String,
    pub size_bytes: u64,
    pub sha256: String,
    pub public: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArchiveLocationReceipt {
    pub custodian_id: String,
    pub provider: String,
    pub public_uri: String,
    pub deposited_at_utc: String,
    pub object_root_sha256: String,
    pub receipt_sha256: String,
    pub independent_of_primary_authors: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArchiveRecoveryDrill {
    pub drill_id: String,
    pub source_location_uri: String,
    pub restored_root_sha256: String,
    pub verifier_id: String,
    pub independent_verifier: bool,
    pub commands_sha256: String,
    pub completed_at_utc: String,
    pub succeeded: bool,
    pub evidence_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResearchArchivePlan {
    pub stewardship_id: String,
    pub authority_root_sha256: String,
    pub files: Vec<ArchiveFilePlan>,
    pub locations: Vec<ArchiveLocationReceipt>,
    pub recovery_drill: ArchiveRecoveryDrill,
    pub retention_years: u32,
    pub license_identifier: String,
    pub created_at_utc: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResearchArchiveManifest {
    pub archive_version: String,
    pub stewardship_id: String,
    pub authority_root_sha256: String,
    pub files: Vec<ArchivedFile>,
    pub files_root_sha256: String,
    pub locations: Vec<ArchiveLocationReceipt>,
    pub recovery_drill: ArchiveRecoveryDrill,
    pub retention_years: u32,
    pub license_identifier: String,
    pub created_at_utc: String,
    pub archive_sha256: String,
}

#[derive(Serialize)]
struct ArchiveCommitment<'a> {
    archive_version: &'a str,
    stewardship_id: &'a str,
    authority_root_sha256: &'a str,
    files: &'a [ArchivedFile],
    files_root_sha256: &'a str,
    locations: &'a [ArchiveLocationReceipt],
    recovery_drill: &'a ArchiveRecoveryDrill,
    retention_years: u32,
    license_identifier: &'a str,
    created_at_utc: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResearchArchiveIssue {
    WrongVersion { found: String },
    EmptyField { field: String },
    InvalidDigest { field: String },
    UnsafePath { path: String },
    MissingFile { path: String },
    FileReadFailed { path: String },
    DuplicatePath { path: String },
    DuplicateRole { role: ArchiveFileRole },
    MissingRequiredRole { role: ArchiveFileRole },
    SecretMaterialIncluded { path: String },
    PersonalDataIncluded { path: String },
    NonPublicRequiredFile { role: ArchiveFileRole },
    FileMetadataMismatch { path: String },
    TooFewArchiveLocations,
    DuplicateArchiveProvider { provider: String },
    CustodianNotIndependent { custodian_id: String },
    RecoveryDrillFailed,
    RecoveryVerifierNotIndependent,
    RecoveryRootMismatch,
    RetentionTooShort { years: u32 },
    SerializationFailed,
    ArchiveDigestMismatch,
}

pub fn research_archive_commitment(
    manifest: &ResearchArchiveManifest,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&ArchiveCommitment {
        archive_version: &manifest.archive_version,
        stewardship_id: &manifest.stewardship_id,
        authority_root_sha256: &manifest.authority_root_sha256,
        files: &manifest.files,
        files_root_sha256: &manifest.files_root_sha256,
        locations: &manifest.locations,
        recovery_drill: &manifest.recovery_drill,
        retention_years: manifest.retention_years,
        license_identifier: &manifest.license_identifier,
        created_at_utc: &manifest.created_at_utc,
    })
}

pub fn archived_files_root_commitment(files: &[ArchivedFile]) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&files)
}

pub fn seal_research_archive(
    root: &Path,
    plan: &ResearchArchivePlan,
) -> Result<ResearchArchiveManifest, Vec<ResearchArchiveIssue>> {
    let mut issues = Vec::new();
    let mut files = Vec::new();
    for item in &plan.files {
        if item.contains_secret_material {
            issues.push(ResearchArchiveIssue::SecretMaterialIncluded {
                path: item.relative_path.clone(),
            });
        }
        if item.contains_personal_data {
            issues.push(ResearchArchiveIssue::PersonalDataIncluded {
                path: item.relative_path.clone(),
            });
        }
        if !safe_relative_path(&item.relative_path) {
            issues.push(ResearchArchiveIssue::UnsafePath {
                path: item.relative_path.clone(),
            });
            continue;
        }
        let path = root.join(&item.relative_path);
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                issues.push(ResearchArchiveIssue::MissingFile {
                    path: item.relative_path.clone(),
                });
                continue;
            }
            Err(_) => {
                issues.push(ResearchArchiveIssue::FileReadFailed {
                    path: item.relative_path.clone(),
                });
                continue;
            }
        };
        files.push(ArchivedFile {
            role: item.role,
            relative_path: item.relative_path.clone(),
            media_type: item.media_type.clone(),
            size_bytes: bytes.len() as u64,
            sha256: sha256_hex(&bytes),
            public: item.public,
        });
    }
    if !issues.is_empty() {
        return Err(issues);
    }
    files.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    let mut locations = plan.locations.clone();
    locations.sort_by(|left, right| left.provider.cmp(&right.provider));
    let mut manifest = ResearchArchiveManifest {
        archive_version: RESEARCH_ARCHIVE_VERSION.into(),
        stewardship_id: plan.stewardship_id.clone(),
        authority_root_sha256: plan.authority_root_sha256.clone(),
        files_root_sha256: archived_files_root_commitment(&files)
            .map_err(|_| vec![ResearchArchiveIssue::SerializationFailed])?,
        files,
        locations,
        recovery_drill: plan.recovery_drill.clone(),
        retention_years: plan.retention_years,
        license_identifier: plan.license_identifier.clone(),
        created_at_utc: plan.created_at_utc.clone(),
        archive_sha256: String::new(),
    };
    manifest.archive_sha256 = research_archive_commitment(&manifest)
        .map_err(|_| vec![ResearchArchiveIssue::SerializationFailed])?;
    let issues = validate_research_archive(root, plan, &manifest);
    if issues.is_empty() {
        Ok(manifest)
    } else {
        Err(issues)
    }
}

pub fn validate_research_archive(
    root: &Path,
    plan: &ResearchArchivePlan,
    manifest: &ResearchArchiveManifest,
) -> Vec<ResearchArchiveIssue> {
    let mut issues = Vec::new();
    if manifest.archive_version != RESEARCH_ARCHIVE_VERSION {
        issues.push(ResearchArchiveIssue::WrongVersion {
            found: manifest.archive_version.clone(),
        });
    }
    for (field, value) in [
        ("stewardship_id", manifest.stewardship_id.as_str()),
        ("license_identifier", manifest.license_identifier.as_str()),
        ("created_at_utc", manifest.created_at_utc.as_str()),
    ] {
        if value.trim().is_empty() {
            issues.push(ResearchArchiveIssue::EmptyField {
                field: field.into(),
            });
        }
    }
    for (field, digest) in [
        (
            "authority_root_sha256",
            manifest.authority_root_sha256.as_str(),
        ),
        ("files_root_sha256", manifest.files_root_sha256.as_str()),
        ("archive_sha256", manifest.archive_sha256.as_str()),
        (
            "recovery_drill.restored_root_sha256",
            manifest.recovery_drill.restored_root_sha256.as_str(),
        ),
        (
            "recovery_drill.commands_sha256",
            manifest.recovery_drill.commands_sha256.as_str(),
        ),
        (
            "recovery_drill.evidence_sha256",
            manifest.recovery_drill.evidence_sha256.as_str(),
        ),
    ] {
        if !is_sha256(digest) {
            issues.push(ResearchArchiveIssue::InvalidDigest {
                field: field.into(),
            });
        }
    }
    if manifest.stewardship_id != plan.stewardship_id
        || manifest.authority_root_sha256 != plan.authority_root_sha256
    {
        issues.push(ResearchArchiveIssue::FileMetadataMismatch {
            path: "archive authority".into(),
        });
    }
    validate_files(root, plan, manifest, &mut issues);
    validate_locations(manifest, &mut issues);
    if !manifest.recovery_drill.succeeded {
        issues.push(ResearchArchiveIssue::RecoveryDrillFailed);
    }
    if !manifest.recovery_drill.independent_verifier {
        issues.push(ResearchArchiveIssue::RecoveryVerifierNotIndependent);
    }
    if manifest.recovery_drill.restored_root_sha256 != manifest.files_root_sha256 {
        issues.push(ResearchArchiveIssue::RecoveryRootMismatch);
    }
    if manifest.retention_years < 10 {
        issues.push(ResearchArchiveIssue::RetentionTooShort {
            years: manifest.retention_years,
        });
    }
    match research_archive_commitment(manifest) {
        Ok(digest) if digest == manifest.archive_sha256 => {}
        Ok(_) => issues.push(ResearchArchiveIssue::ArchiveDigestMismatch),
        Err(_) => issues.push(ResearchArchiveIssue::SerializationFailed),
    }
    issues
}

fn validate_files(
    root: &Path,
    plan: &ResearchArchivePlan,
    manifest: &ResearchArchiveManifest,
    issues: &mut Vec<ResearchArchiveIssue>,
) {
    let mut paths = BTreeSet::new();
    let mut roles = BTreeSet::new();
    for file in &manifest.files {
        if !paths.insert(file.relative_path.as_str()) {
            issues.push(ResearchArchiveIssue::DuplicatePath {
                path: file.relative_path.clone(),
            });
        }
        if !roles.insert(file.role) {
            issues.push(ResearchArchiveIssue::DuplicateRole { role: file.role });
        }
        if !safe_relative_path(&file.relative_path) {
            issues.push(ResearchArchiveIssue::UnsafePath {
                path: file.relative_path.clone(),
            });
            continue;
        }
        match fs::read(root.join(&file.relative_path)) {
            Ok(bytes)
                if bytes.len() as u64 == file.size_bytes && sha256_hex(&bytes) == file.sha256 => {}
            Ok(_) => issues.push(ResearchArchiveIssue::FileMetadataMismatch {
                path: file.relative_path.clone(),
            }),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                issues.push(ResearchArchiveIssue::MissingFile {
                    path: file.relative_path.clone(),
                });
            }
            Err(_) => issues.push(ResearchArchiveIssue::FileReadFailed {
                path: file.relative_path.clone(),
            }),
        }
        if !is_sha256(&file.sha256) {
            issues.push(ResearchArchiveIssue::InvalidDigest {
                field: format!("file.{}.sha256", file.relative_path),
            });
        }
    }
    match archived_files_root_commitment(&manifest.files) {
        Ok(digest) if digest == manifest.files_root_sha256 => {}
        _ => issues.push(ResearchArchiveIssue::FileMetadataMismatch {
            path: "files_root_sha256".into(),
        }),
    }
    for required in [
        ArchiveFileRole::SourceArchive,
        ArchiveFileRole::EnvironmentLock,
        ArchiveFileRole::FinalRelease,
        ArchiveFileRole::ReplicationProtocol,
        ArchiveFileRole::ReplicationSiteRegistry,
        ArchiveFileRole::ReplicationExecutions,
        ArchiveFileRole::ReplicationSynthesis,
        ArchiveFileRole::AnalysisCode,
        ArchiveFileRole::IndependentVerifier,
        ArchiveFileRole::Documentation,
        ArchiveFileRole::License,
    ] {
        if !roles.contains(&required) {
            issues.push(ResearchArchiveIssue::MissingRequiredRole { role: required });
        }
        if manifest
            .files
            .iter()
            .any(|file| file.role == required && !file.public)
        {
            issues.push(ResearchArchiveIssue::NonPublicRequiredFile { role: required });
        }
    }
    for item in &plan.files {
        if item.contains_secret_material {
            issues.push(ResearchArchiveIssue::SecretMaterialIncluded {
                path: item.relative_path.clone(),
            });
        }
        if item.contains_personal_data {
            issues.push(ResearchArchiveIssue::PersonalDataIncluded {
                path: item.relative_path.clone(),
            });
        }
    }
}

fn validate_locations(manifest: &ResearchArchiveManifest, issues: &mut Vec<ResearchArchiveIssue>) {
    if manifest.locations.len() < 2 {
        issues.push(ResearchArchiveIssue::TooFewArchiveLocations);
    }
    let mut providers = BTreeSet::new();
    for receipt in &manifest.locations {
        if !providers.insert(receipt.provider.as_str()) {
            issues.push(ResearchArchiveIssue::DuplicateArchiveProvider {
                provider: receipt.provider.clone(),
            });
        }
        for (field, value) in [
            ("custodian_id", receipt.custodian_id.as_str()),
            ("provider", receipt.provider.as_str()),
            ("public_uri", receipt.public_uri.as_str()),
            ("deposited_at_utc", receipt.deposited_at_utc.as_str()),
        ] {
            if value.trim().is_empty() {
                issues.push(ResearchArchiveIssue::EmptyField {
                    field: format!("location.{}.{field}", receipt.provider),
                });
            }
        }
        for (field, digest) in [
            ("object_root_sha256", receipt.object_root_sha256.as_str()),
            ("receipt_sha256", receipt.receipt_sha256.as_str()),
        ] {
            if !is_sha256(digest) {
                issues.push(ResearchArchiveIssue::InvalidDigest {
                    field: format!("location.{}.{field}", receipt.provider),
                });
            }
        }
        if receipt.object_root_sha256 != manifest.files_root_sha256 {
            issues.push(ResearchArchiveIssue::FileMetadataMismatch {
                path: format!("location.{}.object_root_sha256", receipt.provider),
            });
        }
        if !receipt.independent_of_primary_authors {
            issues.push(ResearchArchiveIssue::CustodianNotIndependent {
                custodian_id: receipt.custodian_id.clone(),
            });
        }
    }
}

fn safe_relative_path(value: &str) -> bool {
    let path = Path::new(value);
    !value.trim().is_empty()
        && !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn traversal_paths_are_rejected() {
        assert!(!safe_relative_path("../secret"));
        assert!(!safe_relative_path("/absolute"));
        assert!(safe_relative_path("release/manifest.json"));
    }

    #[test]
    fn archive_requires_multiple_providers() {
        let manifest = ResearchArchiveManifest {
            archive_version: RESEARCH_ARCHIVE_VERSION.into(),
            stewardship_id: "id".into(),
            authority_root_sha256: "a".repeat(64),
            files: Vec::new(),
            files_root_sha256: "d".repeat(64),
            locations: vec![ArchiveLocationReceipt {
                custodian_id: "custodian".into(),
                provider: "provider".into(),
                public_uri: "uri".into(),
                deposited_at_utc: "now".into(),
                object_root_sha256: "b".repeat(64),
                receipt_sha256: "c".repeat(64),
                independent_of_primary_authors: true,
            }],
            recovery_drill: ArchiveRecoveryDrill {
                drill_id: "drill".into(),
                source_location_uri: "uri".into(),
                restored_root_sha256: "d".repeat(64),
                verifier_id: "verifier".into(),
                independent_verifier: true,
                commands_sha256: "e".repeat(64),
                completed_at_utc: "now".into(),
                succeeded: true,
                evidence_sha256: "f".repeat(64),
            },
            retention_years: 10,
            license_identifier: "AGPL-3.0-or-later".into(),
            created_at_utc: "now".into(),
            archive_sha256: "d".repeat(64),
        };
        let mut issues = Vec::new();
        validate_locations(&manifest, &mut issues);
        assert!(issues.contains(&ResearchArchiveIssue::TooFewArchiveLocations));
    }
}
