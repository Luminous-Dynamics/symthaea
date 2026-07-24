// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Least-privilege handoff package for an independent replication site.
//!
//! A replication site receives the frozen public authority and executable
//! materials needed to reproduce the study. Original participant evidence,
//! unblinded datasets, codebooks, and randomization secrets are forbidden.

use crate::evidence_digest::canonical_json_sha256;
use crate::replication_protocol::{FrozenReplicationProtocol, replication_protocol_commitment};
use crate::replication_site_registry::{
    ReplicationSiteRegistry, ReplicationSiteStatus, replication_site_registry_commitment,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const REPLICATION_PACKAGE_VERSION: &str = "symthaea-muse-replication-package-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ReplicationEvidenceRole {
    SourceFinalRelease,
    SourceCodeSnapshot,
    FrozenReplicationProtocol,
    AnalysisPlan,
    ArtifactGenerationPlan,
    ParticipantRunnerSource,
    EnvironmentLock,
    PublicStudyMaterials,
    SyntheticDryRun,
    OriginalParticipantEvidence,
    OriginalUnblindedDataset,
    OriginalBlindingCodebook,
    OriginalRandomizationKey,
}

impl ReplicationEvidenceRole {
    pub fn is_forbidden(self) -> bool {
        matches!(
            self,
            Self::OriginalParticipantEvidence
                | Self::OriginalUnblindedDataset
                | Self::OriginalBlindingCodebook
                | Self::OriginalRandomizationKey
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplicationPackageEntry {
    pub role: ReplicationEvidenceRole,
    pub sha256: String,
    pub public_uri: String,
    pub media_type: String,
    pub notes: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplicationSitePackage {
    pub package_version: String,
    pub replication_id: String,
    pub site_id: String,
    pub protocol_sha256: String,
    pub site_registry_sha256: String,
    pub entries: Vec<ReplicationPackageEntry>,
    pub issued_by: String,
    pub issued_at_utc: String,
    pub receipt_required: bool,
    pub package_sha256: String,
}

#[derive(Serialize)]
struct PackageCommitment<'a> {
    package_version: &'a str,
    replication_id: &'a str,
    site_id: &'a str,
    protocol_sha256: &'a str,
    site_registry_sha256: &'a str,
    entries: &'a [ReplicationPackageEntry],
    issued_by: &'a str,
    issued_at_utc: &'a str,
    receipt_required: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplicationPackageIssue {
    WrongVersion {
        found: String,
    },
    InvalidProtocol,
    InvalidSiteRegistry,
    AuthorityMismatch,
    UnknownOrInactiveSite {
        site_id: String,
    },
    EmptyField {
        role: Option<ReplicationEvidenceRole>,
        field: String,
    },
    InvalidDigest {
        role: Option<ReplicationEvidenceRole>,
        field: String,
    },
    DuplicateRole {
        role: ReplicationEvidenceRole,
    },
    MissingRequiredRole {
        role: ReplicationEvidenceRole,
    },
    ForbiddenRole {
        role: ReplicationEvidenceRole,
    },
    ReceiptNotRequired,
    SerializationFailed,
    PackageDigestMismatch,
}

pub fn replication_package_commitment(
    package: &ReplicationSitePackage,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&PackageCommitment {
        package_version: &package.package_version,
        replication_id: &package.replication_id,
        site_id: &package.site_id,
        protocol_sha256: &package.protocol_sha256,
        site_registry_sha256: &package.site_registry_sha256,
        entries: &package.entries,
        issued_by: &package.issued_by,
        issued_at_utc: &package.issued_at_utc,
        receipt_required: package.receipt_required,
    })
}

pub fn build_replication_package(
    protocol: &FrozenReplicationProtocol,
    registry: &ReplicationSiteRegistry,
    site_id: String,
    mut entries: Vec<ReplicationPackageEntry>,
    issued_by: String,
    issued_at_utc: String,
) -> Result<ReplicationSitePackage, Vec<ReplicationPackageIssue>> {
    entries.sort_by_key(|entry| entry.role);
    let mut package = ReplicationSitePackage {
        package_version: REPLICATION_PACKAGE_VERSION.into(),
        replication_id: protocol.replication_id.clone(),
        site_id,
        protocol_sha256: protocol.protocol_sha256.clone(),
        site_registry_sha256: registry.registry_sha256.clone(),
        entries,
        issued_by,
        issued_at_utc,
        receipt_required: true,
        package_sha256: String::new(),
    };
    package.package_sha256 = replication_package_commitment(&package)
        .map_err(|_| vec![ReplicationPackageIssue::SerializationFailed])?;
    let issues = validate_replication_package(protocol, registry, &package);
    if issues.is_empty() {
        Ok(package)
    } else {
        Err(issues)
    }
}

pub fn validate_replication_package(
    protocol: &FrozenReplicationProtocol,
    registry: &ReplicationSiteRegistry,
    package: &ReplicationSitePackage,
) -> Vec<ReplicationPackageIssue> {
    let mut issues = Vec::new();
    if package.package_version != REPLICATION_PACKAGE_VERSION {
        issues.push(ReplicationPackageIssue::WrongVersion {
            found: package.package_version.clone(),
        });
    }
    let protocol_digest = match replication_protocol_commitment(protocol) {
        Ok(value) if value == protocol.protocol_sha256 => value,
        _ => {
            issues.push(ReplicationPackageIssue::InvalidProtocol);
            String::new()
        }
    };
    let registry_digest = match replication_site_registry_commitment(registry) {
        Ok(value) if value == registry.registry_sha256 => value,
        _ => {
            issues.push(ReplicationPackageIssue::InvalidSiteRegistry);
            String::new()
        }
    };
    if package.replication_id != protocol.replication_id
        || package.protocol_sha256 != protocol_digest
        || package.site_registry_sha256 != registry_digest
    {
        issues.push(ReplicationPackageIssue::AuthorityMismatch);
    }
    if !registry.sites.iter().any(|site| {
        site.site_id == package.site_id && site.site_status == ReplicationSiteStatus::Registered
    }) {
        issues.push(ReplicationPackageIssue::UnknownOrInactiveSite {
            site_id: package.site_id.clone(),
        });
    }
    for (field, value) in [
        ("replication_id", package.replication_id.as_str()),
        ("site_id", package.site_id.as_str()),
        ("issued_by", package.issued_by.as_str()),
        ("issued_at_utc", package.issued_at_utc.as_str()),
    ] {
        if value.trim().is_empty() {
            issues.push(ReplicationPackageIssue::EmptyField {
                role: None,
                field: field.into(),
            });
        }
    }
    for (field, digest) in [
        ("protocol_sha256", package.protocol_sha256.as_str()),
        (
            "site_registry_sha256",
            package.site_registry_sha256.as_str(),
        ),
        ("package_sha256", package.package_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            issues.push(ReplicationPackageIssue::InvalidDigest {
                role: None,
                field: field.into(),
            });
        }
    }
    if !package.receipt_required {
        issues.push(ReplicationPackageIssue::ReceiptNotRequired);
    }
    validate_entries(package, &mut issues);
    match replication_package_commitment(package) {
        Ok(digest) if digest == package.package_sha256 => {}
        Ok(_) => issues.push(ReplicationPackageIssue::PackageDigestMismatch),
        Err(_) => issues.push(ReplicationPackageIssue::SerializationFailed),
    }
    issues
}

fn validate_entries(package: &ReplicationSitePackage, issues: &mut Vec<ReplicationPackageIssue>) {
    let mut roles = BTreeSet::new();
    for entry in &package.entries {
        if !roles.insert(entry.role) {
            issues.push(ReplicationPackageIssue::DuplicateRole { role: entry.role });
        }
        if entry.role.is_forbidden() {
            issues.push(ReplicationPackageIssue::ForbiddenRole { role: entry.role });
        }
        for (field, value) in [
            ("public_uri", entry.public_uri.as_str()),
            ("media_type", entry.media_type.as_str()),
        ] {
            if value.trim().is_empty() {
                issues.push(ReplicationPackageIssue::EmptyField {
                    role: Some(entry.role),
                    field: field.into(),
                });
            }
        }
        if !is_sha256(&entry.sha256) {
            issues.push(ReplicationPackageIssue::InvalidDigest {
                role: Some(entry.role),
                field: "sha256".into(),
            });
        }
    }
    for role in [
        ReplicationEvidenceRole::SourceFinalRelease,
        ReplicationEvidenceRole::SourceCodeSnapshot,
        ReplicationEvidenceRole::FrozenReplicationProtocol,
        ReplicationEvidenceRole::AnalysisPlan,
        ReplicationEvidenceRole::ArtifactGenerationPlan,
        ReplicationEvidenceRole::ParticipantRunnerSource,
        ReplicationEvidenceRole::EnvironmentLock,
        ReplicationEvidenceRole::PublicStudyMaterials,
        ReplicationEvidenceRole::SyntheticDryRun,
    ] {
        if !roles.contains(&role) {
            issues.push(ReplicationPackageIssue::MissingRequiredRole { role });
        }
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forbidden_original_evidence_is_rejected() {
        let package = ReplicationSitePackage {
            package_version: REPLICATION_PACKAGE_VERSION.into(),
            replication_id: "rep".into(),
            site_id: "site".into(),
            protocol_sha256: "a".repeat(64),
            site_registry_sha256: "b".repeat(64),
            entries: vec![ReplicationPackageEntry {
                role: ReplicationEvidenceRole::OriginalRandomizationKey,
                sha256: "c".repeat(64),
                public_uri: "https://example.invalid/key".into(),
                media_type: "application/octet-stream".into(),
                notes: String::new(),
            }],
            issued_by: "issuer".into(),
            issued_at_utc: "now".into(),
            receipt_required: true,
            package_sha256: String::new(),
        };
        let mut issues = Vec::new();
        validate_entries(&package, &mut issues);
        assert!(issues.iter().any(|issue| matches!(
            issue,
            ReplicationPackageIssue::ForbiddenRole {
                role: ReplicationEvidenceRole::OriginalRandomizationKey
            }
        )));
    }
}
