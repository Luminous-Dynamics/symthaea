// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reviewer-specific, least-privilege external-review packages.

use crate::evidence_digest::canonical_json_sha256;
use crate::external_review_protocol::{
    ExternalReviewerIdentity, FrozenExternalReviewProtocol, ReviewerAccessClass,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const EXTERNAL_REVIEW_PACKAGE_VERSION: &str = "symthaea-muse-external-review-package-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ReviewEvidenceSensitivity {
    PublicMethods,
    BlindedArtifacts,
    EmbargoedAuthorities,
    UnblindedAssignments,
    ParticipantIdentifying,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReviewEvidenceEntry {
    pub evidence_role: String,
    pub relative_path: String,
    pub sha256: String,
    pub sensitivity: ReviewEvidenceSensitivity,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReviewEvidenceIndex {
    pub study_operations_release_sha256: String,
    pub entries: Vec<ReviewEvidenceEntry>,
    pub index_sha256: String,
}

#[derive(Serialize)]
struct ReviewEvidenceIndexCommitment<'a> {
    study_operations_release_sha256: &'a str,
    entries: &'a [ReviewEvidenceEntry],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalReviewPackage {
    pub package_version: String,
    pub protocol_sha256: String,
    pub evidence_index_sha256: String,
    pub reviewer: ExternalReviewerIdentity,
    pub issued_at_utc: String,
    pub instructions_sha256: String,
    pub included_entries: Vec<ReviewEvidenceEntry>,
    pub excluded_sensitivities: Vec<ReviewEvidenceSensitivity>,
    pub package_sha256: String,
}

#[derive(Serialize)]
struct ExternalReviewPackageCommitment<'a> {
    package_version: &'a str,
    protocol_sha256: &'a str,
    evidence_index_sha256: &'a str,
    reviewer: &'a ExternalReviewerIdentity,
    issued_at_utc: &'a str,
    instructions_sha256: &'a str,
    included_entries: &'a [ReviewEvidenceEntry],
    excluded_sensitivities: &'a [ReviewEvidenceSensitivity],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExternalReviewPackageIssue {
    WrongVersion {
        found: String,
    },
    SerializationFailed {
        field: String,
    },
    DigestMismatch {
        field: String,
    },
    InvalidDigest {
        field: String,
    },
    EmptyField {
        field: String,
    },
    UnknownReviewer {
        reviewer_id: String,
    },
    ReviewerIdentityMismatch {
        reviewer_id: String,
    },
    IndexDigestMismatch,
    DuplicateIndexedEvidenceRole {
        evidence_role: String,
    },
    DuplicateIndexedEvidencePath {
        relative_path: String,
    },
    InvalidIndexedEvidence {
        evidence_role: String,
    },
    DuplicateEvidenceRole {
        evidence_role: String,
    },
    DuplicateEvidencePath {
        relative_path: String,
    },
    MissingAllowedEvidence {
        evidence_role: String,
    },
    EvidenceNotInIndex {
        evidence_role: String,
    },
    EvidenceEntryMismatch {
        evidence_role: String,
    },
    ForbiddenSensitivity {
        reviewer_id: String,
        sensitivity: ReviewEvidenceSensitivity,
    },
    ParticipantIdentifyingEvidenceIncluded,
    MissingExcludedSensitivity {
        sensitivity: ReviewEvidenceSensitivity,
    },
    PackageDigestMismatch,
}

pub fn review_evidence_index_commitment(
    index: &ReviewEvidenceIndex,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&ReviewEvidenceIndexCommitment {
        study_operations_release_sha256: &index.study_operations_release_sha256,
        entries: &index.entries,
    })
}

pub fn seal_review_evidence_index(
    index: &mut ReviewEvidenceIndex,
) -> Result<(), serde_json::Error> {
    index.entries.sort_by(|a, b| {
        a.evidence_role
            .cmp(&b.evidence_role)
            .then_with(|| a.relative_path.cmp(&b.relative_path))
    });
    index.index_sha256 = review_evidence_index_commitment(index)?;
    Ok(())
}

pub fn external_review_package_commitment(
    package: &ExternalReviewPackage,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&ExternalReviewPackageCommitment {
        package_version: &package.package_version,
        protocol_sha256: &package.protocol_sha256,
        evidence_index_sha256: &package.evidence_index_sha256,
        reviewer: &package.reviewer,
        issued_at_utc: &package.issued_at_utc,
        instructions_sha256: &package.instructions_sha256,
        included_entries: &package.included_entries,
        excluded_sensitivities: &package.excluded_sensitivities,
    })
}

pub fn build_external_review_package(
    protocol: &FrozenExternalReviewProtocol,
    index: &ReviewEvidenceIndex,
    reviewer_id: &str,
    issued_at_utc: String,
    instructions_sha256: String,
) -> Result<ExternalReviewPackage, Vec<ExternalReviewPackageIssue>> {
    let reviewer = protocol
        .reviewers
        .iter()
        .find(|reviewer| reviewer.reviewer_id == reviewer_id)
        .cloned()
        .ok_or_else(|| {
            vec![ExternalReviewPackageIssue::UnknownReviewer {
                reviewer_id: reviewer_id.into(),
            }]
        })?;
    let excluded_sensitivities = excluded_for_access(reviewer.access_class);
    let included_entries = index
        .entries
        .iter()
        .filter(|entry| !excluded_sensitivities.contains(&entry.sensitivity))
        .cloned()
        .collect::<Vec<_>>();
    let mut package = ExternalReviewPackage {
        package_version: EXTERNAL_REVIEW_PACKAGE_VERSION.into(),
        protocol_sha256: protocol.protocol_sha256.clone(),
        evidence_index_sha256: index.index_sha256.clone(),
        reviewer,
        issued_at_utc,
        instructions_sha256,
        included_entries,
        excluded_sensitivities,
        package_sha256: String::new(),
    };
    package.included_entries.sort_by(|a, b| {
        a.evidence_role
            .cmp(&b.evidence_role)
            .then_with(|| a.relative_path.cmp(&b.relative_path))
    });
    package.excluded_sensitivities.sort();
    package.package_sha256 = external_review_package_commitment(&package).map_err(|_| {
        vec![ExternalReviewPackageIssue::SerializationFailed {
            field: "package".into(),
        }]
    })?;
    let issues = validate_external_review_package(protocol, index, &package);
    if issues.is_empty() {
        Ok(package)
    } else {
        Err(issues)
    }
}

pub fn validate_external_review_package(
    protocol: &FrozenExternalReviewProtocol,
    index: &ReviewEvidenceIndex,
    package: &ExternalReviewPackage,
) -> Vec<ExternalReviewPackageIssue> {
    let mut issues = Vec::new();
    match review_evidence_index_commitment(index) {
        Ok(value) if value == index.index_sha256 => {}
        Ok(_) => issues.push(ExternalReviewPackageIssue::IndexDigestMismatch),
        Err(_) => issues.push(ExternalReviewPackageIssue::SerializationFailed {
            field: "evidence_index".into(),
        }),
    }
    let mut indexed_roles = BTreeSet::new();
    let mut indexed_paths = BTreeSet::new();
    for entry in &index.entries {
        if !indexed_roles.insert(entry.evidence_role.clone()) {
            issues.push(ExternalReviewPackageIssue::DuplicateIndexedEvidenceRole {
                evidence_role: entry.evidence_role.clone(),
            });
        }
        if !indexed_paths.insert(entry.relative_path.clone()) {
            issues.push(ExternalReviewPackageIssue::DuplicateIndexedEvidencePath {
                relative_path: entry.relative_path.clone(),
            });
        }
        if entry.evidence_role.trim().is_empty()
            || entry.relative_path.trim().is_empty()
            || entry.description.trim().is_empty()
            || !is_sha256(&entry.sha256)
        {
            issues.push(ExternalReviewPackageIssue::InvalidIndexedEvidence {
                evidence_role: entry.evidence_role.clone(),
            });
        }
    }
    if package.package_version != EXTERNAL_REVIEW_PACKAGE_VERSION {
        issues.push(ExternalReviewPackageIssue::WrongVersion {
            found: package.package_version.clone(),
        });
    }
    if package.protocol_sha256 != protocol.protocol_sha256 {
        issues.push(ExternalReviewPackageIssue::DigestMismatch {
            field: "protocol_sha256".into(),
        });
    }
    if package.evidence_index_sha256 != index.index_sha256 {
        issues.push(ExternalReviewPackageIssue::DigestMismatch {
            field: "evidence_index_sha256".into(),
        });
    }
    for (field, digest) in [
        ("instructions_sha256", package.instructions_sha256.as_str()),
        ("package_sha256", package.package_sha256.as_str()),
        ("index_sha256", index.index_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            issues.push(ExternalReviewPackageIssue::InvalidDigest {
                field: field.into(),
            });
        }
    }
    if package.issued_at_utc.trim().is_empty() {
        issues.push(ExternalReviewPackageIssue::EmptyField {
            field: "issued_at_utc".into(),
        });
    }
    let Some(expected_reviewer) = protocol
        .reviewers
        .iter()
        .find(|reviewer| reviewer.reviewer_id == package.reviewer.reviewer_id)
    else {
        issues.push(ExternalReviewPackageIssue::UnknownReviewer {
            reviewer_id: package.reviewer.reviewer_id.clone(),
        });
        return issues;
    };
    if expected_reviewer != &package.reviewer {
        issues.push(ExternalReviewPackageIssue::ReviewerIdentityMismatch {
            reviewer_id: package.reviewer.reviewer_id.clone(),
        });
    }

    let expected_exclusions = excluded_for_access(package.reviewer.access_class);
    for sensitivity in &expected_exclusions {
        if !package.excluded_sensitivities.contains(sensitivity) {
            issues.push(ExternalReviewPackageIssue::MissingExcludedSensitivity {
                sensitivity: *sensitivity,
            });
        }
    }
    let included_roles = package
        .included_entries
        .iter()
        .map(|entry| entry.evidence_role.as_str())
        .collect::<BTreeSet<_>>();
    for entry in &index.entries {
        if !expected_exclusions.contains(&entry.sensitivity)
            && !included_roles.contains(entry.evidence_role.as_str())
        {
            issues.push(ExternalReviewPackageIssue::MissingAllowedEvidence {
                evidence_role: entry.evidence_role.clone(),
            });
        }
    }
    let mut roles = BTreeSet::new();
    let mut paths = BTreeSet::new();
    for entry in &package.included_entries {
        if !roles.insert(entry.evidence_role.clone()) {
            issues.push(ExternalReviewPackageIssue::DuplicateEvidenceRole {
                evidence_role: entry.evidence_role.clone(),
            });
        }
        if !paths.insert(entry.relative_path.clone()) {
            issues.push(ExternalReviewPackageIssue::DuplicateEvidencePath {
                relative_path: entry.relative_path.clone(),
            });
        }
        if !is_sha256(&entry.sha256) {
            issues.push(ExternalReviewPackageIssue::InvalidDigest {
                field: format!("entry.{}.sha256", entry.evidence_role),
            });
        }
        let Some(indexed) = index
            .entries
            .iter()
            .find(|indexed| indexed.evidence_role == entry.evidence_role)
        else {
            issues.push(ExternalReviewPackageIssue::EvidenceNotInIndex {
                evidence_role: entry.evidence_role.clone(),
            });
            continue;
        };
        if indexed != entry {
            issues.push(ExternalReviewPackageIssue::EvidenceEntryMismatch {
                evidence_role: entry.evidence_role.clone(),
            });
        }
        if expected_exclusions.contains(&entry.sensitivity) {
            issues.push(ExternalReviewPackageIssue::ForbiddenSensitivity {
                reviewer_id: package.reviewer.reviewer_id.clone(),
                sensitivity: entry.sensitivity,
            });
        }
        if entry.sensitivity == ReviewEvidenceSensitivity::ParticipantIdentifying {
            issues.push(ExternalReviewPackageIssue::ParticipantIdentifyingEvidenceIncluded);
        }
    }
    match external_review_package_commitment(package) {
        Ok(value) if value == package.package_sha256 => {}
        Ok(_) => issues.push(ExternalReviewPackageIssue::PackageDigestMismatch),
        Err(_) => issues.push(ExternalReviewPackageIssue::SerializationFailed {
            field: "package".into(),
        }),
    }
    issues
}

fn excluded_for_access(access: ReviewerAccessClass) -> Vec<ReviewEvidenceSensitivity> {
    match access {
        ReviewerAccessClass::BlindedMethods => vec![
            ReviewEvidenceSensitivity::EmbargoedAuthorities,
            ReviewEvidenceSensitivity::UnblindedAssignments,
            ReviewEvidenceSensitivity::ParticipantIdentifying,
        ],
        ReviewerAccessClass::EmbargoedReproducibility => vec![
            ReviewEvidenceSensitivity::UnblindedAssignments,
            ReviewEvidenceSensitivity::ParticipantIdentifying,
        ],
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blinded_access_excludes_all_private_authorities() {
        let excluded = excluded_for_access(ReviewerAccessClass::BlindedMethods);
        assert!(excluded.contains(&ReviewEvidenceSensitivity::EmbargoedAuthorities));
        assert!(excluded.contains(&ReviewEvidenceSensitivity::UnblindedAssignments));
        assert!(excluded.contains(&ReviewEvidenceSensitivity::ParticipantIdentifying));
    }

    #[test]
    fn no_access_class_can_receive_participant_identifiers() {
        for access in [
            ReviewerAccessClass::BlindedMethods,
            ReviewerAccessClass::EmbargoedReproducibility,
        ] {
            assert!(
                excluded_for_access(access)
                    .contains(&ReviewEvidenceSensitivity::ParticipantIdentifying)
            );
        }
    }
}
