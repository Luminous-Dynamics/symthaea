// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Frozen external-review protocol for confirmatory-study readiness.
//!
//! V11 treats external review as evidence, not an informal email exchange. The
//! protocol freezes the required expertise, the questions each reviewer must
//! answer, conflict-of-interest declarations, and the evidence release being
//! reviewed before any confirmatory participant is enrolled.

use crate::evidence_digest::canonical_json_sha256;
use crate::study_operations_release::StudyOperationsReleaseBundle;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const EXTERNAL_REVIEW_PROTOCOL_VERSION: &str = "symthaea-muse-external-review-protocol-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ExternalReviewRole {
    QuantitativeMethods,
    MusicTheoryAndComposition,
    HumanSubjectsAndUx,
    ReproducibilityEngineering,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReviewerAccessClass {
    /// Public methods, blinded artifacts, protocol, and analysis plan only.
    BlindedMethods,
    /// Reproduction authorities and private codebook access under embargo.
    EmbargoedReproducibility,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalReviewerIdentity {
    pub reviewer_id: String,
    pub role: ExternalReviewRole,
    pub access_class: ReviewerAccessClass,
    pub organization: String,
    pub contact_commitment_sha256: String,
    pub conflict_of_interest_declaration: String,
    pub independent_of_authors: bool,
    pub confidentiality_commitment_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalReviewQuestion {
    pub question_id: String,
    pub role: ExternalReviewRole,
    pub prompt: String,
    pub required_evidence_roles: Vec<String>,
    pub blocking_if_unanswered: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrozenExternalReviewProtocol {
    pub protocol_version: String,
    pub study_operations_release_sha256: String,
    pub opened_at_utc: String,
    pub response_deadline_utc: String,
    pub required_roles: Vec<ExternalReviewRole>,
    pub minimum_reviewers_by_role: BTreeMap<ExternalReviewRole, usize>,
    pub questions: Vec<ExternalReviewQuestion>,
    pub reviewers: Vec<ExternalReviewerIdentity>,
    pub preregistration_uri: String,
    pub preregistration_receipt_sha256: String,
    pub protocol_sha256: String,
}

#[derive(Serialize)]
struct ExternalReviewProtocolCommitment<'a> {
    protocol_version: &'a str,
    study_operations_release_sha256: &'a str,
    opened_at_utc: &'a str,
    response_deadline_utc: &'a str,
    required_roles: &'a [ExternalReviewRole],
    minimum_reviewers_by_role: &'a BTreeMap<ExternalReviewRole, usize>,
    questions: &'a [ExternalReviewQuestion],
    reviewers: &'a [ExternalReviewerIdentity],
    preregistration_uri: &'a str,
    preregistration_receipt_sha256: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExternalReviewProtocolIssue {
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
    DuplicateRequiredRole {
        role: ExternalReviewRole,
    },
    MissingRequiredRole {
        role: ExternalReviewRole,
    },
    ZeroReviewerRequirement {
        role: ExternalReviewRole,
    },
    DuplicateReviewerId {
        reviewer_id: String,
    },
    ReviewerNotIndependent {
        reviewer_id: String,
    },
    EmptyConflictDeclaration {
        reviewer_id: String,
    },
    ReviewerRoleNotRequired {
        reviewer_id: String,
        role: ExternalReviewRole,
    },
    InsufficientReviewers {
        role: ExternalReviewRole,
        found: usize,
        required: usize,
    },
    DuplicateQuestionId {
        question_id: String,
    },
    EmptyQuestionEvidence {
        question_id: String,
    },
    MissingBlockingQuestion {
        role: ExternalReviewRole,
    },
    ProtocolDigestMismatch,
}

pub fn external_review_protocol_commitment(
    protocol: &FrozenExternalReviewProtocol,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&ExternalReviewProtocolCommitment {
        protocol_version: &protocol.protocol_version,
        study_operations_release_sha256: &protocol.study_operations_release_sha256,
        opened_at_utc: &protocol.opened_at_utc,
        response_deadline_utc: &protocol.response_deadline_utc,
        required_roles: &protocol.required_roles,
        minimum_reviewers_by_role: &protocol.minimum_reviewers_by_role,
        questions: &protocol.questions,
        reviewers: &protocol.reviewers,
        preregistration_uri: &protocol.preregistration_uri,
        preregistration_receipt_sha256: &protocol.preregistration_receipt_sha256,
    })
}

pub fn seal_external_review_protocol(
    protocol: &mut FrozenExternalReviewProtocol,
) -> Result<(), serde_json::Error> {
    protocol.required_roles.sort();
    protocol
        .questions
        .sort_by(|a, b| a.question_id.cmp(&b.question_id));
    protocol
        .reviewers
        .sort_by(|a, b| a.reviewer_id.cmp(&b.reviewer_id));
    for question in &mut protocol.questions {
        question.required_evidence_roles.sort();
        question.required_evidence_roles.dedup();
    }
    protocol.protocol_sha256 = external_review_protocol_commitment(protocol)?;
    Ok(())
}

pub fn validate_external_review_protocol(
    operations_release: &StudyOperationsReleaseBundle,
    protocol: &FrozenExternalReviewProtocol,
) -> Vec<ExternalReviewProtocolIssue> {
    let mut issues = Vec::new();
    if protocol.protocol_version != EXTERNAL_REVIEW_PROTOCOL_VERSION {
        issues.push(ExternalReviewProtocolIssue::WrongVersion {
            found: protocol.protocol_version.clone(),
        });
    }
    verify_digest(
        operations_release,
        &protocol.study_operations_release_sha256,
        "study_operations_release_sha256",
        &mut issues,
    );
    for (field, digest) in [
        (
            "preregistration_receipt_sha256",
            protocol.preregistration_receipt_sha256.as_str(),
        ),
        ("protocol_sha256", protocol.protocol_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            issues.push(ExternalReviewProtocolIssue::InvalidDigest {
                field: field.into(),
            });
        }
    }
    for (field, value) in [
        ("opened_at_utc", protocol.opened_at_utc.as_str()),
        (
            "response_deadline_utc",
            protocol.response_deadline_utc.as_str(),
        ),
        ("preregistration_uri", protocol.preregistration_uri.as_str()),
    ] {
        if value.trim().is_empty() {
            issues.push(ExternalReviewProtocolIssue::EmptyField {
                field: field.into(),
            });
        }
    }

    let canonical_roles = [
        ExternalReviewRole::QuantitativeMethods,
        ExternalReviewRole::MusicTheoryAndComposition,
        ExternalReviewRole::HumanSubjectsAndUx,
        ExternalReviewRole::ReproducibilityEngineering,
    ];
    let mut required_roles = BTreeSet::new();
    for role in &protocol.required_roles {
        if !required_roles.insert(*role) {
            issues.push(ExternalReviewProtocolIssue::DuplicateRequiredRole { role: *role });
        }
    }
    for role in canonical_roles {
        if !required_roles.contains(&role) {
            issues.push(ExternalReviewProtocolIssue::MissingRequiredRole { role });
        }
        if protocol
            .minimum_reviewers_by_role
            .get(&role)
            .copied()
            .unwrap_or(0)
            == 0
        {
            issues.push(ExternalReviewProtocolIssue::ZeroReviewerRequirement { role });
        }
    }

    let mut reviewer_ids = BTreeSet::new();
    let mut reviewer_counts = BTreeMap::<ExternalReviewRole, usize>::new();
    for reviewer in &protocol.reviewers {
        if !reviewer_ids.insert(reviewer.reviewer_id.clone()) {
            issues.push(ExternalReviewProtocolIssue::DuplicateReviewerId {
                reviewer_id: reviewer.reviewer_id.clone(),
            });
        }
        for (field, value) in [
            ("reviewer_id", reviewer.reviewer_id.as_str()),
            ("organization", reviewer.organization.as_str()),
        ] {
            if value.trim().is_empty() {
                issues.push(ExternalReviewProtocolIssue::EmptyField {
                    field: format!("reviewer.{}.{field}", reviewer.reviewer_id),
                });
            }
        }
        for (field, digest) in [
            (
                "contact_commitment_sha256",
                reviewer.contact_commitment_sha256.as_str(),
            ),
            (
                "confidentiality_commitment_sha256",
                reviewer.confidentiality_commitment_sha256.as_str(),
            ),
        ] {
            if !is_sha256(digest) {
                issues.push(ExternalReviewProtocolIssue::InvalidDigest {
                    field: format!("reviewer.{}.{field}", reviewer.reviewer_id),
                });
            }
        }
        if !reviewer.independent_of_authors {
            issues.push(ExternalReviewProtocolIssue::ReviewerNotIndependent {
                reviewer_id: reviewer.reviewer_id.clone(),
            });
        }
        if reviewer.conflict_of_interest_declaration.trim().is_empty() {
            issues.push(ExternalReviewProtocolIssue::EmptyConflictDeclaration {
                reviewer_id: reviewer.reviewer_id.clone(),
            });
        }
        if !required_roles.contains(&reviewer.role) {
            issues.push(ExternalReviewProtocolIssue::ReviewerRoleNotRequired {
                reviewer_id: reviewer.reviewer_id.clone(),
                role: reviewer.role,
            });
        }
        *reviewer_counts.entry(reviewer.role).or_default() += 1;
    }
    for role in &protocol.required_roles {
        let found = reviewer_counts.get(role).copied().unwrap_or(0);
        let required = protocol
            .minimum_reviewers_by_role
            .get(role)
            .copied()
            .unwrap_or(0);
        if found < required {
            issues.push(ExternalReviewProtocolIssue::InsufficientReviewers {
                role: *role,
                found,
                required,
            });
        }
    }

    let mut question_ids = BTreeSet::new();
    let mut blocking_roles = BTreeSet::new();
    for question in &protocol.questions {
        if !question_ids.insert(question.question_id.clone()) {
            issues.push(ExternalReviewProtocolIssue::DuplicateQuestionId {
                question_id: question.question_id.clone(),
            });
        }
        if question.question_id.trim().is_empty() || question.prompt.trim().is_empty() {
            issues.push(ExternalReviewProtocolIssue::EmptyField {
                field: format!("question.{}", question.question_id),
            });
        }
        if question.required_evidence_roles.is_empty()
            || question
                .required_evidence_roles
                .iter()
                .any(|role| role.trim().is_empty())
        {
            issues.push(ExternalReviewProtocolIssue::EmptyQuestionEvidence {
                question_id: question.question_id.clone(),
            });
        }
        if question.blocking_if_unanswered {
            blocking_roles.insert(question.role);
        }
    }
    for role in &protocol.required_roles {
        if !blocking_roles.contains(role) {
            issues.push(ExternalReviewProtocolIssue::MissingBlockingQuestion { role: *role });
        }
    }

    match external_review_protocol_commitment(protocol) {
        Ok(value) if value == protocol.protocol_sha256 => {}
        Ok(_) => issues.push(ExternalReviewProtocolIssue::ProtocolDigestMismatch),
        Err(_) => issues.push(ExternalReviewProtocolIssue::SerializationFailed {
            field: "protocol".into(),
        }),
    }
    issues
}

fn verify_digest<T: Serialize>(
    value: &T,
    expected: &str,
    field: &str,
    issues: &mut Vec<ExternalReviewProtocolIssue>,
) {
    match canonical_json_sha256(value) {
        Ok(value) if value == expected => {}
        Ok(_) => issues.push(ExternalReviewProtocolIssue::DigestMismatch {
            field: field.into(),
        }),
        Err(_) => issues.push(ExternalReviewProtocolIssue::SerializationFailed {
            field: field.into(),
        }),
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_four_review_roles_are_distinct() {
        let roles = BTreeSet::from([
            ExternalReviewRole::QuantitativeMethods,
            ExternalReviewRole::MusicTheoryAndComposition,
            ExternalReviewRole::HumanSubjectsAndUx,
            ExternalReviewRole::ReproducibilityEngineering,
        ]);
        assert_eq!(roles.len(), 4);
    }

    #[test]
    fn protocol_commitment_ignores_its_own_digest() {
        let protocol = FrozenExternalReviewProtocol {
            protocol_version: EXTERNAL_REVIEW_PROTOCOL_VERSION.into(),
            study_operations_release_sha256: "a".repeat(64),
            opened_at_utc: "2026-07-14T00:00:00Z".into(),
            response_deadline_utc: "2026-08-14T00:00:00Z".into(),
            required_roles: vec![],
            minimum_reviewers_by_role: BTreeMap::new(),
            questions: vec![],
            reviewers: vec![],
            preregistration_uri: "https://example.invalid/review".into(),
            preregistration_receipt_sha256: "b".repeat(64),
            protocol_sha256: "c".repeat(64),
        };
        let first = external_review_protocol_commitment(&protocol).unwrap();
        let mut changed = protocol;
        changed.protocol_sha256 = "d".repeat(64);
        assert_eq!(
            first,
            external_review_protocol_commitment(&changed).unwrap()
        );
    }
}
