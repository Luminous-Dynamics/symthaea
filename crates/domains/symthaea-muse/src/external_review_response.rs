// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sealed external-review responses and reviewer-raised findings.

use crate::evidence_digest::canonical_json_sha256;
use crate::external_review_package::ExternalReviewPackage;
use crate::external_review_protocol::FrozenExternalReviewProtocol;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const EXTERNAL_REVIEW_RESPONSE_VERSION: &str = "symthaea-muse-external-review-response-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ExternalFindingSeverity {
    Informational,
    Minor,
    Major,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExternalReviewVerdict {
    Approve,
    ApproveWithRequiredChanges,
    BlockConfirmatoryStudy,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalQuestionResponse {
    pub question_id: String,
    pub answer: String,
    pub evidence_roles_consulted: Vec<String>,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalReviewFinding {
    pub finding_id: String,
    pub severity: ExternalFindingSeverity,
    pub title: String,
    pub description: String,
    pub affected_evidence_roles: Vec<String>,
    pub required_resolution: String,
    pub blocks_confirmatory_collection: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalReviewResponse {
    pub response_version: String,
    pub protocol_sha256: String,
    pub package_sha256: String,
    pub reviewer_id: String,
    pub completed_at_utc: String,
    pub verdict: ExternalReviewVerdict,
    pub question_responses: Vec<ExternalQuestionResponse>,
    pub findings: Vec<ExternalReviewFinding>,
    pub overall_assessment: String,
    pub external_receipt_uri: String,
    pub external_signature_sha256: String,
    pub response_sha256: String,
}

#[derive(Serialize)]
struct ExternalReviewResponseCommitment<'a> {
    response_version: &'a str,
    protocol_sha256: &'a str,
    package_sha256: &'a str,
    reviewer_id: &'a str,
    completed_at_utc: &'a str,
    verdict: ExternalReviewVerdict,
    question_responses: &'a [ExternalQuestionResponse],
    findings: &'a [ExternalReviewFinding],
    overall_assessment: &'a str,
    external_receipt_uri: &'a str,
    external_signature_sha256: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExternalReviewResponseIssue {
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
    ReviewerMismatch,
    DuplicateQuestionResponse {
        question_id: String,
    },
    MissingRequiredQuestion {
        question_id: String,
    },
    UnknownQuestion {
        question_id: String,
    },
    EmptyEvidenceConsulted {
        question_id: String,
    },
    EvidenceNotInPackage {
        question_id: String,
        evidence_role: String,
    },
    PackageMissingRequiredEvidence {
        question_id: String,
        evidence_role: String,
    },
    RequiredEvidenceNotConsulted {
        question_id: String,
        evidence_role: String,
    },
    DuplicateFindingId {
        finding_id: String,
    },
    EmptyFindingEvidence {
        finding_id: String,
    },
    FindingEvidenceNotInPackage {
        finding_id: String,
        evidence_role: String,
    },
    CriticalFindingNotBlocking {
        finding_id: String,
    },
    BlockingVerdictWithoutFinding,
    ApprovalContradictsBlockingFinding,
    RequiredChangesWithoutFinding,
    ResponseDigestMismatch,
}

pub fn external_review_response_commitment(
    response: &ExternalReviewResponse,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&ExternalReviewResponseCommitment {
        response_version: &response.response_version,
        protocol_sha256: &response.protocol_sha256,
        package_sha256: &response.package_sha256,
        reviewer_id: &response.reviewer_id,
        completed_at_utc: &response.completed_at_utc,
        verdict: response.verdict,
        question_responses: &response.question_responses,
        findings: &response.findings,
        overall_assessment: &response.overall_assessment,
        external_receipt_uri: &response.external_receipt_uri,
        external_signature_sha256: &response.external_signature_sha256,
    })
}

pub fn seal_external_review_response(
    response: &mut ExternalReviewResponse,
) -> Result<(), serde_json::Error> {
    response
        .question_responses
        .sort_by(|a, b| a.question_id.cmp(&b.question_id));
    response
        .findings
        .sort_by(|a, b| a.finding_id.cmp(&b.finding_id));
    for answer in &mut response.question_responses {
        answer.evidence_roles_consulted.sort();
        answer.evidence_roles_consulted.dedup();
        answer.limitations.sort();
    }
    for finding in &mut response.findings {
        finding.affected_evidence_roles.sort();
        finding.affected_evidence_roles.dedup();
    }
    response.response_sha256 = external_review_response_commitment(response)?;
    Ok(())
}

pub fn validate_external_review_response(
    protocol: &FrozenExternalReviewProtocol,
    package: &ExternalReviewPackage,
    response: &ExternalReviewResponse,
) -> Vec<ExternalReviewResponseIssue> {
    let mut issues = Vec::new();
    if response.response_version != EXTERNAL_REVIEW_RESPONSE_VERSION {
        issues.push(ExternalReviewResponseIssue::WrongVersion {
            found: response.response_version.clone(),
        });
    }
    if response.protocol_sha256 != protocol.protocol_sha256 {
        issues.push(ExternalReviewResponseIssue::DigestMismatch {
            field: "protocol_sha256".into(),
        });
    }
    if response.package_sha256 != package.package_sha256 {
        issues.push(ExternalReviewResponseIssue::DigestMismatch {
            field: "package_sha256".into(),
        });
    }
    if response.reviewer_id != package.reviewer.reviewer_id {
        issues.push(ExternalReviewResponseIssue::ReviewerMismatch);
    }
    for (field, digest) in [
        (
            "external_signature_sha256",
            response.external_signature_sha256.as_str(),
        ),
        ("response_sha256", response.response_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            issues.push(ExternalReviewResponseIssue::InvalidDigest {
                field: field.into(),
            });
        }
    }
    for (field, value) in [
        ("reviewer_id", response.reviewer_id.as_str()),
        ("completed_at_utc", response.completed_at_utc.as_str()),
        ("overall_assessment", response.overall_assessment.as_str()),
        (
            "external_receipt_uri",
            response.external_receipt_uri.as_str(),
        ),
    ] {
        if value.trim().is_empty() {
            issues.push(ExternalReviewResponseIssue::EmptyField {
                field: field.into(),
            });
        }
    }

    let package_roles = package
        .included_entries
        .iter()
        .map(|entry| entry.evidence_role.as_str())
        .collect::<BTreeSet<_>>();
    let expected_questions = protocol
        .questions
        .iter()
        .filter(|question| question.role == package.reviewer.role)
        .collect::<Vec<_>>();
    let expected_ids = expected_questions
        .iter()
        .map(|question| question.question_id.as_str())
        .collect::<BTreeSet<_>>();
    let mut answered = BTreeSet::new();
    for answer in &response.question_responses {
        if !answered.insert(answer.question_id.clone()) {
            issues.push(ExternalReviewResponseIssue::DuplicateQuestionResponse {
                question_id: answer.question_id.clone(),
            });
        }
        if !expected_ids.contains(answer.question_id.as_str()) {
            issues.push(ExternalReviewResponseIssue::UnknownQuestion {
                question_id: answer.question_id.clone(),
            });
        }
        let expected_question = expected_questions
            .iter()
            .find(|question| question.question_id == answer.question_id)
            .copied();
        if answer.answer.trim().is_empty() {
            issues.push(ExternalReviewResponseIssue::EmptyField {
                field: format!("question.{}.answer", answer.question_id),
            });
        }
        if answer.evidence_roles_consulted.is_empty() {
            issues.push(ExternalReviewResponseIssue::EmptyEvidenceConsulted {
                question_id: answer.question_id.clone(),
            });
        }
        for evidence_role in &answer.evidence_roles_consulted {
            if !package_roles.contains(evidence_role.as_str()) {
                issues.push(ExternalReviewResponseIssue::EvidenceNotInPackage {
                    question_id: answer.question_id.clone(),
                    evidence_role: evidence_role.clone(),
                });
            }
        }
        if let Some(question) = expected_question {
            for required_role in &question.required_evidence_roles {
                if !package_roles.contains(required_role.as_str()) {
                    issues.push(
                        ExternalReviewResponseIssue::PackageMissingRequiredEvidence {
                            question_id: answer.question_id.clone(),
                            evidence_role: required_role.clone(),
                        },
                    );
                }
                if !answer.evidence_roles_consulted.contains(required_role) {
                    issues.push(ExternalReviewResponseIssue::RequiredEvidenceNotConsulted {
                        question_id: answer.question_id.clone(),
                        evidence_role: required_role.clone(),
                    });
                }
            }
        }
    }
    for question in expected_questions {
        if question.blocking_if_unanswered && !answered.contains(&question.question_id) {
            issues.push(ExternalReviewResponseIssue::MissingRequiredQuestion {
                question_id: question.question_id.clone(),
            });
        }
    }

    let mut finding_ids = BTreeSet::new();
    let mut has_blocking = false;
    for finding in &response.findings {
        if !finding_ids.insert(finding.finding_id.clone()) {
            issues.push(ExternalReviewResponseIssue::DuplicateFindingId {
                finding_id: finding.finding_id.clone(),
            });
        }
        for (field, value) in [
            ("finding_id", finding.finding_id.as_str()),
            ("title", finding.title.as_str()),
            ("description", finding.description.as_str()),
            ("required_resolution", finding.required_resolution.as_str()),
        ] {
            if value.trim().is_empty() {
                issues.push(ExternalReviewResponseIssue::EmptyField {
                    field: format!("finding.{}.{field}", finding.finding_id),
                });
            }
        }
        if finding.affected_evidence_roles.is_empty() {
            issues.push(ExternalReviewResponseIssue::EmptyFindingEvidence {
                finding_id: finding.finding_id.clone(),
            });
        }
        for evidence_role in &finding.affected_evidence_roles {
            if !package_roles.contains(evidence_role.as_str()) {
                issues.push(ExternalReviewResponseIssue::FindingEvidenceNotInPackage {
                    finding_id: finding.finding_id.clone(),
                    evidence_role: evidence_role.clone(),
                });
            }
        }
        if finding.severity == ExternalFindingSeverity::Critical
            && !finding.blocks_confirmatory_collection
        {
            issues.push(ExternalReviewResponseIssue::CriticalFindingNotBlocking {
                finding_id: finding.finding_id.clone(),
            });
        }
        has_blocking |= finding.blocks_confirmatory_collection;
    }
    if response.verdict == ExternalReviewVerdict::BlockConfirmatoryStudy && !has_blocking {
        issues.push(ExternalReviewResponseIssue::BlockingVerdictWithoutFinding);
    }
    if response.verdict == ExternalReviewVerdict::Approve && has_blocking {
        issues.push(ExternalReviewResponseIssue::ApprovalContradictsBlockingFinding);
    }
    if response.verdict == ExternalReviewVerdict::ApproveWithRequiredChanges
        && response.findings.is_empty()
    {
        issues.push(ExternalReviewResponseIssue::RequiredChangesWithoutFinding);
    }
    match external_review_response_commitment(response) {
        Ok(value) if value == response.response_sha256 => {}
        Ok(_) => issues.push(ExternalReviewResponseIssue::ResponseDigestMismatch),
        Err(_) => issues.push(ExternalReviewResponseIssue::SerializationFailed {
            field: "response".into(),
        }),
    }
    issues
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn critical_findings_are_the_highest_severity() {
        assert!(ExternalFindingSeverity::Critical > ExternalFindingSeverity::Major);
    }

    #[test]
    fn commitment_excludes_response_digest() {
        let response = ExternalReviewResponse {
            response_version: EXTERNAL_REVIEW_RESPONSE_VERSION.into(),
            protocol_sha256: "a".repeat(64),
            package_sha256: "b".repeat(64),
            reviewer_id: "reviewer-1".into(),
            completed_at_utc: "2026-07-14T00:00:00Z".into(),
            verdict: ExternalReviewVerdict::Approve,
            question_responses: vec![],
            findings: vec![],
            overall_assessment: "reviewed".into(),
            external_receipt_uri: "https://example.invalid/receipt".into(),
            external_signature_sha256: "c".repeat(64),
            response_sha256: "d".repeat(64),
        };
        let digest = external_review_response_commitment(&response).unwrap();
        let mut changed = response;
        changed.response_sha256 = "e".repeat(64);
        assert_eq!(
            digest,
            external_review_response_commitment(&changed).unwrap()
        );
    }
}
