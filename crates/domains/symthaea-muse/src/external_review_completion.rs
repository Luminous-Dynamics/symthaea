// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Consolidated evidence that the frozen external review completed cleanly.

use crate::evidence_digest::canonical_json_sha256;
use crate::external_review_package::{
    ExternalReviewPackage, ReviewEvidenceIndex, validate_external_review_package,
};
use crate::external_review_protocol::{ExternalReviewRole, FrozenExternalReviewProtocol};
use crate::external_review_resolution::{
    ExternalReviewResolutionLedger, FindingDisposition, validate_external_review_resolution_ledger,
};
use crate::external_review_response::{
    ExternalFindingSeverity, ExternalReviewResponse, ExternalReviewVerdict,
    validate_external_review_response,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const EXTERNAL_REVIEW_COMPLETION_VERSION: &str = "symthaea-muse-external-review-completion-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalReviewRoleCompletion {
    pub role: ExternalReviewRole,
    pub required_reviewers: usize,
    pub completed_reviewers: usize,
    pub approval_count: usize,
    pub required_change_count: usize,
    pub blocking_verdict_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalReviewCompletionEvidence {
    pub completion_version: String,
    pub protocol_sha256: String,
    pub evidence_index_sha256: String,
    pub package_sha256s: Vec<String>,
    pub response_sha256s: Vec<String>,
    pub resolution_ledger_sha256: String,
    pub role_completion: Vec<ExternalReviewRoleCompletion>,
    pub finding_count_by_severity: BTreeMap<ExternalFindingSeverity, usize>,
    pub unresolved_finding_count: usize,
    pub blocking_finding_count: usize,
    pub all_packages_valid: bool,
    pub all_responses_valid: bool,
    pub all_required_roles_complete: bool,
    pub all_findings_resolved: bool,
    pub completed_at_utc: String,
    pub completion_sha256: String,
}

#[derive(Serialize)]
struct ExternalReviewCompletionCommitment<'a> {
    completion_version: &'a str,
    protocol_sha256: &'a str,
    evidence_index_sha256: &'a str,
    package_sha256s: &'a [String],
    response_sha256s: &'a [String],
    resolution_ledger_sha256: &'a str,
    role_completion: &'a [ExternalReviewRoleCompletion],
    finding_count_by_severity: &'a BTreeMap<ExternalFindingSeverity, usize>,
    unresolved_finding_count: usize,
    blocking_finding_count: usize,
    all_packages_valid: bool,
    all_responses_valid: bool,
    all_required_roles_complete: bool,
    all_findings_resolved: bool,
    completed_at_utc: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExternalReviewCompletionIssue {
    WrongVersion {
        found: String,
    },
    SerializationFailed {
        field: String,
    },
    InvalidDigest {
        field: String,
    },
    DigestMismatch {
        field: String,
    },
    EmptyField {
        field: String,
    },
    DuplicatePackageReviewer {
        reviewer_id: String,
    },
    DuplicateResponseReviewer {
        reviewer_id: String,
    },
    MissingPackage {
        reviewer_id: String,
    },
    MissingResponse {
        reviewer_id: String,
    },
    PackageValidationFailed {
        reviewer_id: String,
        issue_count: usize,
    },
    ResponseValidationFailed {
        reviewer_id: String,
        issue_count: usize,
    },
    ResolutionValidationFailed {
        issue_count: usize,
    },
    IncompleteRole {
        role: ExternalReviewRole,
    },
    BlockingVerdict {
        reviewer_id: String,
    },
    UnresolvedFindings {
        found: usize,
    },
    CompletionFlagFalse {
        field: String,
    },
    BlockingVerdictSummary {
        found: usize,
    },
    CompletionDigestMismatch,
}

pub fn external_review_completion_commitment(
    completion: &ExternalReviewCompletionEvidence,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&ExternalReviewCompletionCommitment {
        completion_version: &completion.completion_version,
        protocol_sha256: &completion.protocol_sha256,
        evidence_index_sha256: &completion.evidence_index_sha256,
        package_sha256s: &completion.package_sha256s,
        response_sha256s: &completion.response_sha256s,
        resolution_ledger_sha256: &completion.resolution_ledger_sha256,
        role_completion: &completion.role_completion,
        finding_count_by_severity: &completion.finding_count_by_severity,
        unresolved_finding_count: completion.unresolved_finding_count,
        blocking_finding_count: completion.blocking_finding_count,
        all_packages_valid: completion.all_packages_valid,
        all_responses_valid: completion.all_responses_valid,
        all_required_roles_complete: completion.all_required_roles_complete,
        all_findings_resolved: completion.all_findings_resolved,
        completed_at_utc: &completion.completed_at_utc,
    })
}

pub fn build_external_review_completion(
    protocol: &FrozenExternalReviewProtocol,
    index: &ReviewEvidenceIndex,
    packages: &[ExternalReviewPackage],
    responses: &[ExternalReviewResponse],
    resolution: &ExternalReviewResolutionLedger,
    completed_at_utc: String,
) -> Result<ExternalReviewCompletionEvidence, Vec<ExternalReviewCompletionIssue>> {
    let mut issues = Vec::new();
    let mut package_by_reviewer = BTreeMap::new();
    for package in packages {
        if package_by_reviewer
            .insert(package.reviewer.reviewer_id.clone(), package)
            .is_some()
        {
            issues.push(ExternalReviewCompletionIssue::DuplicatePackageReviewer {
                reviewer_id: package.reviewer.reviewer_id.clone(),
            });
        }
    }
    let mut response_by_reviewer = BTreeMap::new();
    for response in responses {
        if response_by_reviewer
            .insert(response.reviewer_id.clone(), response)
            .is_some()
        {
            issues.push(ExternalReviewCompletionIssue::DuplicateResponseReviewer {
                reviewer_id: response.reviewer_id.clone(),
            });
        }
    }

    let mut all_packages_valid = true;
    let mut all_responses_valid = true;
    for reviewer in &protocol.reviewers {
        let Some(package) = package_by_reviewer.get(&reviewer.reviewer_id).copied() else {
            issues.push(ExternalReviewCompletionIssue::MissingPackage {
                reviewer_id: reviewer.reviewer_id.clone(),
            });
            all_packages_valid = false;
            continue;
        };
        let package_issues = validate_external_review_package(protocol, index, package);
        if !package_issues.is_empty() {
            all_packages_valid = false;
            issues.push(ExternalReviewCompletionIssue::PackageValidationFailed {
                reviewer_id: reviewer.reviewer_id.clone(),
                issue_count: package_issues.len(),
            });
        }
        let Some(response) = response_by_reviewer.get(&reviewer.reviewer_id).copied() else {
            issues.push(ExternalReviewCompletionIssue::MissingResponse {
                reviewer_id: reviewer.reviewer_id.clone(),
            });
            all_responses_valid = false;
            continue;
        };
        let response_issues = validate_external_review_response(protocol, package, response);
        if !response_issues.is_empty() {
            all_responses_valid = false;
            issues.push(ExternalReviewCompletionIssue::ResponseValidationFailed {
                reviewer_id: reviewer.reviewer_id.clone(),
                issue_count: response_issues.len(),
            });
        }
        if response.verdict == ExternalReviewVerdict::BlockConfirmatoryStudy {
            issues.push(ExternalReviewCompletionIssue::BlockingVerdict {
                reviewer_id: reviewer.reviewer_id.clone(),
            });
        }
    }

    let resolution_issues =
        validate_external_review_resolution_ledger(protocol, responses, resolution);
    if !resolution_issues.is_empty() {
        issues.push(ExternalReviewCompletionIssue::ResolutionValidationFailed {
            issue_count: resolution_issues.len(),
        });
    }

    let mut finding_count_by_severity = BTreeMap::new();
    let mut blocking_finding_count = 0;
    for response in responses {
        for finding in &response.findings {
            *finding_count_by_severity
                .entry(finding.severity)
                .or_default() += 1;
            if finding.blocks_confirmatory_collection {
                blocking_finding_count += 1;
            }
        }
    }
    let unresolved_finding_count = resolution
        .resolutions
        .iter()
        .filter(|resolution| resolution.disposition == FindingDisposition::Open)
        .count();
    if unresolved_finding_count > 0 {
        issues.push(ExternalReviewCompletionIssue::UnresolvedFindings {
            found: unresolved_finding_count,
        });
    }

    let mut role_completion = Vec::new();
    let mut all_required_roles_complete = true;
    for role in &protocol.required_roles {
        let required = protocol
            .minimum_reviewers_by_role
            .get(role)
            .copied()
            .unwrap_or(0);
        let matching = protocol
            .reviewers
            .iter()
            .filter(|reviewer| reviewer.role == *role)
            .filter_map(|reviewer| response_by_reviewer.get(&reviewer.reviewer_id).copied())
            .collect::<Vec<_>>();
        if matching.len() < required {
            all_required_roles_complete = false;
            issues.push(ExternalReviewCompletionIssue::IncompleteRole { role: *role });
        }
        role_completion.push(ExternalReviewRoleCompletion {
            role: *role,
            required_reviewers: required,
            completed_reviewers: matching.len(),
            approval_count: matching
                .iter()
                .filter(|response| response.verdict == ExternalReviewVerdict::Approve)
                .count(),
            required_change_count: matching
                .iter()
                .filter(|response| {
                    response.verdict == ExternalReviewVerdict::ApproveWithRequiredChanges
                })
                .count(),
            blocking_verdict_count: matching
                .iter()
                .filter(|response| {
                    response.verdict == ExternalReviewVerdict::BlockConfirmatoryStudy
                })
                .count(),
        });
    }
    role_completion.sort_by_key(|status| status.role);

    if !issues.is_empty() {
        return Err(issues);
    }
    let mut completion = ExternalReviewCompletionEvidence {
        completion_version: EXTERNAL_REVIEW_COMPLETION_VERSION.into(),
        protocol_sha256: protocol.protocol_sha256.clone(),
        evidence_index_sha256: index.index_sha256.clone(),
        package_sha256s: packages
            .iter()
            .map(|package| package.package_sha256.clone())
            .collect(),
        response_sha256s: responses
            .iter()
            .map(|response| response.response_sha256.clone())
            .collect(),
        resolution_ledger_sha256: resolution.ledger_sha256.clone(),
        role_completion,
        finding_count_by_severity,
        unresolved_finding_count,
        blocking_finding_count,
        all_packages_valid,
        all_responses_valid,
        all_required_roles_complete,
        all_findings_resolved: resolution_issues.is_empty() && unresolved_finding_count == 0,
        completed_at_utc,
        completion_sha256: String::new(),
    };
    completion.package_sha256s.sort();
    completion.response_sha256s.sort();
    completion.completion_sha256 =
        external_review_completion_commitment(&completion).map_err(|_| {
            vec![ExternalReviewCompletionIssue::SerializationFailed {
                field: "completion".into(),
            }]
        })?;
    Ok(completion)
}

pub fn validate_external_review_completion(
    completion: &ExternalReviewCompletionEvidence,
) -> Vec<ExternalReviewCompletionIssue> {
    let mut issues = Vec::new();
    if completion.completion_version != EXTERNAL_REVIEW_COMPLETION_VERSION {
        issues.push(ExternalReviewCompletionIssue::WrongVersion {
            found: completion.completion_version.clone(),
        });
    }
    for (field, digest) in [
        ("protocol_sha256", completion.protocol_sha256.as_str()),
        (
            "evidence_index_sha256",
            completion.evidence_index_sha256.as_str(),
        ),
        (
            "resolution_ledger_sha256",
            completion.resolution_ledger_sha256.as_str(),
        ),
        ("completion_sha256", completion.completion_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            issues.push(ExternalReviewCompletionIssue::InvalidDigest {
                field: field.into(),
            });
        }
    }
    for digest in completion
        .package_sha256s
        .iter()
        .chain(completion.response_sha256s.iter())
    {
        if !is_sha256(digest) {
            issues.push(ExternalReviewCompletionIssue::InvalidDigest {
                field: "package_or_response_sha256".into(),
            });
        }
    }
    if completion.completed_at_utc.trim().is_empty() {
        issues.push(ExternalReviewCompletionIssue::EmptyField {
            field: "completed_at_utc".into(),
        });
    }
    if completion.unresolved_finding_count > 0 || !completion.all_findings_resolved {
        issues.push(ExternalReviewCompletionIssue::UnresolvedFindings {
            found: completion.unresolved_finding_count,
        });
    }
    for (field, value) in [
        ("all_packages_valid", completion.all_packages_valid),
        ("all_responses_valid", completion.all_responses_valid),
        (
            "all_required_roles_complete",
            completion.all_required_roles_complete,
        ),
        ("all_findings_resolved", completion.all_findings_resolved),
    ] {
        if !value {
            issues.push(ExternalReviewCompletionIssue::CompletionFlagFalse {
                field: field.into(),
            });
        }
    }
    let blocking_verdicts = completion
        .role_completion
        .iter()
        .map(|role| role.blocking_verdict_count)
        .sum::<usize>();
    if blocking_verdicts > 0 {
        issues.push(ExternalReviewCompletionIssue::BlockingVerdictSummary {
            found: blocking_verdicts,
        });
    }
    for role in &completion.role_completion {
        if role.completed_reviewers < role.required_reviewers {
            issues.push(ExternalReviewCompletionIssue::IncompleteRole { role: role.role });
        }
    }
    match external_review_completion_commitment(completion) {
        Ok(value) if value == completion.completion_sha256 => {}
        Ok(_) => issues.push(ExternalReviewCompletionIssue::CompletionDigestMismatch),
        Err(_) => issues.push(ExternalReviewCompletionIssue::SerializationFailed {
            field: "completion".into(),
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
    fn completion_version_is_explicit() {
        assert!(EXTERNAL_REVIEW_COMPLETION_VERSION.ends_with("v1"));
    }

    #[test]
    fn severity_registry_can_represent_all_levels() {
        let mut counts = BTreeMap::new();
        for severity in [
            ExternalFindingSeverity::Informational,
            ExternalFindingSeverity::Minor,
            ExternalFindingSeverity::Major,
            ExternalFindingSeverity::Critical,
        ] {
            counts.insert(severity, 0usize);
        }
        assert_eq!(counts.len(), 4);
    }
}
