// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Immutable disposition ledger for external-review findings.

use crate::evidence_digest::canonical_json_sha256;
use crate::external_review_protocol::FrozenExternalReviewProtocol;
use crate::external_review_response::{ExternalFindingSeverity, ExternalReviewResponse};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const EXTERNAL_REVIEW_RESOLUTION_VERSION: &str = "symthaea-muse-external-review-resolution-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FindingDisposition {
    Open,
    Fixed,
    RejectedWithRationale,
    DeferredToFutureWork,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalFindingResolution {
    pub finding_id: String,
    pub source_response_sha256: String,
    pub disposition: FindingDisposition,
    pub resolution_summary: String,
    pub change_set_sha256: String,
    pub replacement_evidence_sha256: Vec<String>,
    pub resolved_at_utc: String,
    pub resolved_by: String,
    pub reviewer_acceptance_id: String,
    pub reviewer_acceptance_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalReviewResolutionLedger {
    pub ledger_version: String,
    pub protocol_sha256: String,
    pub response_sha256s: Vec<String>,
    pub resolutions: Vec<ExternalFindingResolution>,
    pub locked_at_utc: String,
    pub ledger_sha256: String,
}

#[derive(Serialize)]
struct ExternalReviewResolutionCommitment<'a> {
    ledger_version: &'a str,
    protocol_sha256: &'a str,
    response_sha256s: &'a [String],
    resolutions: &'a [ExternalFindingResolution],
    locked_at_utc: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExternalReviewResolutionIssue {
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
    DuplicateResponseDigest {
        digest: String,
    },
    MissingResponseDigest {
        reviewer_id: String,
    },
    UnknownResponseDigest {
        digest: String,
    },
    DuplicateResolution {
        finding_id: String,
    },
    MissingResolution {
        finding_id: String,
    },
    UnknownFinding {
        finding_id: String,
    },
    DuplicateFindingAcrossResponses {
        finding_id: String,
    },
    SourceResponseMismatch {
        finding_id: String,
    },
    OpenBlockingFinding {
        finding_id: String,
    },
    SevereFindingNotFixed {
        finding_id: String,
        severity: ExternalFindingSeverity,
        disposition: FindingDisposition,
    },
    BlockingFindingRejected {
        finding_id: String,
    },
    FixedFindingWithoutChangeSet {
        finding_id: String,
    },
    FixedFindingWithoutEvidence {
        finding_id: String,
    },
    MissingReviewerAcceptance {
        finding_id: String,
    },
    AcceptanceReviewerUnknown {
        finding_id: String,
        reviewer_id: String,
    },
    AcceptanceReviewerMismatch {
        finding_id: String,
        expected_reviewer_id: String,
        found_reviewer_id: String,
    },
    LedgerDigestMismatch,
}

pub fn external_review_resolution_commitment(
    ledger: &ExternalReviewResolutionLedger,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&ExternalReviewResolutionCommitment {
        ledger_version: &ledger.ledger_version,
        protocol_sha256: &ledger.protocol_sha256,
        response_sha256s: &ledger.response_sha256s,
        resolutions: &ledger.resolutions,
        locked_at_utc: &ledger.locked_at_utc,
    })
}

pub fn seal_external_review_resolution_ledger(
    ledger: &mut ExternalReviewResolutionLedger,
) -> Result<(), serde_json::Error> {
    ledger.response_sha256s.sort();
    ledger.response_sha256s.dedup();
    ledger
        .resolutions
        .sort_by(|a, b| a.finding_id.cmp(&b.finding_id));
    for resolution in &mut ledger.resolutions {
        resolution.replacement_evidence_sha256.sort();
        resolution.replacement_evidence_sha256.dedup();
    }
    ledger.ledger_sha256 = external_review_resolution_commitment(ledger)?;
    Ok(())
}

pub fn validate_external_review_resolution_ledger(
    protocol: &FrozenExternalReviewProtocol,
    responses: &[ExternalReviewResponse],
    ledger: &ExternalReviewResolutionLedger,
) -> Vec<ExternalReviewResolutionIssue> {
    let mut issues = Vec::new();
    if ledger.ledger_version != EXTERNAL_REVIEW_RESOLUTION_VERSION {
        issues.push(ExternalReviewResolutionIssue::WrongVersion {
            found: ledger.ledger_version.clone(),
        });
    }
    if ledger.protocol_sha256 != protocol.protocol_sha256 {
        issues.push(ExternalReviewResolutionIssue::DigestMismatch {
            field: "protocol_sha256".into(),
        });
    }
    if ledger.locked_at_utc.trim().is_empty() {
        issues.push(ExternalReviewResolutionIssue::EmptyField {
            field: "locked_at_utc".into(),
        });
    }

    let mut expected_responses = BTreeMap::<String, &ExternalReviewResponse>::new();
    for response in responses {
        expected_responses.insert(response.response_sha256.clone(), response);
        if !ledger.response_sha256s.contains(&response.response_sha256) {
            issues.push(ExternalReviewResolutionIssue::MissingResponseDigest {
                reviewer_id: response.reviewer_id.clone(),
            });
        }
    }
    let mut response_digests = BTreeSet::new();
    for digest in &ledger.response_sha256s {
        if !response_digests.insert(digest.clone()) {
            issues.push(ExternalReviewResolutionIssue::DuplicateResponseDigest {
                digest: digest.clone(),
            });
        }
        if !is_sha256(digest) {
            issues.push(ExternalReviewResolutionIssue::InvalidDigest {
                field: "response_sha256s".into(),
            });
        }
        if !expected_responses.contains_key(digest) {
            issues.push(ExternalReviewResolutionIssue::UnknownResponseDigest {
                digest: digest.clone(),
            });
        }
    }

    let reviewers = protocol
        .reviewers
        .iter()
        .map(|reviewer| reviewer.reviewer_id.as_str())
        .collect::<BTreeSet<_>>();
    let mut findings = BTreeMap::new();
    for response in responses {
        for finding in &response.findings {
            if findings
                .insert(
                    finding.finding_id.clone(),
                    (
                        response.response_sha256.as_str(),
                        response.reviewer_id.as_str(),
                        finding,
                    ),
                )
                .is_some()
            {
                issues.push(
                    ExternalReviewResolutionIssue::DuplicateFindingAcrossResponses {
                        finding_id: finding.finding_id.clone(),
                    },
                );
            }
        }
    }
    let mut resolutions = BTreeSet::new();
    for resolution in &ledger.resolutions {
        if !resolutions.insert(resolution.finding_id.clone()) {
            issues.push(ExternalReviewResolutionIssue::DuplicateResolution {
                finding_id: resolution.finding_id.clone(),
            });
        }
        let Some((source_response, source_reviewer_id, finding)) =
            findings.get(&resolution.finding_id)
        else {
            issues.push(ExternalReviewResolutionIssue::UnknownFinding {
                finding_id: resolution.finding_id.clone(),
            });
            continue;
        };
        if resolution.source_response_sha256 != *source_response {
            issues.push(ExternalReviewResolutionIssue::SourceResponseMismatch {
                finding_id: resolution.finding_id.clone(),
            });
        }
        for (field, digest) in [
            (
                "source_response_sha256",
                resolution.source_response_sha256.as_str(),
            ),
            ("change_set_sha256", resolution.change_set_sha256.as_str()),
            (
                "reviewer_acceptance_sha256",
                resolution.reviewer_acceptance_sha256.as_str(),
            ),
        ] {
            if !is_sha256(digest) {
                issues.push(ExternalReviewResolutionIssue::InvalidDigest {
                    field: format!("resolution.{}.{field}", resolution.finding_id),
                });
            }
        }
        for digest in &resolution.replacement_evidence_sha256 {
            if !is_sha256(digest) {
                issues.push(ExternalReviewResolutionIssue::InvalidDigest {
                    field: format!(
                        "resolution.{}.replacement_evidence_sha256",
                        resolution.finding_id
                    ),
                });
            }
        }
        for (field, value) in [
            ("resolution_summary", resolution.resolution_summary.as_str()),
            ("resolved_at_utc", resolution.resolved_at_utc.as_str()),
            ("resolved_by", resolution.resolved_by.as_str()),
        ] {
            if value.trim().is_empty() {
                issues.push(ExternalReviewResolutionIssue::EmptyField {
                    field: format!("resolution.{}.{field}", resolution.finding_id),
                });
            }
        }
        if resolution.disposition == FindingDisposition::Open
            && finding.blocks_confirmatory_collection
        {
            issues.push(ExternalReviewResolutionIssue::OpenBlockingFinding {
                finding_id: resolution.finding_id.clone(),
            });
        }
        if matches!(
            finding.severity,
            ExternalFindingSeverity::Major | ExternalFindingSeverity::Critical
        ) && resolution.disposition != FindingDisposition::Fixed
        {
            issues.push(ExternalReviewResolutionIssue::SevereFindingNotFixed {
                finding_id: resolution.finding_id.clone(),
                severity: finding.severity,
                disposition: resolution.disposition,
            });
        }
        if finding.blocks_confirmatory_collection
            && resolution.disposition == FindingDisposition::RejectedWithRationale
        {
            issues.push(ExternalReviewResolutionIssue::BlockingFindingRejected {
                finding_id: resolution.finding_id.clone(),
            });
        }
        if resolution.disposition == FindingDisposition::Fixed {
            if resolution.change_set_sha256 == "0".repeat(64) {
                issues.push(
                    ExternalReviewResolutionIssue::FixedFindingWithoutChangeSet {
                        finding_id: resolution.finding_id.clone(),
                    },
                );
            }
            if resolution.replacement_evidence_sha256.is_empty() {
                issues.push(ExternalReviewResolutionIssue::FixedFindingWithoutEvidence {
                    finding_id: resolution.finding_id.clone(),
                });
            }
        }
        if resolution.reviewer_acceptance_id.trim().is_empty()
            || !is_sha256(&resolution.reviewer_acceptance_sha256)
            || resolution.reviewer_acceptance_sha256 == "0".repeat(64)
        {
            issues.push(ExternalReviewResolutionIssue::MissingReviewerAcceptance {
                finding_id: resolution.finding_id.clone(),
            });
        } else if !reviewers.contains(resolution.reviewer_acceptance_id.as_str()) {
            issues.push(ExternalReviewResolutionIssue::AcceptanceReviewerUnknown {
                finding_id: resolution.finding_id.clone(),
                reviewer_id: resolution.reviewer_acceptance_id.clone(),
            });
        } else if resolution.reviewer_acceptance_id.as_str() != *source_reviewer_id {
            issues.push(ExternalReviewResolutionIssue::AcceptanceReviewerMismatch {
                finding_id: resolution.finding_id.clone(),
                expected_reviewer_id: (*source_reviewer_id).into(),
                found_reviewer_id: resolution.reviewer_acceptance_id.clone(),
            });
        }
    }
    for finding_id in findings.keys() {
        if !resolutions.contains(finding_id) {
            issues.push(ExternalReviewResolutionIssue::MissingResolution {
                finding_id: finding_id.clone(),
            });
        }
    }
    for (field, digest) in [("ledger_sha256", ledger.ledger_sha256.as_str())] {
        if !is_sha256(digest) {
            issues.push(ExternalReviewResolutionIssue::InvalidDigest {
                field: field.into(),
            });
        }
    }
    match external_review_resolution_commitment(ledger) {
        Ok(value) if value == ledger.ledger_sha256 => {}
        Ok(_) => issues.push(ExternalReviewResolutionIssue::LedgerDigestMismatch),
        Err(_) => issues.push(ExternalReviewResolutionIssue::SerializationFailed {
            field: "ledger".into(),
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
    fn severe_findings_cannot_be_deferred_by_policy() {
        assert_ne!(
            FindingDisposition::DeferredToFutureWork,
            FindingDisposition::Fixed
        );
    }

    #[test]
    fn ledger_commitment_ignores_its_own_digest() {
        let ledger = ExternalReviewResolutionLedger {
            ledger_version: EXTERNAL_REVIEW_RESOLUTION_VERSION.into(),
            protocol_sha256: "a".repeat(64),
            response_sha256s: vec![],
            resolutions: vec![],
            locked_at_utc: "2026-07-14T00:00:00Z".into(),
            ledger_sha256: "b".repeat(64),
        };
        let digest = external_review_resolution_commitment(&ledger).unwrap();
        let mut changed = ledger;
        changed.ledger_sha256 = "c".repeat(64);
        assert_eq!(
            digest,
            external_review_resolution_commitment(&changed).unwrap()
        );
    }
}
