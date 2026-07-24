// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Append-only post-publication correction, addendum, and retraction ledger.
//!
//! The original publication record is immutable. Later corrections are linked
//! in a forward digest chain, retain the superseded claim text, and require a
//! public notice. Retraction is terminal for the publication lineage.

use crate::confirmatory_publication::{
    ConfirmatoryPublicationRecord, confirmatory_publication_commitment,
};
use crate::evidence_digest::canonical_json_sha256;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const POST_PUBLICATION_AUDIT_VERSION: &str = "symthaea-muse-post-publication-audit-v1";
const POST_PUBLICATION_GENESIS_SHA256: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PostPublicationEventKind {
    Correction,
    DataAddendum,
    AnalysisAddendum,
    ReplicationNotice,
    Retraction,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostPublicationClaimChange {
    pub claim_id: String,
    pub previous_text: String,
    pub replacement_text: String,
    pub evidence_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostPublicationAuditEvent {
    pub sequence: u32,
    pub event_kind: PostPublicationEventKind,
    pub recorded_at_utc: String,
    pub public_notice_uri: String,
    pub reason: String,
    pub authority_id: String,
    pub authorization_sha256: String,
    pub claim_changes: Vec<PostPublicationClaimChange>,
    pub supporting_evidence_sha256: Vec<String>,
    pub previous_event_sha256: String,
    pub event_sha256: String,
}

#[derive(Serialize)]
struct EventCommitment<'a> {
    publication_sha256: &'a str,
    sequence: u32,
    event_kind: PostPublicationEventKind,
    recorded_at_utc: &'a str,
    public_notice_uri: &'a str,
    reason: &'a str,
    authority_id: &'a str,
    authorization_sha256: &'a str,
    claim_changes: &'a [PostPublicationClaimChange],
    supporting_evidence_sha256: &'a [String],
    previous_event_sha256: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostPublicationAuditLedger {
    pub audit_version: String,
    pub study_id: String,
    pub publication_sha256: String,
    pub events: Vec<PostPublicationAuditEvent>,
    pub current_publication_status: PublicationStatus,
    pub ledger_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PublicationStatus {
    Active,
    Corrected,
    Retracted,
}

#[derive(Serialize)]
struct LedgerCommitment<'a> {
    audit_version: &'a str,
    study_id: &'a str,
    publication_sha256: &'a str,
    events: &'a [PostPublicationAuditEvent],
    current_publication_status: PublicationStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PostPublicationAuditIssue {
    InvalidPublication,
    WrongVersion {
        found: String,
    },
    EmptyField {
        sequence: u32,
        field: String,
    },
    InvalidDigest {
        sequence: u32,
        field: String,
    },
    UnexpectedSequence {
        expected: u32,
        found: u32,
    },
    ChainBroken {
        sequence: u32,
    },
    EventDigestMismatch {
        sequence: u32,
    },
    DuplicateClaimChange {
        sequence: u32,
        claim_id: String,
    },
    EmptyClaimChange {
        sequence: u32,
        claim_id: String,
        field: String,
    },
    RetractionWithoutPublicNotice,
    EventAfterRetraction {
        sequence: u32,
    },
    StatusMismatch,
    SerializationFailed,
    LedgerDigestMismatch,
}

pub fn new_post_publication_audit(
    publication: &ConfirmatoryPublicationRecord,
) -> Result<PostPublicationAuditLedger, Vec<PostPublicationAuditIssue>> {
    let publication_sha256 = confirmatory_publication_commitment(publication)
        .map_err(|_| vec![PostPublicationAuditIssue::SerializationFailed])?;
    if publication_sha256 != publication.record_sha256 {
        return Err(vec![PostPublicationAuditIssue::InvalidPublication]);
    }
    let mut ledger = PostPublicationAuditLedger {
        audit_version: POST_PUBLICATION_AUDIT_VERSION.into(),
        study_id: publication.study_id.clone(),
        publication_sha256,
        events: Vec::new(),
        current_publication_status: PublicationStatus::Active,
        ledger_sha256: String::new(),
    };
    ledger.ledger_sha256 = post_publication_audit_commitment(&ledger)
        .map_err(|_| vec![PostPublicationAuditIssue::SerializationFailed])?;
    Ok(ledger)
}

pub fn post_publication_event_commitment(
    publication_sha256: &str,
    event: &PostPublicationAuditEvent,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&EventCommitment {
        publication_sha256,
        sequence: event.sequence,
        event_kind: event.event_kind,
        recorded_at_utc: &event.recorded_at_utc,
        public_notice_uri: &event.public_notice_uri,
        reason: &event.reason,
        authority_id: &event.authority_id,
        authorization_sha256: &event.authorization_sha256,
        claim_changes: &event.claim_changes,
        supporting_evidence_sha256: &event.supporting_evidence_sha256,
        previous_event_sha256: &event.previous_event_sha256,
    })
}

pub fn post_publication_audit_commitment(
    ledger: &PostPublicationAuditLedger,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&LedgerCommitment {
        audit_version: &ledger.audit_version,
        study_id: &ledger.study_id,
        publication_sha256: &ledger.publication_sha256,
        events: &ledger.events,
        current_publication_status: ledger.current_publication_status,
    })
}

pub fn append_post_publication_event(
    ledger: &mut PostPublicationAuditLedger,
    event_kind: PostPublicationEventKind,
    recorded_at_utc: String,
    public_notice_uri: String,
    reason: String,
    authority_id: String,
    authorization_sha256: String,
    mut claim_changes: Vec<PostPublicationClaimChange>,
    mut supporting_evidence_sha256: Vec<String>,
) -> Result<(), Vec<PostPublicationAuditIssue>> {
    let current_issues = validate_post_publication_audit(ledger);
    if !current_issues.is_empty() {
        return Err(current_issues);
    }
    if ledger.current_publication_status == PublicationStatus::Retracted {
        return Err(vec![PostPublicationAuditIssue::EventAfterRetraction {
            sequence: ledger.events.len() as u32 + 1,
        }]);
    }
    claim_changes.sort_by(|left, right| left.claim_id.cmp(&right.claim_id));
    supporting_evidence_sha256.sort();
    let previous_event_sha256 = ledger
        .events
        .last()
        .map_or(POST_PUBLICATION_GENESIS_SHA256, |event| {
            event.event_sha256.as_str()
        })
        .to_string();
    let mut event = PostPublicationAuditEvent {
        sequence: ledger.events.len() as u32 + 1,
        event_kind,
        recorded_at_utc,
        public_notice_uri,
        reason,
        authority_id,
        authorization_sha256,
        claim_changes,
        supporting_evidence_sha256,
        previous_event_sha256,
        event_sha256: String::new(),
    };
    event.event_sha256 = post_publication_event_commitment(&ledger.publication_sha256, &event)
        .map_err(|_| vec![PostPublicationAuditIssue::SerializationFailed])?;
    ledger.events.push(event);
    ledger.current_publication_status = if event_kind == PostPublicationEventKind::Retraction {
        PublicationStatus::Retracted
    } else {
        PublicationStatus::Corrected
    };
    ledger.ledger_sha256 = post_publication_audit_commitment(ledger)
        .map_err(|_| vec![PostPublicationAuditIssue::SerializationFailed])?;
    let issues = validate_post_publication_audit(ledger);
    if issues.is_empty() {
        Ok(())
    } else {
        Err(issues)
    }
}

pub fn validate_post_publication_audit(
    ledger: &PostPublicationAuditLedger,
) -> Vec<PostPublicationAuditIssue> {
    let mut issues = Vec::new();
    if ledger.audit_version != POST_PUBLICATION_AUDIT_VERSION {
        issues.push(PostPublicationAuditIssue::WrongVersion {
            found: ledger.audit_version.clone(),
        });
    }
    if ledger.study_id.trim().is_empty() || !is_sha256(&ledger.publication_sha256) {
        issues.push(PostPublicationAuditIssue::InvalidPublication);
    }
    let mut previous = POST_PUBLICATION_GENESIS_SHA256.to_string();
    let mut retracted = false;
    for (index, event) in ledger.events.iter().enumerate() {
        let expected = index as u32 + 1;
        if event.sequence != expected {
            issues.push(PostPublicationAuditIssue::UnexpectedSequence {
                expected,
                found: event.sequence,
            });
        }
        if retracted {
            issues.push(PostPublicationAuditIssue::EventAfterRetraction {
                sequence: event.sequence,
            });
        }
        if event.previous_event_sha256 != previous {
            issues.push(PostPublicationAuditIssue::ChainBroken {
                sequence: event.sequence,
            });
        }
        for (field, value) in [
            ("recorded_at_utc", event.recorded_at_utc.as_str()),
            ("public_notice_uri", event.public_notice_uri.as_str()),
            ("reason", event.reason.as_str()),
            ("authority_id", event.authority_id.as_str()),
        ] {
            if value.trim().is_empty() {
                issues.push(PostPublicationAuditIssue::EmptyField {
                    sequence: event.sequence,
                    field: field.into(),
                });
            }
        }
        if !is_sha256(&event.authorization_sha256) {
            issues.push(PostPublicationAuditIssue::InvalidDigest {
                sequence: event.sequence,
                field: "authorization_sha256".into(),
            });
        }
        let mut claims = BTreeSet::new();
        for change in &event.claim_changes {
            if !claims.insert(change.claim_id.as_str()) {
                issues.push(PostPublicationAuditIssue::DuplicateClaimChange {
                    sequence: event.sequence,
                    claim_id: change.claim_id.clone(),
                });
            }
            for (field, valid) in [
                ("claim_id", !change.claim_id.trim().is_empty()),
                ("previous_text", !change.previous_text.trim().is_empty()),
                (
                    "replacement_text",
                    !change.replacement_text.trim().is_empty(),
                ),
                ("evidence_sha256", is_sha256(&change.evidence_sha256)),
            ] {
                if !valid {
                    issues.push(PostPublicationAuditIssue::EmptyClaimChange {
                        sequence: event.sequence,
                        claim_id: change.claim_id.clone(),
                        field: field.into(),
                    });
                }
            }
        }
        for digest in &event.supporting_evidence_sha256 {
            if !is_sha256(digest) {
                issues.push(PostPublicationAuditIssue::InvalidDigest {
                    sequence: event.sequence,
                    field: "supporting_evidence_sha256".into(),
                });
            }
        }
        if event.event_kind == PostPublicationEventKind::Retraction
            && event.public_notice_uri.trim().is_empty()
        {
            issues.push(PostPublicationAuditIssue::RetractionWithoutPublicNotice);
        }
        match post_publication_event_commitment(&ledger.publication_sha256, event) {
            Ok(found) if found == event.event_sha256 => {}
            Ok(_) => issues.push(PostPublicationAuditIssue::EventDigestMismatch {
                sequence: event.sequence,
            }),
            Err(_) => issues.push(PostPublicationAuditIssue::SerializationFailed),
        }
        previous = event.event_sha256.clone();
        retracted = event.event_kind == PostPublicationEventKind::Retraction;
    }
    let expected_status = if retracted {
        PublicationStatus::Retracted
    } else if ledger.events.is_empty() {
        PublicationStatus::Active
    } else {
        PublicationStatus::Corrected
    };
    if ledger.current_publication_status != expected_status {
        issues.push(PostPublicationAuditIssue::StatusMismatch);
    }
    match post_publication_audit_commitment(ledger) {
        Ok(found) if found == ledger.ledger_sha256 => {}
        Ok(_) => issues.push(PostPublicationAuditIssue::LedgerDigestMismatch),
        Err(_) => issues.push(PostPublicationAuditIssue::SerializationFailed),
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
    fn retraction_is_terminal() {
        let mut ledger = PostPublicationAuditLedger {
            audit_version: POST_PUBLICATION_AUDIT_VERSION.into(),
            study_id: "study".into(),
            publication_sha256: "a".repeat(64),
            events: Vec::new(),
            current_publication_status: PublicationStatus::Active,
            ledger_sha256: String::new(),
        };
        ledger.ledger_sha256 = post_publication_audit_commitment(&ledger).unwrap();
        append_post_publication_event(
            &mut ledger,
            PostPublicationEventKind::Retraction,
            "now".into(),
            "https://example.invalid/retraction".into(),
            "reason".into(),
            "authority".into(),
            "b".repeat(64),
            Vec::new(),
            vec!["c".repeat(64)],
        )
        .unwrap();
        let result = append_post_publication_event(
            &mut ledger,
            PostPublicationEventKind::Correction,
            "later".into(),
            "https://example.invalid/correction".into(),
            "reason".into(),
            "authority".into(),
            "d".repeat(64),
            Vec::new(),
            Vec::new(),
        );
        assert!(matches!(
            result,
            Err(found) if found.iter().any(|issue| matches!(issue, PostPublicationAuditIssue::EventAfterRetraction { .. }))
        ));
    }
}
