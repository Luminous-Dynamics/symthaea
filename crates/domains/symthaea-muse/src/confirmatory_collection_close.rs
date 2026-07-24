// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Irreversible close receipt for confirmatory participant collection.
//!
//! Closing binds the frozen protocol, final cohort registry, final outcome-blind
//! monitor snapshot, and sealed participant evidence. It requires independent
//! collection-custodian and governance authorization before unblinding.

use crate::confirmatory_cohort_registry::{
    ConfirmatoryCohortRegistry, validate_confirmatory_cohort_registry,
};
use crate::confirmatory_collection_monitor::{
    ConfirmatoryCollectionDecision, ConfirmatoryCollectionSnapshot,
    validate_confirmatory_collection_snapshot,
};
use crate::confirmatory_collection_protocol::{
    ConfirmatoryCollectionProtocol, validate_confirmatory_collection_protocol,
};
use crate::evidence_digest::canonical_json_sha256;
use crate::participant_evidence::{ParticipantEvidenceEnvelope, participant_evidence_commitment};
use crate::participant_schedule::{ParticipantCohortSpec, ParticipantScheduleBook};
use crate::study_evidence::{EvidenceBlockStatus, raw_evidence_commitment};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const CONFIRMATORY_COLLECTION_CLOSE_VERSION: &str =
    "symthaea-muse-confirmatory-collection-close-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfirmatoryCollectionCloseReason {
    FrozenTargetReached,
    FrozenDeadlineReached,
    GovernanceAbort,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ConfirmatoryCloseSignoffRole {
    CollectionCustodian,
    GovernanceOfficer,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfirmatoryCloseSignoff {
    pub role: ConfirmatoryCloseSignoffRole,
    pub signer_id: String,
    pub authorization_sha256: String,
    pub signed_at_utc: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfirmatoryCollectionCloseReceipt {
    pub close_version: String,
    pub study_id: String,
    pub protocol_sha256: String,
    pub cohort_registry_sha256: String,
    pub final_snapshot_sha256: String,
    pub participant_evidence_sha256: String,
    pub raw_evidence_sha256: String,
    pub included_listener_blocks: u32,
    pub excluded_listener_blocks: u32,
    pub workflow_blocks: u32,
    pub closed_at_utc: String,
    pub close_reason: ConfirmatoryCollectionCloseReason,
    pub collection_irreversibly_closed: bool,
    pub codebook_never_accessed_during_collection: bool,
    pub outcome_statistics_never_computed_during_collection: bool,
    pub signoffs: Vec<ConfirmatoryCloseSignoff>,
    pub receipt_sha256: String,
}

#[derive(Serialize)]
struct CloseCommitment<'a> {
    close_version: &'a str,
    study_id: &'a str,
    protocol_sha256: &'a str,
    cohort_registry_sha256: &'a str,
    final_snapshot_sha256: &'a str,
    participant_evidence_sha256: &'a str,
    raw_evidence_sha256: &'a str,
    included_listener_blocks: u32,
    excluded_listener_blocks: u32,
    workflow_blocks: u32,
    closed_at_utc: &'a str,
    close_reason: ConfirmatoryCollectionCloseReason,
    collection_irreversibly_closed: bool,
    codebook_never_accessed_during_collection: bool,
    outcome_statistics_never_computed_during_collection: bool,
    signoffs: &'a [ConfirmatoryCloseSignoff],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfirmatoryCollectionCloseIssue {
    InvalidProtocol,
    InvalidCohortRegistry,
    InvalidFinalSnapshot,
    WrongVersion {
        found: String,
    },
    EmptyField {
        field: String,
    },
    InvalidDigest {
        field: String,
    },
    EvidenceDigestMismatch {
        field: String,
    },
    EvidenceCountMismatch {
        field: String,
        expected: u32,
        found: u32,
    },
    CloseReasonMismatch,
    IntegrityIncidentStillOpen,
    CollectionNotIrreversiblyClosed,
    CodebookAccessedDuringCollection,
    OutcomeStatisticsComputedDuringCollection,
    MissingSignoff {
        role: ConfirmatoryCloseSignoffRole,
    },
    DuplicateSignoff {
        role: ConfirmatoryCloseSignoffRole,
    },
    DuplicateSigner,
    InvalidSignoff {
        role: ConfirmatoryCloseSignoffRole,
        field: String,
    },
    SerializationFailed,
    ReceiptDigestMismatch,
}

pub fn confirmatory_collection_close_commitment(
    receipt: &ConfirmatoryCollectionCloseReceipt,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&CloseCommitment {
        close_version: &receipt.close_version,
        study_id: &receipt.study_id,
        protocol_sha256: &receipt.protocol_sha256,
        cohort_registry_sha256: &receipt.cohort_registry_sha256,
        final_snapshot_sha256: &receipt.final_snapshot_sha256,
        participant_evidence_sha256: &receipt.participant_evidence_sha256,
        raw_evidence_sha256: &receipt.raw_evidence_sha256,
        included_listener_blocks: receipt.included_listener_blocks,
        excluded_listener_blocks: receipt.excluded_listener_blocks,
        workflow_blocks: receipt.workflow_blocks,
        closed_at_utc: &receipt.closed_at_utc,
        close_reason: receipt.close_reason,
        collection_irreversibly_closed: receipt.collection_irreversibly_closed,
        codebook_never_accessed_during_collection: receipt
            .codebook_never_accessed_during_collection,
        outcome_statistics_never_computed_during_collection: receipt
            .outcome_statistics_never_computed_during_collection,
        signoffs: &receipt.signoffs,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn build_confirmatory_collection_close(
    protocol: &ConfirmatoryCollectionProtocol,
    cohort: &ParticipantCohortSpec,
    participant_schedule: &ParticipantScheduleBook,
    registry: &ConfirmatoryCohortRegistry,
    final_snapshot: &ConfirmatoryCollectionSnapshot,
    evidence: &ParticipantEvidenceEnvelope,
    closed_at_utc: String,
    close_reason: ConfirmatoryCollectionCloseReason,
    mut signoffs: Vec<ConfirmatoryCloseSignoff>,
) -> Result<ConfirmatoryCollectionCloseReceipt, Vec<ConfirmatoryCollectionCloseIssue>> {
    let included_listener_blocks = evidence
        .evidence
        .listener_blocks
        .iter()
        .filter(|block| block.status == EvidenceBlockStatus::Included)
        .count() as u32;
    let excluded_listener_blocks = evidence
        .evidence
        .listener_blocks
        .iter()
        .filter(|block| matches!(block.status, EvidenceBlockStatus::Excluded { .. }))
        .count() as u32;
    let workflow_blocks = evidence.evidence.workflow_blocks.len() as u32;
    signoffs.sort_by_key(|signoff| signoff.role);
    let participant_evidence_sha256 = participant_evidence_commitment(evidence)
        .map_err(|_| vec![ConfirmatoryCollectionCloseIssue::SerializationFailed])?;
    let raw_evidence_sha256 = raw_evidence_commitment(&evidence.evidence)
        .map_err(|_| vec![ConfirmatoryCollectionCloseIssue::SerializationFailed])?;
    let mut receipt = ConfirmatoryCollectionCloseReceipt {
        close_version: CONFIRMATORY_COLLECTION_CLOSE_VERSION.into(),
        study_id: protocol.study_id.clone(),
        protocol_sha256: protocol.protocol_sha256.clone(),
        cohort_registry_sha256: registry.registry_sha256.clone(),
        final_snapshot_sha256: final_snapshot.snapshot_sha256.clone(),
        participant_evidence_sha256,
        raw_evidence_sha256,
        included_listener_blocks,
        excluded_listener_blocks,
        workflow_blocks,
        closed_at_utc,
        close_reason,
        collection_irreversibly_closed: true,
        codebook_never_accessed_during_collection: true,
        outcome_statistics_never_computed_during_collection: true,
        signoffs,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = confirmatory_collection_close_commitment(&receipt)
        .map_err(|_| vec![ConfirmatoryCollectionCloseIssue::SerializationFailed])?;
    let issues = validate_confirmatory_collection_close(
        protocol,
        cohort,
        participant_schedule,
        registry,
        final_snapshot,
        evidence,
        &receipt,
    );
    if issues.is_empty() {
        Ok(receipt)
    } else {
        Err(issues)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn validate_confirmatory_collection_close(
    protocol: &ConfirmatoryCollectionProtocol,
    cohort: &ParticipantCohortSpec,
    participant_schedule: &ParticipantScheduleBook,
    registry: &ConfirmatoryCohortRegistry,
    final_snapshot: &ConfirmatoryCollectionSnapshot,
    evidence: &ParticipantEvidenceEnvelope,
    receipt: &ConfirmatoryCollectionCloseReceipt,
) -> Vec<ConfirmatoryCollectionCloseIssue> {
    let mut issues = Vec::new();
    if !validate_confirmatory_collection_protocol(protocol).is_empty()
        || receipt.protocol_sha256 != protocol.protocol_sha256
    {
        issues.push(ConfirmatoryCollectionCloseIssue::InvalidProtocol);
    }
    if !validate_confirmatory_cohort_registry(
        cohort,
        participant_schedule,
        &protocol.protocol_sha256,
        registry,
    )
    .is_empty()
        || receipt.cohort_registry_sha256 != registry.registry_sha256
    {
        issues.push(ConfirmatoryCollectionCloseIssue::InvalidCohortRegistry);
    }
    if !validate_confirmatory_collection_snapshot(protocol, final_snapshot, None).is_empty()
        || receipt.final_snapshot_sha256 != final_snapshot.snapshot_sha256
    {
        issues.push(ConfirmatoryCollectionCloseIssue::InvalidFinalSnapshot);
    }
    if receipt.close_version != CONFIRMATORY_COLLECTION_CLOSE_VERSION {
        issues.push(ConfirmatoryCollectionCloseIssue::WrongVersion {
            found: receipt.close_version.clone(),
        });
    }
    for (field, value) in [
        ("study_id", receipt.study_id.as_str()),
        ("closed_at_utc", receipt.closed_at_utc.as_str()),
    ] {
        if value.trim().is_empty() {
            issues.push(ConfirmatoryCollectionCloseIssue::EmptyField {
                field: field.into(),
            });
        }
    }
    for (field, value) in [
        ("protocol_sha256", receipt.protocol_sha256.as_str()),
        (
            "cohort_registry_sha256",
            receipt.cohort_registry_sha256.as_str(),
        ),
        (
            "final_snapshot_sha256",
            receipt.final_snapshot_sha256.as_str(),
        ),
        (
            "participant_evidence_sha256",
            receipt.participant_evidence_sha256.as_str(),
        ),
        ("raw_evidence_sha256", receipt.raw_evidence_sha256.as_str()),
    ] {
        if !is_sha256(value) {
            issues.push(ConfirmatoryCollectionCloseIssue::InvalidDigest {
                field: field.into(),
            });
        }
    }
    verify_digest(
        "participant_evidence_sha256",
        participant_evidence_commitment(evidence),
        &receipt.participant_evidence_sha256,
        &mut issues,
    );
    verify_digest(
        "raw_evidence_sha256",
        raw_evidence_commitment(&evidence.evidence),
        &receipt.raw_evidence_sha256,
        &mut issues,
    );
    let included = evidence
        .evidence
        .listener_blocks
        .iter()
        .filter(|block| block.status == EvidenceBlockStatus::Included)
        .count() as u32;
    let excluded = evidence
        .evidence
        .listener_blocks
        .iter()
        .filter(|block| matches!(block.status, EvidenceBlockStatus::Excluded { .. }))
        .count() as u32;
    for (field, expected, found) in [
        (
            "included_listener_blocks",
            included,
            receipt.included_listener_blocks,
        ),
        (
            "excluded_listener_blocks",
            excluded,
            receipt.excluded_listener_blocks,
        ),
        (
            "workflow_blocks",
            evidence.evidence.workflow_blocks.len() as u32,
            receipt.workflow_blocks,
        ),
        (
            "snapshot.included_complete_count",
            final_snapshot.included_complete_count,
            receipt.included_listener_blocks,
        ),
        (
            "snapshot.excluded_complete_count",
            final_snapshot.excluded_complete_count,
            receipt.excluded_listener_blocks,
        ),
    ] {
        if expected != found {
            issues.push(ConfirmatoryCollectionCloseIssue::EvidenceCountMismatch {
                field: field.into(),
                expected,
                found,
            });
        }
    }
    let expected_reason = match final_snapshot.decision {
        ConfirmatoryCollectionDecision::CloseTargetReached => {
            Some(ConfirmatoryCollectionCloseReason::FrozenTargetReached)
        }
        ConfirmatoryCollectionDecision::CloseFrozenDeadline => {
            Some(ConfirmatoryCollectionCloseReason::FrozenDeadlineReached)
        }
        ConfirmatoryCollectionDecision::AbortGovernance => {
            Some(ConfirmatoryCollectionCloseReason::GovernanceAbort)
        }
        ConfirmatoryCollectionDecision::Continue
        | ConfirmatoryCollectionDecision::PauseOperationalIntegrity => None,
    };
    if expected_reason != Some(receipt.close_reason) {
        issues.push(ConfirmatoryCollectionCloseIssue::CloseReasonMismatch);
    }
    if final_snapshot.integrity_incident_open
        && receipt.close_reason != ConfirmatoryCollectionCloseReason::GovernanceAbort
    {
        issues.push(ConfirmatoryCollectionCloseIssue::IntegrityIncidentStillOpen);
    }
    if !receipt.collection_irreversibly_closed {
        issues.push(ConfirmatoryCollectionCloseIssue::CollectionNotIrreversiblyClosed);
    }
    if !receipt.codebook_never_accessed_during_collection {
        issues.push(ConfirmatoryCollectionCloseIssue::CodebookAccessedDuringCollection);
    }
    if !receipt.outcome_statistics_never_computed_during_collection {
        issues.push(ConfirmatoryCollectionCloseIssue::OutcomeStatisticsComputedDuringCollection);
    }
    validate_signoffs(receipt, &mut issues);
    match confirmatory_collection_close_commitment(receipt) {
        Ok(found) if found == receipt.receipt_sha256 => {}
        Ok(_) => issues.push(ConfirmatoryCollectionCloseIssue::ReceiptDigestMismatch),
        Err(_) => issues.push(ConfirmatoryCollectionCloseIssue::SerializationFailed),
    }
    issues
}

fn validate_signoffs(
    receipt: &ConfirmatoryCollectionCloseReceipt,
    issues: &mut Vec<ConfirmatoryCollectionCloseIssue>,
) {
    let mut roles = BTreeSet::new();
    let mut signers = BTreeSet::new();
    for signoff in &receipt.signoffs {
        if !roles.insert(signoff.role) {
            issues.push(ConfirmatoryCollectionCloseIssue::DuplicateSignoff { role: signoff.role });
        }
        if !signers.insert(signoff.signer_id.as_str()) {
            issues.push(ConfirmatoryCollectionCloseIssue::DuplicateSigner);
        }
        for (field, valid) in [
            ("signer_id", !signoff.signer_id.trim().is_empty()),
            (
                "authorization_sha256",
                is_sha256(&signoff.authorization_sha256),
            ),
            ("signed_at_utc", !signoff.signed_at_utc.trim().is_empty()),
        ] {
            if !valid {
                issues.push(ConfirmatoryCollectionCloseIssue::InvalidSignoff {
                    role: signoff.role,
                    field: field.into(),
                });
            }
        }
    }
    for role in [
        ConfirmatoryCloseSignoffRole::CollectionCustodian,
        ConfirmatoryCloseSignoffRole::GovernanceOfficer,
    ] {
        if !roles.contains(&role) {
            issues.push(ConfirmatoryCollectionCloseIssue::MissingSignoff { role });
        }
    }
}

fn verify_digest(
    field: &str,
    expected: Result<String, serde_json::Error>,
    found: &str,
    issues: &mut Vec<ConfirmatoryCollectionCloseIssue>,
) {
    match expected {
        Ok(value) if value == found => {}
        Ok(_) => issues.push(ConfirmatoryCollectionCloseIssue::EvidenceDigestMismatch {
            field: field.into(),
        }),
        Err(_) => issues.push(ConfirmatoryCollectionCloseIssue::SerializationFailed),
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn close_requires_two_distinct_authorities() {
        let receipt = ConfirmatoryCollectionCloseReceipt {
            close_version: CONFIRMATORY_COLLECTION_CLOSE_VERSION.into(),
            study_id: "study".into(),
            protocol_sha256: "1".repeat(64),
            cohort_registry_sha256: "2".repeat(64),
            final_snapshot_sha256: "3".repeat(64),
            participant_evidence_sha256: "4".repeat(64),
            raw_evidence_sha256: "5".repeat(64),
            included_listener_blocks: 1,
            excluded_listener_blocks: 0,
            workflow_blocks: 0,
            closed_at_utc: "now".into(),
            close_reason: ConfirmatoryCollectionCloseReason::FrozenTargetReached,
            collection_irreversibly_closed: true,
            codebook_never_accessed_during_collection: true,
            outcome_statistics_never_computed_during_collection: true,
            signoffs: vec![ConfirmatoryCloseSignoff {
                role: ConfirmatoryCloseSignoffRole::CollectionCustodian,
                signer_id: "same".into(),
                authorization_sha256: "a".repeat(64),
                signed_at_utc: "now".into(),
            }],
            receipt_sha256: String::new(),
        };
        let mut issues = Vec::new();
        validate_signoffs(&receipt, &mut issues);
        assert!(
            issues.contains(&ConfirmatoryCollectionCloseIssue::MissingSignoff {
                role: ConfirmatoryCloseSignoffRole::GovernanceOfficer
            })
        );
    }
}
