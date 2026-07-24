// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Confirmatory-study amendment control after external review.

use crate::evidence_digest::canonical_json_sha256;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const CONFIRMATORY_AMENDMENT_LEDGER_VERSION: &str =
    "symthaea-muse-confirmatory-amendment-ledger-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfirmatoryAuthoritySnapshot {
    pub study_operations_release_sha256: String,
    pub external_review_protocol_sha256: String,
    pub external_review_resolution_sha256: String,
    pub confirmatory_manifest_sha256: String,
    pub methodology_sha256: String,
    pub analysis_plan_sha256: String,
    pub adaptive_checkpoint_sha256: String,
    pub runner_source_sha256: String,
    pub artifact_factory_sha256: String,
    pub preregistration_receipt_sha256: String,
    pub snapshot_sha256: String,
}

#[derive(Serialize)]
struct ConfirmatoryAuthoritySnapshotCommitment<'a> {
    study_operations_release_sha256: &'a str,
    external_review_protocol_sha256: &'a str,
    external_review_resolution_sha256: &'a str,
    confirmatory_manifest_sha256: &'a str,
    methodology_sha256: &'a str,
    analysis_plan_sha256: &'a str,
    adaptive_checkpoint_sha256: &'a str,
    runner_source_sha256: &'a str,
    artifact_factory_sha256: &'a str,
    preregistration_receipt_sha256: &'a str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ConfirmatoryAmendmentMateriality {
    Clerical,
    OperationalOutcomeNeutral,
    InstrumentOrParticipantFlow,
    AnalysisOrEndpoint,
    ModelOrHypothesis,
    PrivacyOrSafety,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfirmatoryAmendmentDecision {
    Rejected,
    AcceptedWithoutRefreeze,
    AcceptedWithFullRefreeze,
    EmergencySafetyStop,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfirmatoryAmendmentRecord {
    pub amendment_id: String,
    pub requested_at_utc: String,
    pub requested_by: String,
    pub affected_authority: String,
    pub previous_sha256: String,
    pub proposed_sha256: String,
    pub materiality: ConfirmatoryAmendmentMateriality,
    pub rationale: String,
    pub decision: ConfirmatoryAmendmentDecision,
    pub decided_at_utc: String,
    pub decision_authority: String,
    pub independent_approval_ids: Vec<String>,
    pub replacement_preregistration_receipt_sha256: String,
    pub replacement_review_resolution_sha256: String,
    pub external_receipt_uri: String,
    pub external_receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfirmatoryAmendmentLedger {
    pub ledger_version: String,
    pub baseline_authority: ConfirmatoryAuthoritySnapshot,
    pub amendments: Vec<ConfirmatoryAmendmentRecord>,
    pub confirmatory_collection_started_at_utc: Option<String>,
    pub locked_at_utc: String,
    pub ledger_sha256: String,
}

#[derive(Serialize)]
struct ConfirmatoryAmendmentLedgerCommitment<'a> {
    ledger_version: &'a str,
    baseline_authority: &'a ConfirmatoryAuthoritySnapshot,
    amendments: &'a [ConfirmatoryAmendmentRecord],
    confirmatory_collection_started_at_utc: &'a Option<String>,
    locked_at_utc: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfirmatoryAmendmentIssue {
    WrongVersion { found: String },
    SerializationFailed { field: String },
    InvalidDigest { field: String },
    EmptyField { field: String },
    BaselineSnapshotDigestMismatch,
    DuplicateAmendmentId { amendment_id: String },
    NoOpAmendment { amendment_id: String },
    AcceptedClericalChangeRequiresNoRefreeze { amendment_id: String },
    MaterialChangeWithoutFullRefreeze { amendment_id: String },
    MaterialChangeWithoutNewPreregistration { amendment_id: String },
    MaterialChangeWithoutNewExternalReview { amendment_id: String },
    AcceptedChangeWithoutIndependentApproval { amendment_id: String },
    AcceptedAfterCollectionStarted { amendment_id: String },
    EmergencyStopBeforeCollection { amendment_id: String },
    InvalidCollectionStart,
    LedgerDigestMismatch,
}

pub fn confirmatory_authority_snapshot_commitment(
    snapshot: &ConfirmatoryAuthoritySnapshot,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&ConfirmatoryAuthoritySnapshotCommitment {
        study_operations_release_sha256: &snapshot.study_operations_release_sha256,
        external_review_protocol_sha256: &snapshot.external_review_protocol_sha256,
        external_review_resolution_sha256: &snapshot.external_review_resolution_sha256,
        confirmatory_manifest_sha256: &snapshot.confirmatory_manifest_sha256,
        methodology_sha256: &snapshot.methodology_sha256,
        analysis_plan_sha256: &snapshot.analysis_plan_sha256,
        adaptive_checkpoint_sha256: &snapshot.adaptive_checkpoint_sha256,
        runner_source_sha256: &snapshot.runner_source_sha256,
        artifact_factory_sha256: &snapshot.artifact_factory_sha256,
        preregistration_receipt_sha256: &snapshot.preregistration_receipt_sha256,
    })
}

pub fn seal_confirmatory_authority_snapshot(
    snapshot: &mut ConfirmatoryAuthoritySnapshot,
) -> Result<(), serde_json::Error> {
    snapshot.snapshot_sha256 = confirmatory_authority_snapshot_commitment(snapshot)?;
    Ok(())
}

pub fn confirmatory_amendment_ledger_commitment(
    ledger: &ConfirmatoryAmendmentLedger,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&ConfirmatoryAmendmentLedgerCommitment {
        ledger_version: &ledger.ledger_version,
        baseline_authority: &ledger.baseline_authority,
        amendments: &ledger.amendments,
        confirmatory_collection_started_at_utc: &ledger.confirmatory_collection_started_at_utc,
        locked_at_utc: &ledger.locked_at_utc,
    })
}

pub fn seal_confirmatory_amendment_ledger(
    ledger: &mut ConfirmatoryAmendmentLedger,
) -> Result<(), serde_json::Error> {
    ledger
        .amendments
        .sort_by(|a, b| a.amendment_id.cmp(&b.amendment_id));
    for amendment in &mut ledger.amendments {
        amendment.independent_approval_ids.sort();
        amendment.independent_approval_ids.dedup();
    }
    ledger.ledger_sha256 = confirmatory_amendment_ledger_commitment(ledger)?;
    Ok(())
}

pub fn validate_confirmatory_amendment_ledger(
    ledger: &ConfirmatoryAmendmentLedger,
) -> Vec<ConfirmatoryAmendmentIssue> {
    let mut issues = Vec::new();
    if ledger.ledger_version != CONFIRMATORY_AMENDMENT_LEDGER_VERSION {
        issues.push(ConfirmatoryAmendmentIssue::WrongVersion {
            found: ledger.ledger_version.clone(),
        });
    }
    validate_snapshot(&ledger.baseline_authority, &mut issues);
    if ledger.locked_at_utc.trim().is_empty() {
        issues.push(ConfirmatoryAmendmentIssue::EmptyField {
            field: "locked_at_utc".into(),
        });
    }
    if ledger
        .confirmatory_collection_started_at_utc
        .as_ref()
        .is_some_and(|value| value.trim().is_empty())
    {
        issues.push(ConfirmatoryAmendmentIssue::InvalidCollectionStart);
    }

    let mut ids = BTreeSet::new();
    for amendment in &ledger.amendments {
        if !ids.insert(amendment.amendment_id.clone()) {
            issues.push(ConfirmatoryAmendmentIssue::DuplicateAmendmentId {
                amendment_id: amendment.amendment_id.clone(),
            });
        }
        for (field, value) in [
            ("amendment_id", amendment.amendment_id.as_str()),
            ("requested_at_utc", amendment.requested_at_utc.as_str()),
            ("requested_by", amendment.requested_by.as_str()),
            ("affected_authority", amendment.affected_authority.as_str()),
            ("rationale", amendment.rationale.as_str()),
            ("decided_at_utc", amendment.decided_at_utc.as_str()),
            ("decision_authority", amendment.decision_authority.as_str()),
            (
                "external_receipt_uri",
                amendment.external_receipt_uri.as_str(),
            ),
        ] {
            if value.trim().is_empty() {
                issues.push(ConfirmatoryAmendmentIssue::EmptyField {
                    field: format!("amendment.{}.{field}", amendment.amendment_id),
                });
            }
        }
        for (field, digest) in [
            ("previous_sha256", amendment.previous_sha256.as_str()),
            ("proposed_sha256", amendment.proposed_sha256.as_str()),
            (
                "replacement_preregistration_receipt_sha256",
                amendment
                    .replacement_preregistration_receipt_sha256
                    .as_str(),
            ),
            (
                "replacement_review_resolution_sha256",
                amendment.replacement_review_resolution_sha256.as_str(),
            ),
            (
                "external_receipt_sha256",
                amendment.external_receipt_sha256.as_str(),
            ),
        ] {
            if !is_sha256(digest) {
                issues.push(ConfirmatoryAmendmentIssue::InvalidDigest {
                    field: format!("amendment.{}.{field}", amendment.amendment_id),
                });
            }
        }
        if amendment.previous_sha256 == amendment.proposed_sha256 {
            issues.push(ConfirmatoryAmendmentIssue::NoOpAmendment {
                amendment_id: amendment.amendment_id.clone(),
            });
        }
        let accepted = matches!(
            amendment.decision,
            ConfirmatoryAmendmentDecision::AcceptedWithoutRefreeze
                | ConfirmatoryAmendmentDecision::AcceptedWithFullRefreeze
        );
        let material = matches!(
            amendment.materiality,
            ConfirmatoryAmendmentMateriality::InstrumentOrParticipantFlow
                | ConfirmatoryAmendmentMateriality::AnalysisOrEndpoint
                | ConfirmatoryAmendmentMateriality::ModelOrHypothesis
                | ConfirmatoryAmendmentMateriality::PrivacyOrSafety
        );
        if amendment.materiality == ConfirmatoryAmendmentMateriality::Clerical
            && amendment.decision == ConfirmatoryAmendmentDecision::AcceptedWithFullRefreeze
        {
            issues.push(
                ConfirmatoryAmendmentIssue::AcceptedClericalChangeRequiresNoRefreeze {
                    amendment_id: amendment.amendment_id.clone(),
                },
            );
        }
        if material
            && accepted
            && amendment.decision != ConfirmatoryAmendmentDecision::AcceptedWithFullRefreeze
        {
            issues.push(
                ConfirmatoryAmendmentIssue::MaterialChangeWithoutFullRefreeze {
                    amendment_id: amendment.amendment_id.clone(),
                },
            );
        }
        if material
            && accepted
            && amendment.replacement_preregistration_receipt_sha256 == "0".repeat(64)
        {
            issues.push(
                ConfirmatoryAmendmentIssue::MaterialChangeWithoutNewPreregistration {
                    amendment_id: amendment.amendment_id.clone(),
                },
            );
        }
        if material && accepted && amendment.replacement_review_resolution_sha256 == "0".repeat(64)
        {
            issues.push(
                ConfirmatoryAmendmentIssue::MaterialChangeWithoutNewExternalReview {
                    amendment_id: amendment.amendment_id.clone(),
                },
            );
        }
        if accepted && amendment.independent_approval_ids.is_empty() {
            issues.push(
                ConfirmatoryAmendmentIssue::AcceptedChangeWithoutIndependentApproval {
                    amendment_id: amendment.amendment_id.clone(),
                },
            );
        }
        if accepted && ledger.confirmatory_collection_started_at_utc.is_some() {
            issues.push(ConfirmatoryAmendmentIssue::AcceptedAfterCollectionStarted {
                amendment_id: amendment.amendment_id.clone(),
            });
        }
        if amendment.decision == ConfirmatoryAmendmentDecision::EmergencySafetyStop
            && ledger.confirmatory_collection_started_at_utc.is_none()
        {
            issues.push(ConfirmatoryAmendmentIssue::EmergencyStopBeforeCollection {
                amendment_id: amendment.amendment_id.clone(),
            });
        }
    }
    if !is_sha256(&ledger.ledger_sha256) {
        issues.push(ConfirmatoryAmendmentIssue::InvalidDigest {
            field: "ledger_sha256".into(),
        });
    }
    match confirmatory_amendment_ledger_commitment(ledger) {
        Ok(value) if value == ledger.ledger_sha256 => {}
        Ok(_) => issues.push(ConfirmatoryAmendmentIssue::LedgerDigestMismatch),
        Err(_) => issues.push(ConfirmatoryAmendmentIssue::SerializationFailed {
            field: "ledger".into(),
        }),
    }
    issues
}

fn validate_snapshot(
    snapshot: &ConfirmatoryAuthoritySnapshot,
    issues: &mut Vec<ConfirmatoryAmendmentIssue>,
) {
    for (field, digest) in [
        (
            "study_operations_release_sha256",
            snapshot.study_operations_release_sha256.as_str(),
        ),
        (
            "external_review_protocol_sha256",
            snapshot.external_review_protocol_sha256.as_str(),
        ),
        (
            "external_review_resolution_sha256",
            snapshot.external_review_resolution_sha256.as_str(),
        ),
        (
            "confirmatory_manifest_sha256",
            snapshot.confirmatory_manifest_sha256.as_str(),
        ),
        ("methodology_sha256", snapshot.methodology_sha256.as_str()),
        (
            "analysis_plan_sha256",
            snapshot.analysis_plan_sha256.as_str(),
        ),
        (
            "adaptive_checkpoint_sha256",
            snapshot.adaptive_checkpoint_sha256.as_str(),
        ),
        (
            "runner_source_sha256",
            snapshot.runner_source_sha256.as_str(),
        ),
        (
            "artifact_factory_sha256",
            snapshot.artifact_factory_sha256.as_str(),
        ),
        (
            "preregistration_receipt_sha256",
            snapshot.preregistration_receipt_sha256.as_str(),
        ),
        ("snapshot_sha256", snapshot.snapshot_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            issues.push(ConfirmatoryAmendmentIssue::InvalidDigest {
                field: format!("baseline_authority.{field}"),
            });
        }
    }
    match confirmatory_authority_snapshot_commitment(snapshot) {
        Ok(value) if value == snapshot.snapshot_sha256 => {}
        Ok(_) => issues.push(ConfirmatoryAmendmentIssue::BaselineSnapshotDigestMismatch),
        Err(_) => issues.push(ConfirmatoryAmendmentIssue::SerializationFailed {
            field: "baseline_authority".into(),
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
    fn collection_start_is_a_one_way_boundary() {
        let accepted = ConfirmatoryAmendmentDecision::AcceptedWithFullRefreeze;
        assert_ne!(accepted, ConfirmatoryAmendmentDecision::EmergencySafetyStop);
    }

    #[test]
    fn materiality_order_places_model_changes_above_clerical_changes() {
        assert!(
            ConfirmatoryAmendmentMateriality::ModelOrHypothesis
                > ConfirmatoryAmendmentMateriality::Clerical
        );
    }
}
