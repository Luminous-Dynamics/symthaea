// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Frozen operational authority for confirmatory participant collection.
//!
//! The protocol binds collection to the exact readiness release and public
//! preregistration. It deliberately excludes outcome-dependent stopping rules:
//! confirmatory accrual may stop only for the frozen target, deadline,
//! governance/safety intervention, or an operational integrity failure.

use crate::evidence_digest::canonical_json_sha256;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const CONFIRMATORY_COLLECTION_PROTOCOL_VERSION: &str =
    "symthaea-muse-confirmatory-collection-protocol-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ConfirmatoryDataRole {
    CollectionOperator,
    GovernanceOfficer,
    EvidenceCustodian,
    BlindedMonitor,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfirmatoryCollectionProtocol {
    pub protocol_version: String,
    pub study_id: String,
    pub readiness_release_sha256: String,
    pub external_preregistration_sha256: String,
    pub manifest_sha256: String,
    pub methodology_sha256: String,
    pub blinded_schedule_sha256: String,
    pub participant_schedule_sha256: String,
    pub artifact_bundle_sha256: String,
    pub cohort_registry_sha256: String,
    pub planned_open_utc: String,
    pub planned_close_utc: String,
    pub target_complete_blocks: u32,
    pub maximum_enrolled_participants: u32,
    pub maximum_exclusion_rate_basis_points: u32,
    pub minimum_completion_rate_basis_points: u32,
    pub outcome_monitoring_prohibited: bool,
    pub codebook_access_prohibited: bool,
    pub collection_roles: Vec<ConfirmatoryDataRole>,
    pub protocol_sha256: String,
}

#[derive(Serialize)]
struct ProtocolCommitment<'a> {
    protocol_version: &'a str,
    study_id: &'a str,
    readiness_release_sha256: &'a str,
    external_preregistration_sha256: &'a str,
    manifest_sha256: &'a str,
    methodology_sha256: &'a str,
    blinded_schedule_sha256: &'a str,
    participant_schedule_sha256: &'a str,
    artifact_bundle_sha256: &'a str,
    cohort_registry_sha256: &'a str,
    planned_open_utc: &'a str,
    planned_close_utc: &'a str,
    target_complete_blocks: u32,
    maximum_enrolled_participants: u32,
    maximum_exclusion_rate_basis_points: u32,
    minimum_completion_rate_basis_points: u32,
    outcome_monitoring_prohibited: bool,
    codebook_access_prohibited: bool,
    collection_roles: &'a [ConfirmatoryDataRole],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfirmatoryCollectionProtocolIssue {
    WrongVersion { found: String },
    EmptyField { field: String },
    InvalidDigest { field: String },
    InvalidTarget,
    MaximumEnrollmentBelowTarget,
    InvalidExclusionThreshold,
    InvalidCompletionThreshold,
    OutcomeMonitoringNotProhibited,
    CodebookAccessNotProhibited,
    DuplicateRole { role: ConfirmatoryDataRole },
    MissingRole { role: ConfirmatoryDataRole },
    SerializationFailed,
    ProtocolDigestMismatch,
}

pub fn confirmatory_collection_protocol_commitment(
    protocol: &ConfirmatoryCollectionProtocol,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&ProtocolCommitment {
        protocol_version: &protocol.protocol_version,
        study_id: &protocol.study_id,
        readiness_release_sha256: &protocol.readiness_release_sha256,
        external_preregistration_sha256: &protocol.external_preregistration_sha256,
        manifest_sha256: &protocol.manifest_sha256,
        methodology_sha256: &protocol.methodology_sha256,
        blinded_schedule_sha256: &protocol.blinded_schedule_sha256,
        participant_schedule_sha256: &protocol.participant_schedule_sha256,
        artifact_bundle_sha256: &protocol.artifact_bundle_sha256,
        cohort_registry_sha256: &protocol.cohort_registry_sha256,
        planned_open_utc: &protocol.planned_open_utc,
        planned_close_utc: &protocol.planned_close_utc,
        target_complete_blocks: protocol.target_complete_blocks,
        maximum_enrolled_participants: protocol.maximum_enrolled_participants,
        maximum_exclusion_rate_basis_points: protocol.maximum_exclusion_rate_basis_points,
        minimum_completion_rate_basis_points: protocol.minimum_completion_rate_basis_points,
        outcome_monitoring_prohibited: protocol.outcome_monitoring_prohibited,
        codebook_access_prohibited: protocol.codebook_access_prohibited,
        collection_roles: &protocol.collection_roles,
    })
}

pub fn seal_confirmatory_collection_protocol(
    protocol: &mut ConfirmatoryCollectionProtocol,
) -> Result<(), Vec<ConfirmatoryCollectionProtocolIssue>> {
    protocol.collection_roles.sort();
    protocol.protocol_sha256 = confirmatory_collection_protocol_commitment(protocol)
        .map_err(|_| vec![ConfirmatoryCollectionProtocolIssue::SerializationFailed])?;
    let issues = validate_confirmatory_collection_protocol(protocol);
    if issues.is_empty() {
        Ok(())
    } else {
        Err(issues)
    }
}

pub fn validate_confirmatory_collection_protocol(
    protocol: &ConfirmatoryCollectionProtocol,
) -> Vec<ConfirmatoryCollectionProtocolIssue> {
    let mut issues = Vec::new();
    if protocol.protocol_version != CONFIRMATORY_COLLECTION_PROTOCOL_VERSION {
        issues.push(ConfirmatoryCollectionProtocolIssue::WrongVersion {
            found: protocol.protocol_version.clone(),
        });
    }
    for (field, value) in [
        ("study_id", protocol.study_id.as_str()),
        ("planned_open_utc", protocol.planned_open_utc.as_str()),
        ("planned_close_utc", protocol.planned_close_utc.as_str()),
    ] {
        if value.trim().is_empty() {
            issues.push(ConfirmatoryCollectionProtocolIssue::EmptyField {
                field: field.into(),
            });
        }
    }
    for (field, value) in [
        (
            "readiness_release_sha256",
            protocol.readiness_release_sha256.as_str(),
        ),
        (
            "external_preregistration_sha256",
            protocol.external_preregistration_sha256.as_str(),
        ),
        ("manifest_sha256", protocol.manifest_sha256.as_str()),
        ("methodology_sha256", protocol.methodology_sha256.as_str()),
        (
            "blinded_schedule_sha256",
            protocol.blinded_schedule_sha256.as_str(),
        ),
        (
            "participant_schedule_sha256",
            protocol.participant_schedule_sha256.as_str(),
        ),
        (
            "artifact_bundle_sha256",
            protocol.artifact_bundle_sha256.as_str(),
        ),
        (
            "cohort_registry_sha256",
            protocol.cohort_registry_sha256.as_str(),
        ),
    ] {
        if !is_sha256(value) {
            issues.push(ConfirmatoryCollectionProtocolIssue::InvalidDigest {
                field: field.into(),
            });
        }
    }
    if protocol.target_complete_blocks == 0 {
        issues.push(ConfirmatoryCollectionProtocolIssue::InvalidTarget);
    }
    if protocol.maximum_enrolled_participants < protocol.target_complete_blocks {
        issues.push(ConfirmatoryCollectionProtocolIssue::MaximumEnrollmentBelowTarget);
    }
    if protocol.maximum_exclusion_rate_basis_points > 10_000 {
        issues.push(ConfirmatoryCollectionProtocolIssue::InvalidExclusionThreshold);
    }
    if protocol.minimum_completion_rate_basis_points > 10_000 {
        issues.push(ConfirmatoryCollectionProtocolIssue::InvalidCompletionThreshold);
    }
    if !protocol.outcome_monitoring_prohibited {
        issues.push(ConfirmatoryCollectionProtocolIssue::OutcomeMonitoringNotProhibited);
    }
    if !protocol.codebook_access_prohibited {
        issues.push(ConfirmatoryCollectionProtocolIssue::CodebookAccessNotProhibited);
    }
    let mut seen = BTreeSet::new();
    for role in &protocol.collection_roles {
        if !seen.insert(*role) {
            issues.push(ConfirmatoryCollectionProtocolIssue::DuplicateRole { role: *role });
        }
    }
    for role in [
        ConfirmatoryDataRole::CollectionOperator,
        ConfirmatoryDataRole::GovernanceOfficer,
        ConfirmatoryDataRole::EvidenceCustodian,
        ConfirmatoryDataRole::BlindedMonitor,
    ] {
        if !seen.contains(&role) {
            issues.push(ConfirmatoryCollectionProtocolIssue::MissingRole { role });
        }
    }
    match confirmatory_collection_protocol_commitment(protocol) {
        Ok(found) if found == protocol.protocol_sha256 => {}
        Ok(_) => issues.push(ConfirmatoryCollectionProtocolIssue::ProtocolDigestMismatch),
        Err(_) => issues.push(ConfirmatoryCollectionProtocolIssue::SerializationFailed),
    }
    issues
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_protocol() -> ConfirmatoryCollectionProtocol {
        let mut protocol = ConfirmatoryCollectionProtocol {
            protocol_version: CONFIRMATORY_COLLECTION_PROTOCOL_VERSION.into(),
            study_id: "study-v12".into(),
            readiness_release_sha256: "1".repeat(64),
            external_preregistration_sha256: "2".repeat(64),
            manifest_sha256: "3".repeat(64),
            methodology_sha256: "4".repeat(64),
            blinded_schedule_sha256: "5".repeat(64),
            participant_schedule_sha256: "6".repeat(64),
            artifact_bundle_sha256: "7".repeat(64),
            cohort_registry_sha256: "8".repeat(64),
            planned_open_utc: "2026-08-01T00:00:00Z".into(),
            planned_close_utc: "2026-09-01T00:00:00Z".into(),
            target_complete_blocks: 96,
            maximum_enrolled_participants: 128,
            maximum_exclusion_rate_basis_points: 2_500,
            minimum_completion_rate_basis_points: 7_500,
            outcome_monitoring_prohibited: true,
            codebook_access_prohibited: true,
            collection_roles: vec![
                ConfirmatoryDataRole::CollectionOperator,
                ConfirmatoryDataRole::GovernanceOfficer,
                ConfirmatoryDataRole::EvidenceCustodian,
                ConfirmatoryDataRole::BlindedMonitor,
            ],
            protocol_sha256: String::new(),
        };
        seal_confirmatory_collection_protocol(&mut protocol).unwrap();
        protocol
    }

    #[test]
    fn valid_protocol_is_self_verifying() {
        assert!(validate_confirmatory_collection_protocol(&valid_protocol()).is_empty());
    }

    #[test]
    fn outcome_monitoring_cannot_be_enabled() {
        let mut protocol = valid_protocol();
        protocol.outcome_monitoring_prohibited = false;
        protocol.protocol_sha256 = confirmatory_collection_protocol_commitment(&protocol).unwrap();
        assert!(
            validate_confirmatory_collection_protocol(&protocol)
                .contains(&ConfirmatoryCollectionProtocolIssue::OutcomeMonitoringNotProhibited)
        );
    }
}
