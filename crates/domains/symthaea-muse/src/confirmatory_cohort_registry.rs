// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Privacy-minimized confirmatory cohort registry.
//!
//! The registry binds every participant token to the frozen participant
//! schedule while excluding names, addresses, email, phone, IP, payment, and
//! raw duplicate-detection identifiers from the scientific evidence package.

use crate::evidence_digest::canonical_json_sha256;
use crate::participant_schedule::{ParticipantCohortSpec, ParticipantScheduleBook};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const CONFIRMATORY_COHORT_REGISTRY_VERSION: &str =
    "symthaea-muse-confirmatory-cohort-registry-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfirmatoryEligibilityEvidence {
    pub adult_confirmed: bool,
    pub informed_consent_capacity_confirmed: bool,
    pub study_language_understood: bool,
    pub audio_playback_check_passed: bool,
    pub unmanaged_hearing_barrier_reported: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfirmatoryParticipantStatus {
    Enrolled,
    InProgress,
    CompletedIncluded,
    CompletedExcluded,
    Withdrawn,
    OperationalFailure,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfirmatoryParticipantRecord {
    pub participant_token: String,
    pub duplicate_guard_sha256: String,
    pub recruitment_source_code: String,
    pub site_id: String,
    pub eligibility: ConfirmatoryEligibilityEvidence,
    pub consent_document_sha256: String,
    pub instructions_sha256: String,
    pub assigned_block_ids: Vec<String>,
    pub status: ConfirmatoryParticipantStatus,
    pub enrolled_at_utc: String,
    pub completed_at_utc: Option<String>,
    pub exclusion_code: Option<String>,
    pub withdrawal_recorded_at_utc: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfirmatoryCohortRegistry {
    pub registry_version: String,
    pub collection_protocol_sha256: String,
    pub participant_schedule_sha256: String,
    pub cohort_id: String,
    pub records: Vec<ConfirmatoryParticipantRecord>,
    pub registry_sha256: String,
}

#[derive(Serialize)]
struct RegistryCommitment<'a> {
    registry_version: &'a str,
    collection_protocol_sha256: &'a str,
    participant_schedule_sha256: &'a str,
    cohort_id: &'a str,
    records: &'a [ConfirmatoryParticipantRecord],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfirmatoryCohortRegistryIssue {
    WrongVersion {
        found: String,
    },
    DigestMismatch {
        field: String,
    },
    SerializationFailed,
    EmptyField {
        participant_token: String,
        field: String,
    },
    DuplicateParticipantToken {
        participant_token: String,
    },
    DuplicateGuardReused {
        participant_token: String,
    },
    UnknownParticipantToken {
        participant_token: String,
    },
    MissingParticipantToken {
        participant_token: String,
    },
    IneligibleParticipant {
        participant_token: String,
        field: String,
    },
    InvalidDigest {
        participant_token: String,
        field: String,
    },
    UnknownAssignedBlock {
        participant_token: String,
        block_id: String,
    },
    BlockOwnedByDifferentParticipant {
        participant_token: String,
        block_id: String,
    },
    DuplicateAssignedBlock {
        participant_token: String,
        block_id: String,
    },
    MissingAssignedBlock {
        participant_token: String,
        block_id: String,
    },
    StatusEvidenceMismatch {
        participant_token: String,
        field: String,
    },
    CohortMismatch,
}

pub fn confirmatory_cohort_registry_commitment(
    registry: &ConfirmatoryCohortRegistry,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&RegistryCommitment {
        registry_version: &registry.registry_version,
        collection_protocol_sha256: &registry.collection_protocol_sha256,
        participant_schedule_sha256: &registry.participant_schedule_sha256,
        cohort_id: &registry.cohort_id,
        records: &registry.records,
    })
}

pub fn seal_confirmatory_cohort_registry(
    registry: &mut ConfirmatoryCohortRegistry,
) -> Result<(), serde_json::Error> {
    registry
        .records
        .sort_by(|left, right| left.participant_token.cmp(&right.participant_token));
    for record in &mut registry.records {
        record.assigned_block_ids.sort();
    }
    registry.registry_sha256 = confirmatory_cohort_registry_commitment(registry)?;
    Ok(())
}

pub fn validate_confirmatory_cohort_registry(
    cohort: &ParticipantCohortSpec,
    schedule: &ParticipantScheduleBook,
    collection_protocol_sha256: &str,
    registry: &ConfirmatoryCohortRegistry,
) -> Vec<ConfirmatoryCohortRegistryIssue> {
    let mut issues = Vec::new();
    if registry.registry_version != CONFIRMATORY_COHORT_REGISTRY_VERSION {
        issues.push(ConfirmatoryCohortRegistryIssue::WrongVersion {
            found: registry.registry_version.clone(),
        });
    }
    if registry.collection_protocol_sha256 != collection_protocol_sha256 {
        issues.push(ConfirmatoryCohortRegistryIssue::DigestMismatch {
            field: "collection_protocol_sha256".into(),
        });
    }
    match canonical_json_sha256(schedule) {
        Ok(found) if found == registry.participant_schedule_sha256 => {}
        Ok(_) => issues.push(ConfirmatoryCohortRegistryIssue::DigestMismatch {
            field: "participant_schedule_sha256".into(),
        }),
        Err(_) => issues.push(ConfirmatoryCohortRegistryIssue::SerializationFailed),
    }
    if registry.cohort_id != cohort.cohort_id || registry.cohort_id != schedule.cohort_id {
        issues.push(ConfirmatoryCohortRegistryIssue::CohortMismatch);
    }

    let cohort_tokens: BTreeSet<_> = cohort
        .participant_tokens
        .iter()
        .map(String::as_str)
        .collect();
    let mut record_tokens = BTreeSet::new();
    let mut duplicate_guards = BTreeSet::new();
    let assignments: BTreeMap<_, Vec<_>> =
        schedule
            .blocks
            .iter()
            .fold(BTreeMap::<&str, Vec<&str>>::new(), |mut map, block| {
                map.entry(block.participant_token.as_str())
                    .or_default()
                    .push(block.block_id.as_str());
                map
            });
    let block_owner: BTreeMap<_, _> = schedule
        .blocks
        .iter()
        .map(|block| (block.block_id.as_str(), block.participant_token.as_str()))
        .collect();

    for record in &registry.records {
        let token = record.participant_token.as_str();
        if !record_tokens.insert(token) {
            issues.push(ConfirmatoryCohortRegistryIssue::DuplicateParticipantToken {
                participant_token: record.participant_token.clone(),
            });
        }
        if !cohort_tokens.contains(token) {
            issues.push(ConfirmatoryCohortRegistryIssue::UnknownParticipantToken {
                participant_token: record.participant_token.clone(),
            });
        }
        if !is_sha256(&record.duplicate_guard_sha256) {
            issues.push(ConfirmatoryCohortRegistryIssue::InvalidDigest {
                participant_token: record.participant_token.clone(),
                field: "duplicate_guard_sha256".into(),
            });
        } else if !duplicate_guards.insert(record.duplicate_guard_sha256.as_str()) {
            issues.push(ConfirmatoryCohortRegistryIssue::DuplicateGuardReused {
                participant_token: record.participant_token.clone(),
            });
        }
        for (field, value) in [
            (
                "recruitment_source_code",
                record.recruitment_source_code.as_str(),
            ),
            ("site_id", record.site_id.as_str()),
            ("enrolled_at_utc", record.enrolled_at_utc.as_str()),
        ] {
            if value.trim().is_empty() {
                issues.push(ConfirmatoryCohortRegistryIssue::EmptyField {
                    participant_token: record.participant_token.clone(),
                    field: field.into(),
                });
            }
        }
        for (field, value) in [
            (
                "consent_document_sha256",
                record.consent_document_sha256.as_str(),
            ),
            ("instructions_sha256", record.instructions_sha256.as_str()),
        ] {
            if !is_sha256(value) {
                issues.push(ConfirmatoryCohortRegistryIssue::InvalidDigest {
                    participant_token: record.participant_token.clone(),
                    field: field.into(),
                });
            }
        }
        for (field, accepted) in [
            ("adult_confirmed", record.eligibility.adult_confirmed),
            (
                "informed_consent_capacity_confirmed",
                record.eligibility.informed_consent_capacity_confirmed,
            ),
            (
                "study_language_understood",
                record.eligibility.study_language_understood,
            ),
            (
                "audio_playback_check_passed",
                record.eligibility.audio_playback_check_passed,
            ),
        ] {
            if !accepted {
                issues.push(ConfirmatoryCohortRegistryIssue::IneligibleParticipant {
                    participant_token: record.participant_token.clone(),
                    field: field.into(),
                });
            }
        }
        if record.eligibility.unmanaged_hearing_barrier_reported {
            issues.push(ConfirmatoryCohortRegistryIssue::IneligibleParticipant {
                participant_token: record.participant_token.clone(),
                field: "unmanaged_hearing_barrier_reported".into(),
            });
        }
        let mut assigned = BTreeSet::new();
        for block_id in &record.assigned_block_ids {
            if !assigned.insert(block_id.as_str()) {
                issues.push(ConfirmatoryCohortRegistryIssue::DuplicateAssignedBlock {
                    participant_token: record.participant_token.clone(),
                    block_id: block_id.clone(),
                });
            }
            match block_owner.get(block_id.as_str()) {
                None => issues.push(ConfirmatoryCohortRegistryIssue::UnknownAssignedBlock {
                    participant_token: record.participant_token.clone(),
                    block_id: block_id.clone(),
                }),
                Some(owner) if *owner != token => {
                    issues.push(
                        ConfirmatoryCohortRegistryIssue::BlockOwnedByDifferentParticipant {
                            participant_token: record.participant_token.clone(),
                            block_id: block_id.clone(),
                        },
                    );
                }
                Some(_) => {}
            }
        }
        for expected in assignments.get(token).into_iter().flatten() {
            if !assigned.contains(*expected) {
                issues.push(ConfirmatoryCohortRegistryIssue::MissingAssignedBlock {
                    participant_token: record.participant_token.clone(),
                    block_id: (*expected).into(),
                });
            }
        }
        validate_status_evidence(record, &mut issues);
    }
    for token in cohort_tokens.difference(&record_tokens) {
        issues.push(ConfirmatoryCohortRegistryIssue::MissingParticipantToken {
            participant_token: (*token).into(),
        });
    }
    match confirmatory_cohort_registry_commitment(registry) {
        Ok(found) if found == registry.registry_sha256 => {}
        Ok(_) => issues.push(ConfirmatoryCohortRegistryIssue::DigestMismatch {
            field: "registry_sha256".into(),
        }),
        Err(_) => issues.push(ConfirmatoryCohortRegistryIssue::SerializationFailed),
    }
    issues
}

fn validate_status_evidence(
    record: &ConfirmatoryParticipantRecord,
    issues: &mut Vec<ConfirmatoryCohortRegistryIssue>,
) {
    let token = record.participant_token.clone();
    let completed = matches!(
        record.status,
        ConfirmatoryParticipantStatus::CompletedIncluded
            | ConfirmatoryParticipantStatus::CompletedExcluded
    );
    if completed != record.completed_at_utc.is_some() {
        issues.push(ConfirmatoryCohortRegistryIssue::StatusEvidenceMismatch {
            participant_token: token.clone(),
            field: "completed_at_utc".into(),
        });
    }
    let excluded = record.status == ConfirmatoryParticipantStatus::CompletedExcluded;
    if excluded != record.exclusion_code.is_some() {
        issues.push(ConfirmatoryCohortRegistryIssue::StatusEvidenceMismatch {
            participant_token: token.clone(),
            field: "exclusion_code".into(),
        });
    }
    let withdrawn = record.status == ConfirmatoryParticipantStatus::Withdrawn;
    if withdrawn != record.withdrawal_recorded_at_utc.is_some() {
        issues.push(ConfirmatoryCohortRegistryIssue::StatusEvidenceMismatch {
            participant_token: token,
            field: "withdrawal_recorded_at_utc".into(),
        });
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_contact_fields_are_absent_from_record_schema() {
        let value = serde_json::to_value(ConfirmatoryParticipantRecord {
            participant_token: "p1".into(),
            duplicate_guard_sha256: "a".repeat(64),
            recruitment_source_code: "panel".into(),
            site_id: "remote".into(),
            eligibility: ConfirmatoryEligibilityEvidence {
                adult_confirmed: true,
                informed_consent_capacity_confirmed: true,
                study_language_understood: true,
                audio_playback_check_passed: true,
                unmanaged_hearing_barrier_reported: false,
            },
            consent_document_sha256: "b".repeat(64),
            instructions_sha256: "c".repeat(64),
            assigned_block_ids: vec!["block".into()],
            status: ConfirmatoryParticipantStatus::Enrolled,
            enrolled_at_utc: "now".into(),
            completed_at_utc: None,
            exclusion_code: None,
            withdrawal_recorded_at_utc: None,
        })
        .unwrap();
        let object = value.as_object().unwrap();
        for forbidden in ["name", "email", "phone", "ip_address", "payment_details"] {
            assert!(!object.contains_key(forbidden));
        }
    }
}
