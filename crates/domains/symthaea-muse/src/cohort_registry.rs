// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Privacy-minimized pilot cohort registry.
//!
//! Raw names, email addresses, phone numbers, IP addresses, and payment details
//! are deliberately absent. Contact and compensation systems may hold their own
//! records, but the study evidence layer receives only pseudonymous tokens and
//! one-way duplicate-guard commitments.

use crate::evidence_digest::canonical_json_sha256;
use crate::pilot_protocol::FrozenPilotProtocol;
use crate::pilot_schedule::{PilotCohortSpec, PilotParticipantScheduleBook};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const PILOT_COHORT_REGISTRY_VERSION: &str = "symthaea-muse-pilot-cohort-registry-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PilotEligibilityEvidence {
    pub adult_confirmed: bool,
    pub informed_consent_capacity_confirmed: bool,
    pub study_language_understood: bool,
    pub audio_playback_check_passed: bool,
    pub self_reported_unmanaged_hearing_barrier: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PilotParticipantStatus {
    Enrolled,
    InProgress,
    Completed,
    Excluded,
    Withdrawn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompensationStatus {
    NotApplicable,
    Pending,
    Approved,
    PaidByExternalSystem,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PilotParticipantRecord {
    pub participant_token: String,
    /// One-way commitment produced by the recruitment system to detect repeat
    /// enrollment without placing the underlying identifier in study evidence.
    pub duplicate_guard_sha256: String,
    pub recruitment_source_code: String,
    pub eligibility: PilotEligibilityEvidence,
    pub consent_document_sha256: String,
    pub instructions_sha256: String,
    pub assigned_block_ids: Vec<String>,
    pub status: PilotParticipantStatus,
    pub enrolled_at_utc: String,
    pub completed_at_utc: Option<String>,
    pub exclusion_code: Option<String>,
    pub withdrawal_recorded_at_utc: Option<String>,
    pub compensation_status: CompensationStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PilotCohortRegistry {
    pub registry_version: String,
    pub pilot_protocol_sha256: String,
    pub pilot_schedule_sha256: String,
    pub cohort_id: String,
    pub wave_id: String,
    pub records: Vec<PilotParticipantRecord>,
    pub registry_sha256: String,
}

#[derive(Serialize)]
struct RegistryCommitment<'a> {
    registry_version: &'a str,
    pilot_protocol_sha256: &'a str,
    pilot_schedule_sha256: &'a str,
    cohort_id: &'a str,
    wave_id: &'a str,
    records: &'a [PilotParticipantRecord],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PilotCohortRegistryIssue {
    WrongVersion {
        found: String,
    },
    SerializationFailed {
        field: String,
    },
    DigestMismatch {
        field: String,
    },
    CohortIdentityMismatch {
        field: String,
    },
    DuplicateParticipantToken {
        participant_token: String,
    },
    UnknownParticipantToken {
        participant_token: String,
    },
    MissingParticipantToken {
        participant_token: String,
    },
    InvalidDigest {
        participant_token: String,
        field: String,
    },
    DuplicateGuardReused {
        participant_token: String,
    },
    EmptyField {
        participant_token: String,
        field: String,
    },
    IneligibleParticipant {
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
}

pub fn pilot_cohort_registry_commitment(
    registry: &PilotCohortRegistry,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&RegistryCommitment {
        registry_version: &registry.registry_version,
        pilot_protocol_sha256: &registry.pilot_protocol_sha256,
        pilot_schedule_sha256: &registry.pilot_schedule_sha256,
        cohort_id: &registry.cohort_id,
        wave_id: &registry.wave_id,
        records: &registry.records,
    })
}

pub fn seal_pilot_cohort_registry(
    registry: &mut PilotCohortRegistry,
) -> Result<(), serde_json::Error> {
    registry
        .records
        .sort_by(|left, right| left.participant_token.cmp(&right.participant_token));
    for record in &mut registry.records {
        record.assigned_block_ids.sort();
    }
    registry.registry_sha256 = pilot_cohort_registry_commitment(registry)?;
    Ok(())
}

pub fn validate_pilot_cohort_registry(
    protocol: &FrozenPilotProtocol,
    cohort: &PilotCohortSpec,
    schedule: &PilotParticipantScheduleBook,
    registry: &PilotCohortRegistry,
) -> Vec<PilotCohortRegistryIssue> {
    let mut issues = Vec::new();
    if registry.registry_version != PILOT_COHORT_REGISTRY_VERSION {
        issues.push(PilotCohortRegistryIssue::WrongVersion {
            found: registry.registry_version.clone(),
        });
    }
    verify_digest(
        protocol,
        &registry.pilot_protocol_sha256,
        "pilot_protocol_sha256",
        &mut issues,
    );
    verify_digest(
        schedule,
        &registry.pilot_schedule_sha256,
        "pilot_schedule_sha256",
        &mut issues,
    );
    if registry.cohort_id != cohort.cohort_id || registry.cohort_id != schedule.cohort_id {
        issues.push(PilotCohortRegistryIssue::CohortIdentityMismatch {
            field: "cohort_id".into(),
        });
    }
    if registry.wave_id != cohort.wave_id || registry.wave_id != schedule.wave_id {
        issues.push(PilotCohortRegistryIssue::CohortIdentityMismatch {
            field: "wave_id".into(),
        });
    }

    let expected_tokens: BTreeSet<_> = cohort.participant_tokens.iter().cloned().collect();
    let block_owner: BTreeMap<_, _> = schedule
        .blocks
        .iter()
        .map(|block| (block.block_id.as_str(), block.participant_token.as_str()))
        .collect();
    let expected_blocks: BTreeMap<_, Vec<_>> = cohort
        .participant_tokens
        .iter()
        .map(|token| {
            let mut blocks: Vec<_> = schedule
                .blocks
                .iter()
                .filter(|block| block.participant_token == *token)
                .map(|block| block.block_id.as_str())
                .collect();
            blocks.sort();
            (token.as_str(), blocks)
        })
        .collect();

    let mut tokens = BTreeSet::new();
    let mut duplicate_guards = BTreeSet::new();
    for record in &registry.records {
        if !tokens.insert(record.participant_token.clone()) {
            issues.push(PilotCohortRegistryIssue::DuplicateParticipantToken {
                participant_token: record.participant_token.clone(),
            });
        }
        if !expected_tokens.contains(&record.participant_token) {
            issues.push(PilotCohortRegistryIssue::UnknownParticipantToken {
                participant_token: record.participant_token.clone(),
            });
        }
        for (field, digest) in [
            (
                "duplicate_guard_sha256",
                record.duplicate_guard_sha256.as_str(),
            ),
            (
                "consent_document_sha256",
                record.consent_document_sha256.as_str(),
            ),
            ("instructions_sha256", record.instructions_sha256.as_str()),
        ] {
            if !is_sha256(digest) {
                issues.push(PilotCohortRegistryIssue::InvalidDigest {
                    participant_token: record.participant_token.clone(),
                    field: field.into(),
                });
            }
        }
        if !duplicate_guards.insert(record.duplicate_guard_sha256.as_str()) {
            issues.push(PilotCohortRegistryIssue::DuplicateGuardReused {
                participant_token: record.participant_token.clone(),
            });
        }
        for (field, value) in [
            ("participant_token", record.participant_token.as_str()),
            (
                "recruitment_source_code",
                record.recruitment_source_code.as_str(),
            ),
            ("enrolled_at_utc", record.enrolled_at_utc.as_str()),
        ] {
            if value.trim().is_empty() {
                issues.push(PilotCohortRegistryIssue::EmptyField {
                    participant_token: record.participant_token.clone(),
                    field: field.into(),
                });
            }
        }
        for (field, eligible) in [
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
            (
                "no_unmanaged_hearing_barrier",
                !record.eligibility.self_reported_unmanaged_hearing_barrier,
            ),
        ] {
            if !eligible {
                issues.push(PilotCohortRegistryIssue::IneligibleParticipant {
                    participant_token: record.participant_token.clone(),
                    field: field.into(),
                });
            }
        }
        let mut assigned = BTreeSet::new();
        for block_id in &record.assigned_block_ids {
            if !assigned.insert(block_id.as_str()) {
                issues.push(PilotCohortRegistryIssue::DuplicateAssignedBlock {
                    participant_token: record.participant_token.clone(),
                    block_id: block_id.clone(),
                });
            }
            match block_owner.get(block_id.as_str()) {
                None => issues.push(PilotCohortRegistryIssue::UnknownAssignedBlock {
                    participant_token: record.participant_token.clone(),
                    block_id: block_id.clone(),
                }),
                Some(owner) if *owner != record.participant_token => {
                    issues.push(PilotCohortRegistryIssue::BlockOwnedByDifferentParticipant {
                        participant_token: record.participant_token.clone(),
                        block_id: block_id.clone(),
                    });
                }
                Some(_) => {}
            }
        }
        if let Some(expected) = expected_blocks.get(record.participant_token.as_str()) {
            for block_id in expected {
                if !assigned.contains(block_id) {
                    issues.push(PilotCohortRegistryIssue::MissingAssignedBlock {
                        participant_token: record.participant_token.clone(),
                        block_id: (*block_id).to_string(),
                    });
                }
            }
        }
        validate_status(record, &mut issues);
    }
    for token in expected_tokens.difference(&tokens) {
        issues.push(PilotCohortRegistryIssue::MissingParticipantToken {
            participant_token: token.clone(),
        });
    }
    match pilot_cohort_registry_commitment(registry) {
        Ok(value) if value == registry.registry_sha256 => {}
        Ok(_) => issues.push(PilotCohortRegistryIssue::DigestMismatch {
            field: "registry_sha256".into(),
        }),
        Err(_) => issues.push(PilotCohortRegistryIssue::SerializationFailed {
            field: "registry".into(),
        }),
    }
    issues
}

fn validate_status(record: &PilotParticipantRecord, issues: &mut Vec<PilotCohortRegistryIssue>) {
    let mismatch = |field: &str, issues: &mut Vec<PilotCohortRegistryIssue>| {
        issues.push(PilotCohortRegistryIssue::StatusEvidenceMismatch {
            participant_token: record.participant_token.clone(),
            field: field.into(),
        });
    };
    match record.status {
        PilotParticipantStatus::Enrolled | PilotParticipantStatus::InProgress => {
            if record.completed_at_utc.is_some()
                || record.exclusion_code.is_some()
                || record.withdrawal_recorded_at_utc.is_some()
            {
                mismatch("open_status_has_terminal_evidence", issues);
            }
        }
        PilotParticipantStatus::Completed => {
            if record.completed_at_utc.as_deref().is_none_or(str::is_empty)
                || record.exclusion_code.is_some()
                || record.withdrawal_recorded_at_utc.is_some()
            {
                mismatch("completed", issues);
            }
        }
        PilotParticipantStatus::Excluded => {
            if record.exclusion_code.as_deref().is_none_or(str::is_empty)
                || record.withdrawal_recorded_at_utc.is_some()
            {
                mismatch("excluded", issues);
            }
        }
        PilotParticipantStatus::Withdrawn => {
            if record
                .withdrawal_recorded_at_utc
                .as_deref()
                .is_none_or(str::is_empty)
            {
                mismatch("withdrawn", issues);
            }
        }
    }
}

fn verify_digest<T: Serialize>(
    value: &T,
    expected: &str,
    field: &str,
    issues: &mut Vec<PilotCohortRegistryIssue>,
) {
    match canonical_json_sha256(value) {
        Ok(value) if value == expected => {}
        Ok(_) => issues.push(PilotCohortRegistryIssue::DigestMismatch {
            field: field.into(),
        }),
        Err(_) => issues.push(PilotCohortRegistryIssue::SerializationFailed {
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
    fn terminal_statuses_require_matching_evidence() {
        let mut issues = Vec::new();
        let record = PilotParticipantRecord {
            participant_token: "p-1".into(),
            duplicate_guard_sha256: "a".repeat(64),
            recruitment_source_code: "community".into(),
            eligibility: PilotEligibilityEvidence {
                adult_confirmed: true,
                informed_consent_capacity_confirmed: true,
                study_language_understood: true,
                audio_playback_check_passed: true,
                self_reported_unmanaged_hearing_barrier: false,
            },
            consent_document_sha256: "b".repeat(64),
            instructions_sha256: "c".repeat(64),
            assigned_block_ids: Vec::new(),
            status: PilotParticipantStatus::Completed,
            enrolled_at_utc: "2026-07-14T00:00:00Z".into(),
            completed_at_utc: None,
            exclusion_code: None,
            withdrawal_recorded_at_utc: None,
            compensation_status: CompensationStatus::Pending,
        };
        validate_status(&record, &mut issues);
        assert_eq!(issues.len(), 1);
    }
}
