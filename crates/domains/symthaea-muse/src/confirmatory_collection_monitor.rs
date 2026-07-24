// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Outcome-blind confirmatory accrual monitoring.
//!
//! Monitoring records operational facts only: assignments, completed sessions,
//! exclusions, withdrawals, timing, and evidence-chain integrity. It contains
//! no ratings, ranks, policy labels, arm identities, or endpoint estimates.

use crate::confirmatory_collection_protocol::{
    ConfirmatoryCollectionProtocol, validate_confirmatory_collection_protocol,
};
use crate::evidence_digest::canonical_json_sha256;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const CONFIRMATORY_COLLECTION_MONITOR_VERSION: &str =
    "symthaea-muse-confirmatory-collection-monitor-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ConfirmatorySessionDisposition {
    Assigned,
    Started,
    CompleteIncluded,
    CompleteExcluded,
    Withdrawn,
    OperationalFailure,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfirmatorySessionStatus {
    pub participant_token: String,
    pub block_id: String,
    pub site_id: String,
    pub disposition: ConfirmatorySessionDisposition,
    pub package_sha256: String,
    pub session_log_sha256: Option<String>,
    pub preregistered_exclusion_code: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfirmatoryCollectionDecision {
    Continue,
    PauseOperationalIntegrity,
    CloseTargetReached,
    CloseFrozenDeadline,
    AbortGovernance,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfirmatoryCollectionSnapshot {
    pub monitor_version: String,
    pub protocol_sha256: String,
    pub snapshot_sequence: u32,
    pub recorded_at_utc: String,
    pub previous_snapshot_sha256: String,
    pub sessions: Vec<ConfirmatorySessionStatus>,
    pub enrolled_count: u32,
    pub started_count: u32,
    pub included_complete_count: u32,
    pub excluded_complete_count: u32,
    pub withdrawn_count: u32,
    pub operational_failure_count: u32,
    pub exclusion_rate_basis_points: u32,
    pub completion_rate_basis_points: u32,
    pub integrity_incident_open: bool,
    pub frozen_deadline_reached: bool,
    pub governance_abort_order_sha256: Option<String>,
    pub decision: ConfirmatoryCollectionDecision,
    pub snapshot_sha256: String,
}

#[derive(Serialize)]
struct SnapshotCommitment<'a> {
    monitor_version: &'a str,
    protocol_sha256: &'a str,
    snapshot_sequence: u32,
    recorded_at_utc: &'a str,
    previous_snapshot_sha256: &'a str,
    sessions: &'a [ConfirmatorySessionStatus],
    enrolled_count: u32,
    started_count: u32,
    included_complete_count: u32,
    excluded_complete_count: u32,
    withdrawn_count: u32,
    operational_failure_count: u32,
    exclusion_rate_basis_points: u32,
    completion_rate_basis_points: u32,
    integrity_incident_open: bool,
    frozen_deadline_reached: bool,
    governance_abort_order_sha256: &'a Option<String>,
    decision: ConfirmatoryCollectionDecision,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfirmatoryCollectionMonitorIssue {
    InvalidProtocol,
    WrongVersion {
        found: String,
    },
    InvalidDigest {
        field: String,
    },
    EmptyField {
        field: String,
    },
    InvalidSequence,
    DuplicateParticipant {
        participant_token: String,
    },
    DuplicateBlock {
        block_id: String,
    },
    InvalidSessionDigest {
        block_id: String,
    },
    MissingCompletedSessionDigest {
        block_id: String,
    },
    UnexpectedCompletedSessionDigest {
        block_id: String,
    },
    ExclusionCodeMismatch {
        block_id: String,
    },
    CountMismatch {
        field: String,
        expected: u32,
        found: u32,
    },
    RateMismatch {
        field: String,
        expected: u32,
        found: u32,
    },
    EnrollmentExceedsMaximum,
    DecisionMismatch {
        expected: ConfirmatoryCollectionDecision,
        found: ConfirmatoryCollectionDecision,
    },
    GovernanceAbortDigestMissing,
    UnexpectedGovernanceAbortDigest,
    SerializationFailed,
    SnapshotDigestMismatch,
    SnapshotChainBroken,
}

pub fn confirmatory_collection_snapshot_commitment(
    snapshot: &ConfirmatoryCollectionSnapshot,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&SnapshotCommitment {
        monitor_version: &snapshot.monitor_version,
        protocol_sha256: &snapshot.protocol_sha256,
        snapshot_sequence: snapshot.snapshot_sequence,
        recorded_at_utc: &snapshot.recorded_at_utc,
        previous_snapshot_sha256: &snapshot.previous_snapshot_sha256,
        sessions: &snapshot.sessions,
        enrolled_count: snapshot.enrolled_count,
        started_count: snapshot.started_count,
        included_complete_count: snapshot.included_complete_count,
        excluded_complete_count: snapshot.excluded_complete_count,
        withdrawn_count: snapshot.withdrawn_count,
        operational_failure_count: snapshot.operational_failure_count,
        exclusion_rate_basis_points: snapshot.exclusion_rate_basis_points,
        completion_rate_basis_points: snapshot.completion_rate_basis_points,
        integrity_incident_open: snapshot.integrity_incident_open,
        frozen_deadline_reached: snapshot.frozen_deadline_reached,
        governance_abort_order_sha256: &snapshot.governance_abort_order_sha256,
        decision: snapshot.decision,
    })
}

pub fn build_confirmatory_collection_snapshot(
    protocol: &ConfirmatoryCollectionProtocol,
    snapshot_sequence: u32,
    recorded_at_utc: String,
    previous_snapshot_sha256: String,
    mut sessions: Vec<ConfirmatorySessionStatus>,
    integrity_incident_open: bool,
    governance_abort_order_sha256: Option<String>,
    frozen_deadline_reached: bool,
) -> Result<ConfirmatoryCollectionSnapshot, Vec<ConfirmatoryCollectionMonitorIssue>> {
    if !validate_confirmatory_collection_protocol(protocol).is_empty() {
        return Err(vec![ConfirmatoryCollectionMonitorIssue::InvalidProtocol]);
    }
    sessions.sort_by(|left, right| {
        left.participant_token
            .cmp(&right.participant_token)
            .then_with(|| left.block_id.cmp(&right.block_id))
    });
    let counts = count_sessions(&sessions);
    let enrolled_count = sessions.len() as u32;
    let completed = counts.included_complete + counts.excluded_complete;
    let exclusion_rate_basis_points = rate_basis_points(counts.excluded_complete, completed);
    let completion_rate_basis_points = rate_basis_points(completed, enrolled_count);
    let decision = derive_decision(
        protocol,
        counts.included_complete,
        integrity_incident_open,
        governance_abort_order_sha256.is_some(),
        frozen_deadline_reached,
    );
    let mut snapshot = ConfirmatoryCollectionSnapshot {
        monitor_version: CONFIRMATORY_COLLECTION_MONITOR_VERSION.into(),
        protocol_sha256: protocol.protocol_sha256.clone(),
        snapshot_sequence,
        recorded_at_utc,
        previous_snapshot_sha256,
        sessions,
        enrolled_count,
        started_count: counts.started,
        included_complete_count: counts.included_complete,
        excluded_complete_count: counts.excluded_complete,
        withdrawn_count: counts.withdrawn,
        operational_failure_count: counts.operational_failure,
        exclusion_rate_basis_points,
        completion_rate_basis_points,
        integrity_incident_open,
        frozen_deadline_reached,
        governance_abort_order_sha256,
        decision,
        snapshot_sha256: String::new(),
    };
    snapshot.snapshot_sha256 = confirmatory_collection_snapshot_commitment(&snapshot)
        .map_err(|_| vec![ConfirmatoryCollectionMonitorIssue::SerializationFailed])?;
    let issues = validate_confirmatory_collection_snapshot(protocol, &snapshot, None);
    if issues.is_empty() {
        Ok(snapshot)
    } else {
        Err(issues)
    }
}

pub fn validate_confirmatory_collection_snapshot(
    protocol: &ConfirmatoryCollectionProtocol,
    snapshot: &ConfirmatoryCollectionSnapshot,
    previous: Option<&ConfirmatoryCollectionSnapshot>,
) -> Vec<ConfirmatoryCollectionMonitorIssue> {
    let mut issues = Vec::new();
    if !validate_confirmatory_collection_protocol(protocol).is_empty()
        || snapshot.protocol_sha256 != protocol.protocol_sha256
    {
        issues.push(ConfirmatoryCollectionMonitorIssue::InvalidProtocol);
    }
    if snapshot.monitor_version != CONFIRMATORY_COLLECTION_MONITOR_VERSION {
        issues.push(ConfirmatoryCollectionMonitorIssue::WrongVersion {
            found: snapshot.monitor_version.clone(),
        });
    }
    if snapshot.snapshot_sequence == 0 {
        issues.push(ConfirmatoryCollectionMonitorIssue::InvalidSequence);
    }
    if snapshot.recorded_at_utc.trim().is_empty() {
        issues.push(ConfirmatoryCollectionMonitorIssue::EmptyField {
            field: "recorded_at_utc".into(),
        });
    }
    if !is_sha256(&snapshot.previous_snapshot_sha256) {
        issues.push(ConfirmatoryCollectionMonitorIssue::InvalidDigest {
            field: "previous_snapshot_sha256".into(),
        });
    }
    if let Some(previous) = previous {
        if snapshot.snapshot_sequence != previous.snapshot_sequence + 1
            || snapshot.previous_snapshot_sha256 != previous.snapshot_sha256
        {
            issues.push(ConfirmatoryCollectionMonitorIssue::SnapshotChainBroken);
        }
    }
    let mut participants = BTreeSet::new();
    let mut blocks = BTreeSet::new();
    for session in &snapshot.sessions {
        if session.participant_token.trim().is_empty() {
            issues.push(ConfirmatoryCollectionMonitorIssue::EmptyField {
                field: "participant_token".into(),
            });
        }
        if session.block_id.trim().is_empty() {
            issues.push(ConfirmatoryCollectionMonitorIssue::EmptyField {
                field: "block_id".into(),
            });
        }
        if session.site_id.trim().is_empty() {
            issues.push(ConfirmatoryCollectionMonitorIssue::EmptyField {
                field: "site_id".into(),
            });
        }
        if !participants.insert(session.participant_token.as_str()) {
            issues.push(ConfirmatoryCollectionMonitorIssue::DuplicateParticipant {
                participant_token: session.participant_token.clone(),
            });
        }
        if !blocks.insert(session.block_id.as_str()) {
            issues.push(ConfirmatoryCollectionMonitorIssue::DuplicateBlock {
                block_id: session.block_id.clone(),
            });
        }
        if !is_sha256(&session.package_sha256) {
            issues.push(ConfirmatoryCollectionMonitorIssue::InvalidSessionDigest {
                block_id: session.block_id.clone(),
            });
        }
        let completed = matches!(
            session.disposition,
            ConfirmatorySessionDisposition::CompleteIncluded
                | ConfirmatorySessionDisposition::CompleteExcluded
        );
        match (&session.session_log_sha256, completed) {
            (Some(value), _) if !is_sha256(value) => {
                issues.push(ConfirmatoryCollectionMonitorIssue::InvalidSessionDigest {
                    block_id: session.block_id.clone(),
                });
            }
            (None, true) => {
                issues.push(
                    ConfirmatoryCollectionMonitorIssue::MissingCompletedSessionDigest {
                        block_id: session.block_id.clone(),
                    },
                );
            }
            (Some(_), false) => {
                issues.push(
                    ConfirmatoryCollectionMonitorIssue::UnexpectedCompletedSessionDigest {
                        block_id: session.block_id.clone(),
                    },
                );
            }
            _ => {}
        }
        let excluded = session.disposition == ConfirmatorySessionDisposition::CompleteExcluded;
        if excluded != session.preregistered_exclusion_code.is_some() {
            issues.push(ConfirmatoryCollectionMonitorIssue::ExclusionCodeMismatch {
                block_id: session.block_id.clone(),
            });
        }
    }
    let counts = count_sessions(&snapshot.sessions);
    for (field, expected, found) in [
        (
            "enrolled_count",
            snapshot.sessions.len() as u32,
            snapshot.enrolled_count,
        ),
        ("started_count", counts.started, snapshot.started_count),
        (
            "included_complete_count",
            counts.included_complete,
            snapshot.included_complete_count,
        ),
        (
            "excluded_complete_count",
            counts.excluded_complete,
            snapshot.excluded_complete_count,
        ),
        (
            "withdrawn_count",
            counts.withdrawn,
            snapshot.withdrawn_count,
        ),
        (
            "operational_failure_count",
            counts.operational_failure,
            snapshot.operational_failure_count,
        ),
    ] {
        if expected != found {
            issues.push(ConfirmatoryCollectionMonitorIssue::CountMismatch {
                field: field.into(),
                expected,
                found,
            });
        }
    }
    if snapshot.enrolled_count > protocol.maximum_enrolled_participants {
        issues.push(ConfirmatoryCollectionMonitorIssue::EnrollmentExceedsMaximum);
    }
    let completed = counts.included_complete + counts.excluded_complete;
    for (field, expected, found) in [
        (
            "exclusion_rate_basis_points",
            rate_basis_points(counts.excluded_complete, completed),
            snapshot.exclusion_rate_basis_points,
        ),
        (
            "completion_rate_basis_points",
            rate_basis_points(completed, snapshot.enrolled_count),
            snapshot.completion_rate_basis_points,
        ),
    ] {
        if expected != found {
            issues.push(ConfirmatoryCollectionMonitorIssue::RateMismatch {
                field: field.into(),
                expected,
                found,
            });
        }
    }
    if snapshot.governance_abort_order_sha256.is_some()
        && !snapshot
            .governance_abort_order_sha256
            .as_deref()
            .is_some_and(is_sha256)
    {
        issues.push(ConfirmatoryCollectionMonitorIssue::GovernanceAbortDigestMissing);
    }
    if snapshot.decision == ConfirmatoryCollectionDecision::AbortGovernance
        && snapshot.governance_abort_order_sha256.is_none()
    {
        issues.push(ConfirmatoryCollectionMonitorIssue::GovernanceAbortDigestMissing);
    }
    if snapshot.decision != ConfirmatoryCollectionDecision::AbortGovernance
        && snapshot.governance_abort_order_sha256.is_some()
    {
        issues.push(ConfirmatoryCollectionMonitorIssue::UnexpectedGovernanceAbortDigest);
    }
    let expected_decision = derive_decision(
        protocol,
        counts.included_complete,
        snapshot.integrity_incident_open,
        snapshot.governance_abort_order_sha256.is_some(),
        snapshot.frozen_deadline_reached,
    );
    if expected_decision != snapshot.decision {
        issues.push(ConfirmatoryCollectionMonitorIssue::DecisionMismatch {
            expected: expected_decision,
            found: snapshot.decision,
        });
    }
    match confirmatory_collection_snapshot_commitment(snapshot) {
        Ok(found) if found == snapshot.snapshot_sha256 => {}
        Ok(_) => issues.push(ConfirmatoryCollectionMonitorIssue::SnapshotDigestMismatch),
        Err(_) => issues.push(ConfirmatoryCollectionMonitorIssue::SerializationFailed),
    }
    issues
}

#[derive(Default)]
struct SessionCounts {
    started: u32,
    included_complete: u32,
    excluded_complete: u32,
    withdrawn: u32,
    operational_failure: u32,
}

fn count_sessions(sessions: &[ConfirmatorySessionStatus]) -> SessionCounts {
    let mut counts = SessionCounts::default();
    for session in sessions {
        match session.disposition {
            ConfirmatorySessionDisposition::Assigned => {}
            ConfirmatorySessionDisposition::Started => counts.started += 1,
            ConfirmatorySessionDisposition::CompleteIncluded => {
                counts.started += 1;
                counts.included_complete += 1;
            }
            ConfirmatorySessionDisposition::CompleteExcluded => {
                counts.started += 1;
                counts.excluded_complete += 1;
            }
            ConfirmatorySessionDisposition::Withdrawn => {
                counts.started += 1;
                counts.withdrawn += 1;
            }
            ConfirmatorySessionDisposition::OperationalFailure => {
                counts.started += 1;
                counts.operational_failure += 1;
            }
        }
    }
    counts
}

fn derive_decision(
    protocol: &ConfirmatoryCollectionProtocol,
    included_complete_count: u32,
    integrity_incident_open: bool,
    governance_abort: bool,
    frozen_deadline_reached: bool,
) -> ConfirmatoryCollectionDecision {
    if governance_abort {
        ConfirmatoryCollectionDecision::AbortGovernance
    } else if integrity_incident_open {
        ConfirmatoryCollectionDecision::PauseOperationalIntegrity
    } else if included_complete_count >= protocol.target_complete_blocks {
        ConfirmatoryCollectionDecision::CloseTargetReached
    } else if frozen_deadline_reached {
        ConfirmatoryCollectionDecision::CloseFrozenDeadline
    } else {
        ConfirmatoryCollectionDecision::Continue
    }
}

fn rate_basis_points(numerator: u32, denominator: u32) -> u32 {
    if denominator == 0 {
        0
    } else {
        ((u64::from(numerator) * 10_000) / u64::from(denominator)) as u32
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::confirmatory_collection_protocol::{
        CONFIRMATORY_COLLECTION_PROTOCOL_VERSION, ConfirmatoryDataRole,
        seal_confirmatory_collection_protocol,
    };

    fn protocol() -> ConfirmatoryCollectionProtocol {
        let mut value = ConfirmatoryCollectionProtocol {
            protocol_version: CONFIRMATORY_COLLECTION_PROTOCOL_VERSION.into(),
            study_id: "study".into(),
            readiness_release_sha256: "1".repeat(64),
            external_preregistration_sha256: "2".repeat(64),
            manifest_sha256: "3".repeat(64),
            methodology_sha256: "4".repeat(64),
            blinded_schedule_sha256: "5".repeat(64),
            participant_schedule_sha256: "6".repeat(64),
            artifact_bundle_sha256: "7".repeat(64),
            cohort_registry_sha256: "8".repeat(64),
            planned_open_utc: "open".into(),
            planned_close_utc: "close".into(),
            target_complete_blocks: 1,
            maximum_enrolled_participants: 4,
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
        seal_confirmatory_collection_protocol(&mut value).unwrap();
        value
    }

    #[test]
    fn target_completion_closes_without_outcomes() {
        let status = ConfirmatorySessionStatus {
            participant_token: "p1".into(),
            block_id: "b1".into(),
            site_id: "s1".into(),
            disposition: ConfirmatorySessionDisposition::CompleteIncluded,
            package_sha256: "a".repeat(64),
            session_log_sha256: Some("b".repeat(64)),
            preregistered_exclusion_code: None,
        };
        let snapshot = build_confirmatory_collection_snapshot(
            &protocol(),
            1,
            "now".into(),
            "0".repeat(64),
            vec![status],
            false,
            None,
            false,
        )
        .unwrap();
        assert_eq!(
            snapshot.decision,
            ConfirmatoryCollectionDecision::CloseTargetReached
        );
    }

    #[test]
    fn integrity_incident_pauses_before_target_rule() {
        let snapshot = build_confirmatory_collection_snapshot(
            &protocol(),
            1,
            "now".into(),
            "0".repeat(64),
            Vec::new(),
            true,
            None,
            false,
        )
        .unwrap();
        assert_eq!(
            snapshot.decision,
            ConfirmatoryCollectionDecision::PauseOperationalIntegrity
        );
    }
}
