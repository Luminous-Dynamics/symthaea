// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Blinded, outcome-neutral pilot monitoring.
//!
//! The monitor sees completion, attention, technical-failure, exclusion,
//! duration, and replay burden. It deliberately receives no arm labels, ranks,
//! ratings, recognition outcomes, or musical-policy summaries.

use crate::evidence_digest::canonical_json_sha256;
use crate::pilot_protocol::FrozenPilotProtocol;
use crate::study_evidence::PreregisteredExclusion;
use crate::study_runner::{
    StudyRunnerPackage, StudySessionEvent, StudySessionLog, validate_session_log,
};
use serde::{Deserialize, Serialize};

pub const PILOT_MONITORING_VERSION: &str = "symthaea-muse-pilot-monitoring-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PilotOperationalRecord {
    pub block_id: String,
    pub participant_token: String,
    pub finalized: bool,
    pub included: bool,
    pub attention_check_passed: bool,
    pub technical_failure: bool,
    pub excluded: bool,
    pub exclusion_reason: Option<PreregisteredExclusion>,
    pub session_duration_seconds: Option<u64>,
    pub replay_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PilotOperationalDecision {
    ContinueCurrentWave,
    OpenNextWave,
    PauseForTechnicalReview,
    ReadyToClosePilot,
    StopAtMaximumEnrollment,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PilotOperationalSnapshot {
    pub monitoring_version: String,
    pub pilot_protocol_sha256: String,
    pub observed_at_utc: String,
    pub enrolled_participants: usize,
    pub finalized_participants: usize,
    pub included_participants: usize,
    pub completion_rate: f64,
    pub attention_pass_rate: f64,
    pub technical_failure_rate: f64,
    pub exclusion_rate: f64,
    pub median_session_seconds: Option<u64>,
    pub total_replays: u64,
    pub decision: PilotOperationalDecision,
    pub snapshot_sha256: String,
}

#[derive(Serialize)]
struct SnapshotCommitment<'a> {
    monitoring_version: &'a str,
    pilot_protocol_sha256: &'a str,
    observed_at_utc: &'a str,
    enrolled_participants: usize,
    finalized_participants: usize,
    included_participants: usize,
    completion_rate: f64,
    attention_pass_rate: f64,
    technical_failure_rate: f64,
    exclusion_rate: f64,
    median_session_seconds: Option<u64>,
    total_replays: u64,
    decision: PilotOperationalDecision,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PilotMonitoringIssue {
    InvalidSession { block_id: String },
    PackageLogMismatch { block_id: String },
    DuplicateBlock { block_id: String },
    EmptyObservationTime,
    SerializationFailed { field: String },
    DigestMismatch { field: String },
    WrongVersion { found: String },
    InvalidRate { field: String },
    CountInconsistency,
}

pub fn compile_pilot_operational_record(
    package: &StudyRunnerPackage,
    log: &StudySessionLog,
) -> Result<PilotOperationalRecord, Vec<PilotMonitoringIssue>> {
    if package.block_id != log.block_id
        || package.participant_token != log.participant_token
        || package.package_sha256 != log.package_sha256
    {
        return Err(vec![PilotMonitoringIssue::PackageLogMismatch {
            block_id: log.block_id.clone(),
        }]);
    }
    if !validate_session_log(package, log, false).is_empty() {
        return Err(vec![PilotMonitoringIssue::InvalidSession {
            block_id: log.block_id.clone(),
        }]);
    }
    let finalized = log
        .events
        .iter()
        .any(|event| matches!(&event.event, StudySessionEvent::BlockFinalized));
    let exclusion_reason = log.events.iter().find_map(|event| match &event.event {
        StudySessionEvent::BlockExcluded { reason } => Some(reason.clone()),
        _ => None,
    });
    let excluded = exclusion_reason.is_some();
    let attention_responses: Vec<_> = log
        .events
        .iter()
        .filter_map(|event| match &event.event {
            StudySessionEvent::ResponseRecorded {
                attention_check_response,
                ..
            } => *attention_check_response,
            _ => None,
        })
        .collect();
    let attention_check_passed = if package.protocol.require_attention_check {
        !attention_responses.is_empty()
            && attention_responses
                .iter()
                .all(|response| *response == package.protocol.attention_check_expected_index)
    } else {
        true
    };
    let technical_failure = matches!(
        &exclusion_reason,
        Some(PreregisteredExclusion::TechnicalPlaybackFailure)
    );
    let replay_count = log
        .events
        .iter()
        .filter(|event| {
            matches!(
                &event.event,
                StudySessionEvent::PlaybackStarted { replay_index, .. } if *replay_index > 0
            )
        })
        .count() as u32;
    let session_duration_seconds = match (log.events.first(), log.events.last()) {
        (Some(first), Some(last))
            if last.server_received_unix_ms >= first.server_received_unix_ms =>
        {
            Some((last.server_received_unix_ms - first.server_received_unix_ms) / 1000)
        }
        _ => None,
    };
    Ok(PilotOperationalRecord {
        block_id: log.block_id.clone(),
        participant_token: log.participant_token.clone(),
        finalized,
        included: finalized && !excluded,
        attention_check_passed,
        technical_failure,
        excluded,
        exclusion_reason,
        session_duration_seconds,
        replay_count,
    })
}

pub fn pilot_operational_snapshot_commitment(
    snapshot: &PilotOperationalSnapshot,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&SnapshotCommitment {
        monitoring_version: &snapshot.monitoring_version,
        pilot_protocol_sha256: &snapshot.pilot_protocol_sha256,
        observed_at_utc: &snapshot.observed_at_utc,
        enrolled_participants: snapshot.enrolled_participants,
        finalized_participants: snapshot.finalized_participants,
        included_participants: snapshot.included_participants,
        completion_rate: snapshot.completion_rate,
        attention_pass_rate: snapshot.attention_pass_rate,
        technical_failure_rate: snapshot.technical_failure_rate,
        exclusion_rate: snapshot.exclusion_rate,
        median_session_seconds: snapshot.median_session_seconds,
        total_replays: snapshot.total_replays,
        decision: snapshot.decision,
    })
}

pub fn build_pilot_operational_snapshot(
    protocol: &FrozenPilotProtocol,
    observed_at_utc: String,
    records: &[PilotOperationalRecord],
) -> Result<PilotOperationalSnapshot, Vec<PilotMonitoringIssue>> {
    if observed_at_utc.trim().is_empty() {
        return Err(vec![PilotMonitoringIssue::EmptyObservationTime]);
    }
    let mut block_ids = std::collections::BTreeSet::new();
    for record in records {
        if !block_ids.insert(record.block_id.as_str()) {
            return Err(vec![PilotMonitoringIssue::DuplicateBlock {
                block_id: record.block_id.clone(),
            }]);
        }
    }
    let enrolled_participants = records.len();
    let finalized_participants = records.iter().filter(|record| record.finalized).count();
    let included_participants = records.iter().filter(|record| record.included).count();
    let excluded_count = records.iter().filter(|record| record.excluded).count();
    let attention_evaluable: Vec<_> = records.iter().filter(|record| record.finalized).collect();
    let technical_failures = records
        .iter()
        .filter(|record| record.technical_failure)
        .count();
    let completion_rate = rate(finalized_participants, enrolled_participants);
    let attention_pass_rate = rate(
        attention_evaluable
            .iter()
            .filter(|record| record.attention_check_passed)
            .count(),
        attention_evaluable.len(),
    );
    let technical_failure_rate = rate(technical_failures, enrolled_participants);
    let exclusion_rate = rate(excluded_count, finalized_participants);
    let mut durations: Vec<_> = records
        .iter()
        .filter_map(|record| record.session_duration_seconds)
        .collect();
    durations.sort_unstable();
    let median_session_seconds = median_u64(&durations);
    let total_replays = records
        .iter()
        .map(|record| u64::from(record.replay_count))
        .sum();
    let thresholds = &protocol.thresholds;
    let operational_failure = technical_failure_rate > thresholds.maximum_technical_failure_rate
        || (finalized_participants >= thresholds.cohort_wave_size
            && completion_rate < thresholds.minimum_completion_rate)
        || (finalized_participants >= thresholds.cohort_wave_size
            && attention_pass_rate < thresholds.minimum_attention_pass_rate)
        || (finalized_participants >= thresholds.cohort_wave_size
            && exclusion_rate > thresholds.maximum_exclusion_rate)
        || median_session_seconds
            .is_some_and(|value| value > thresholds.maximum_median_session_seconds);
    let ready = finalized_participants >= thresholds.minimum_completed_participants
        && completion_rate >= thresholds.minimum_completion_rate
        && attention_pass_rate >= thresholds.minimum_attention_pass_rate
        && technical_failure_rate <= thresholds.maximum_technical_failure_rate
        && exclusion_rate <= thresholds.maximum_exclusion_rate
        && median_session_seconds
            .is_none_or(|value| value <= thresholds.maximum_median_session_seconds);
    let decision = if operational_failure {
        PilotOperationalDecision::PauseForTechnicalReview
    } else if ready {
        PilotOperationalDecision::ReadyToClosePilot
    } else if enrolled_participants >= thresholds.maximum_enrolled_participants {
        PilotOperationalDecision::StopAtMaximumEnrollment
    } else if finalized_participants > 0
        && finalized_participants % thresholds.cohort_wave_size == 0
    {
        PilotOperationalDecision::OpenNextWave
    } else {
        PilotOperationalDecision::ContinueCurrentWave
    };
    let protocol_sha256 = canonical_json_sha256(protocol).map_err(|_| {
        vec![PilotMonitoringIssue::SerializationFailed {
            field: "pilot_protocol".into(),
        }]
    })?;
    let mut snapshot = PilotOperationalSnapshot {
        monitoring_version: PILOT_MONITORING_VERSION.into(),
        pilot_protocol_sha256: protocol_sha256,
        observed_at_utc,
        enrolled_participants,
        finalized_participants,
        included_participants,
        completion_rate,
        attention_pass_rate,
        technical_failure_rate,
        exclusion_rate,
        median_session_seconds,
        total_replays,
        decision,
        snapshot_sha256: String::new(),
    };
    snapshot.snapshot_sha256 = pilot_operational_snapshot_commitment(&snapshot).map_err(|_| {
        vec![PilotMonitoringIssue::SerializationFailed {
            field: "pilot_operational_snapshot".into(),
        }]
    })?;
    Ok(snapshot)
}

pub fn validate_pilot_operational_snapshot(
    protocol: &FrozenPilotProtocol,
    snapshot: &PilotOperationalSnapshot,
) -> Vec<PilotMonitoringIssue> {
    let mut issues = Vec::new();
    if snapshot.monitoring_version != PILOT_MONITORING_VERSION {
        issues.push(PilotMonitoringIssue::WrongVersion {
            found: snapshot.monitoring_version.clone(),
        });
    }
    match canonical_json_sha256(protocol) {
        Ok(value) if value == snapshot.pilot_protocol_sha256 => {}
        Ok(_) => issues.push(PilotMonitoringIssue::DigestMismatch {
            field: "pilot_protocol_sha256".into(),
        }),
        Err(_) => issues.push(PilotMonitoringIssue::SerializationFailed {
            field: "pilot_protocol".into(),
        }),
    }
    for (field, value) in [
        ("completion_rate", snapshot.completion_rate),
        ("attention_pass_rate", snapshot.attention_pass_rate),
        ("technical_failure_rate", snapshot.technical_failure_rate),
        ("exclusion_rate", snapshot.exclusion_rate),
    ] {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            issues.push(PilotMonitoringIssue::InvalidRate {
                field: field.into(),
            });
        }
    }
    if snapshot.included_participants > snapshot.finalized_participants
        || snapshot.finalized_participants > snapshot.enrolled_participants
    {
        issues.push(PilotMonitoringIssue::CountInconsistency);
    }
    match pilot_operational_snapshot_commitment(snapshot) {
        Ok(value) if value == snapshot.snapshot_sha256 => {}
        Ok(_) => issues.push(PilotMonitoringIssue::DigestMismatch {
            field: "snapshot_sha256".into(),
        }),
        Err(_) => issues.push(PilotMonitoringIssue::SerializationFailed {
            field: "pilot_operational_snapshot".into(),
        }),
    }
    issues
}

fn rate(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn median_u64(values: &[u64]) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    let middle = values.len() / 2;
    if values.len() % 2 == 1 {
        Some(values[middle])
    } else {
        Some(values[middle - 1].saturating_add(values[middle]) / 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monitoring_record_has_no_arm_or_outcome_fields() {
        let record = PilotOperationalRecord {
            block_id: "b".into(),
            participant_token: "p".into(),
            finalized: true,
            included: true,
            attention_check_passed: true,
            technical_failure: false,
            excluded: false,
            exclusion_reason: None,
            session_duration_seconds: Some(120),
            replay_count: 0,
        };
        let json = serde_json::to_string(&record).unwrap();
        for forbidden in ["arm", "rank", "preference", "recognized", "recapitulation"] {
            assert!(!json.contains(forbidden));
        }
    }

    #[test]
    fn empty_rates_are_zero_not_nan() {
        assert_eq!(rate(0, 0), 0.0);
    }
}
