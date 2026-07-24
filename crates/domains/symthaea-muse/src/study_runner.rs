// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Participant-facing study packages and hash-chained session evidence.
//!
//! The runner package contains anonymous presentation identifiers only. The
//! state machine enforces prospective order, playback completion, response
//! ranges, complete rankings, and finalization before a listener block can be
//! compiled into V8 evidence.

use crate::blinded_study::BlindedSchedule;
use crate::cognitive_experiment::FrozenTrialKey;
use crate::evidence_digest::{canonical_json_sha256, sha256_hex};
use crate::participant_schedule::{ParticipantBlockAssignment, ParticipantScheduleBook};
use crate::pilot_schedule::PilotParticipantScheduleBook;
use crate::study_artifact::StudyArtifactBundle;
use crate::study_evidence::{
    EvidenceBlockStatus, ListenerPresentationResponse, ListenerResponseBlock,
    PreregisteredExclusion,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const STUDY_RUNNER_PACKAGE_VERSION: &str = "symthaea-muse-study-runner-package-v1";
pub const STUDY_SESSION_LOG_VERSION: &str = "symthaea-muse-study-session-log-v1";
const GENESIS_EVENT_DIGEST: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunnerProtocol {
    pub consent_document_text: String,
    pub consent_document_sha256: String,
    pub instructions_text: String,
    pub instructions_sha256: String,
    pub minimum_playback_fraction: f64,
    pub maximum_replays_per_presentation: u8,
    pub require_attention_check: bool,
    pub attention_check_prompt: String,
    pub attention_check_options: Vec<String>,
    pub attention_check_expected_index: u8,
    pub prevent_backtracking: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunnerPresentation {
    pub presentation_id: String,
    pub anonymous_code: String,
    pub display_position: u8,
    pub audio_relative_path: String,
    pub audio_sha256: String,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StudyRunnerPackage {
    pub package_version: String,
    pub participant_schedule_sha256: String,
    pub artifact_bundle_sha256: String,
    pub block_id: String,
    pub participant_token: String,
    pub key: FrozenTrialKey,
    pub protocol: RunnerProtocol,
    pub presentations: Vec<RunnerPresentation>,
    pub package_sha256: String,
}

#[derive(Serialize)]
struct RunnerPackageCommitment<'a> {
    package_version: &'a str,
    participant_schedule_sha256: &'a str,
    artifact_bundle_sha256: &'a str,
    block_id: &'a str,
    participant_token: &'a str,
    key: &'a FrozenTrialKey,
    protocol: &'a RunnerProtocol,
    presentations: &'a [RunnerPresentation],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StudySessionEvent {
    ConsentAccepted {
        consent_document_sha256: String,
    },
    InstructionsAcknowledged {
        instructions_sha256: String,
    },
    PlaybackStarted {
        presentation_id: String,
        replay_index: u8,
    },
    PlaybackCompleted {
        presentation_id: String,
        replay_index: u8,
        listened_ms: u64,
        media_duration_ms: u64,
    },
    ResponseRecorded {
        presentation_id: String,
        return_recognized: bool,
        development_instability: f32,
        earned_recapitulation: f32,
        attention_check_response: Option<u8>,
        elapsed_ms: u64,
    },
    RankingsSubmitted {
        presentation_ids_best_to_worst: Vec<String>,
    },
    BlockExcluded {
        reason: PreregisteredExclusion,
    },
    BlockFinalized,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StudySessionEventEnvelope {
    pub sequence: u64,
    pub previous_event_sha256: String,
    pub server_received_unix_ms: u64,
    pub client_elapsed_ms: u64,
    pub event: StudySessionEvent,
    pub event_sha256: String,
}

#[derive(Serialize)]
struct SessionEventCommitment<'a> {
    package_sha256: &'a str,
    sequence: u64,
    previous_event_sha256: &'a str,
    server_received_unix_ms: u64,
    client_elapsed_ms: u64,
    event: &'a StudySessionEvent,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StudySessionLog {
    pub log_version: String,
    pub package_sha256: String,
    pub block_id: String,
    pub participant_token: String,
    pub events: Vec<StudySessionEventEnvelope>,
    pub log_sha256: String,
}

#[derive(Serialize)]
struct SessionLogCommitment<'a> {
    log_version: &'a str,
    package_sha256: &'a str,
    block_id: &'a str,
    participant_token: &'a str,
    events: &'a [StudySessionEventEnvelope],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StudyRunnerIssue {
    WrongPackageVersion {
        found: String,
    },
    WrongLogVersion {
        found: String,
    },
    InvalidProtocol {
        field: String,
    },
    DigestMismatch {
        field: String,
    },
    SerializationFailed {
        field: String,
    },
    UnknownBlock {
        block_id: String,
    },
    DuplicatePresentation {
        presentation_id: String,
    },
    MissingPresentation {
        presentation_id: String,
    },
    UnknownPresentation {
        presentation_id: String,
    },
    ArtifactDigestMismatch {
        presentation_id: String,
    },
    PackageIdentityMismatch {
        field: String,
    },
    SequenceMismatch {
        found: u64,
        expected: u64,
    },
    PreviousEventMismatch {
        sequence: u64,
    },
    EventDigestMismatch {
        sequence: u64,
    },
    NonMonotonicServerTime {
        sequence: u64,
    },
    NonMonotonicClientTime {
        sequence: u64,
    },
    ConsentRequired,
    ConsentDigestMismatch,
    InstructionsRequired,
    InstructionsDigestMismatch,
    UnexpectedEventAfterFinalization,
    PresentationOrderViolation {
        presentation_id: String,
    },
    ReplayLimitExceeded {
        presentation_id: String,
    },
    DuplicateReplay {
        presentation_id: String,
        replay_index: u8,
    },
    PlaybackNotStarted {
        presentation_id: String,
    },
    PlaybackIncomplete {
        presentation_id: String,
    },
    MediaDurationMismatch {
        presentation_id: String,
    },
    ResponseBeforePlayback {
        presentation_id: String,
    },
    DuplicateResponse {
        presentation_id: String,
    },
    InvalidResponse {
        presentation_id: String,
        field: String,
    },
    IncompleteResponses,
    InvalidRanking,
    MissingRanking,
    FinalizationRequired,
    ExclusionConflict,
    FailedAttentionCheckNotExcluded,
}

pub fn runner_package_commitment(
    package: &StudyRunnerPackage,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&RunnerPackageCommitment {
        package_version: &package.package_version,
        participant_schedule_sha256: &package.participant_schedule_sha256,
        artifact_bundle_sha256: &package.artifact_bundle_sha256,
        block_id: &package.block_id,
        participant_token: &package.participant_token,
        key: &package.key,
        protocol: &package.protocol,
        presentations: &package.presentations,
    })
}

pub fn session_event_commitment(
    package_sha256: &str,
    envelope: &StudySessionEventEnvelope,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&SessionEventCommitment {
        package_sha256,
        sequence: envelope.sequence,
        previous_event_sha256: &envelope.previous_event_sha256,
        server_received_unix_ms: envelope.server_received_unix_ms,
        client_elapsed_ms: envelope.client_elapsed_ms,
        event: &envelope.event,
    })
}

pub fn session_log_commitment(log: &StudySessionLog) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&SessionLogCommitment {
        log_version: &log.log_version,
        package_sha256: &log.package_sha256,
        block_id: &log.block_id,
        participant_token: &log.participant_token,
        events: &log.events,
    })
}

pub fn build_runner_package(
    schedule: &BlindedSchedule,
    participant_schedule: &ParticipantScheduleBook,
    artifacts: &StudyArtifactBundle,
    block_id: &str,
    protocol: RunnerProtocol,
) -> Result<StudyRunnerPackage, Vec<StudyRunnerIssue>> {
    let Some(block) = participant_schedule
        .blocks
        .iter()
        .find(|candidate| candidate.block_id == block_id)
    else {
        return Err(vec![StudyRunnerIssue::UnknownBlock {
            block_id: block_id.into(),
        }]);
    };
    let participant_schedule_sha256 =
        canonical_json_sha256(participant_schedule).map_err(|_| {
            vec![StudyRunnerIssue::SerializationFailed {
                field: "participant_schedule".into(),
            }]
        })?;
    build_runner_package_from_assignment(
        schedule,
        &participant_schedule.base_schedule_sha256,
        participant_schedule_sha256,
        artifacts,
        block,
        protocol,
    )
}

pub fn build_pilot_runner_package(
    schedule: &BlindedSchedule,
    pilot_schedule: &PilotParticipantScheduleBook,
    artifacts: &StudyArtifactBundle,
    block_id: &str,
    protocol: RunnerProtocol,
) -> Result<StudyRunnerPackage, Vec<StudyRunnerIssue>> {
    let Some(block) = pilot_schedule
        .blocks
        .iter()
        .find(|candidate| candidate.block_id == block_id)
    else {
        return Err(vec![StudyRunnerIssue::UnknownBlock {
            block_id: block_id.into(),
        }]);
    };
    let participant_schedule_sha256 = canonical_json_sha256(pilot_schedule).map_err(|_| {
        vec![StudyRunnerIssue::SerializationFailed {
            field: "pilot_participant_schedule".into(),
        }]
    })?;
    build_runner_package_from_assignment(
        schedule,
        &pilot_schedule.base_schedule_sha256,
        participant_schedule_sha256,
        artifacts,
        block,
        protocol,
    )
}

fn build_runner_package_from_assignment(
    schedule: &BlindedSchedule,
    assignment_base_schedule_sha256: &str,
    participant_schedule_sha256: String,
    artifacts: &StudyArtifactBundle,
    block: &ParticipantBlockAssignment,
    protocol: RunnerProtocol,
) -> Result<StudyRunnerPackage, Vec<StudyRunnerIssue>> {
    let mut issues = validate_protocol(&protocol);
    match canonical_json_sha256(schedule) {
        Ok(schedule_sha256) => {
            if assignment_base_schedule_sha256 != schedule_sha256 {
                issues.push(StudyRunnerIssue::DigestMismatch {
                    field: "participant_base_schedule_sha256".into(),
                });
            }
            if artifacts.schedule_sha256 != schedule_sha256 {
                issues.push(StudyRunnerIssue::DigestMismatch {
                    field: "artifact_schedule_sha256".into(),
                });
            }
        }
        Err(_) => issues.push(StudyRunnerIssue::SerializationFailed {
            field: "schedule".into(),
        }),
    }
    let schedule_by_id: BTreeMap<_, _> = schedule
        .presentations
        .iter()
        .map(|presentation| (presentation.presentation_id.as_str(), presentation))
        .collect();
    let artifacts_by_id: BTreeMap<_, _> = artifacts
        .records
        .iter()
        .map(|record| (record.presentation_id.as_str(), record))
        .collect();
    let mut seen = BTreeSet::new();
    let mut presentations = Vec::with_capacity(block.ordered_presentation_ids.len());
    for (position, presentation_id) in block.ordered_presentation_ids.iter().enumerate() {
        if !seen.insert(presentation_id.as_str()) {
            issues.push(StudyRunnerIssue::DuplicatePresentation {
                presentation_id: presentation_id.clone(),
            });
            continue;
        }
        let Some(schedule_entry) = schedule_by_id.get(presentation_id.as_str()) else {
            issues.push(StudyRunnerIssue::UnknownPresentation {
                presentation_id: presentation_id.clone(),
            });
            continue;
        };
        let Some(artifact) = artifacts_by_id.get(presentation_id.as_str()) else {
            issues.push(StudyRunnerIssue::MissingPresentation {
                presentation_id: presentation_id.clone(),
            });
            continue;
        };
        if schedule_entry.audio_sha256 != artifact.audio.sha256 {
            issues.push(StudyRunnerIssue::ArtifactDigestMismatch {
                presentation_id: presentation_id.clone(),
            });
        }
        presentations.push(RunnerPresentation {
            presentation_id: presentation_id.clone(),
            anonymous_code: schedule_entry.anonymous_code.clone(),
            display_position: position as u8,
            audio_relative_path: artifact.audio.relative_path.clone(),
            audio_sha256: artifact.audio.sha256.clone(),
            duration_ms: artifact.wav_audit.duration_ms,
        });
    }
    if !issues.is_empty() {
        return Err(issues);
    }
    let mut package = StudyRunnerPackage {
        package_version: STUDY_RUNNER_PACKAGE_VERSION.into(),
        participant_schedule_sha256,
        artifact_bundle_sha256: artifacts.bundle_sha256.clone(),
        block_id: block.block_id.clone(),
        participant_token: block.participant_token.clone(),
        key: block.key.clone(),
        protocol,
        presentations,
        package_sha256: String::new(),
    };
    package.package_sha256 = runner_package_commitment(&package).map_err(|_| {
        vec![StudyRunnerIssue::SerializationFailed {
            field: "runner_package".into(),
        }]
    })?;
    Ok(package)
}

pub fn validate_runner_package(
    package: &StudyRunnerPackage,
    schedule: &BlindedSchedule,
    participant_schedule: &ParticipantScheduleBook,
    artifacts: &StudyArtifactBundle,
) -> Vec<StudyRunnerIssue> {
    if package.package_version != STUDY_RUNNER_PACKAGE_VERSION {
        return vec![StudyRunnerIssue::WrongPackageVersion {
            found: package.package_version.clone(),
        }];
    }
    let mut issues = validate_protocol(&package.protocol);
    match build_runner_package(
        schedule,
        participant_schedule,
        artifacts,
        &package.block_id,
        package.protocol.clone(),
    ) {
        Ok(rebuilt) if rebuilt == *package => {}
        Ok(_) => issues.push(StudyRunnerIssue::PackageIdentityMismatch {
            field: "rebuilt_package".into(),
        }),
        Err(found) => issues.extend(found),
    }
    let assignment = participant_schedule
        .blocks
        .iter()
        .find(|block| block.block_id == package.block_id);
    let Some(assignment) = assignment else {
        issues.push(StudyRunnerIssue::UnknownBlock {
            block_id: package.block_id.clone(),
        });
        return issues;
    };
    if package.participant_token != assignment.participant_token {
        issues.push(StudyRunnerIssue::PackageIdentityMismatch {
            field: "participant_token".into(),
        });
    }
    if package.key != assignment.key {
        issues.push(StudyRunnerIssue::PackageIdentityMismatch {
            field: "key".into(),
        });
    }
    for (position, presentation) in package.presentations.iter().enumerate() {
        if presentation.display_position != position as u8 {
            issues.push(StudyRunnerIssue::PackageIdentityMismatch {
                field: format!("display_position:{}", presentation.presentation_id),
            });
        }
    }
    let ids: Vec<_> = package
        .presentations
        .iter()
        .map(|presentation| presentation.presentation_id.as_str())
        .collect();
    let assigned: Vec<_> = assignment
        .ordered_presentation_ids
        .iter()
        .map(String::as_str)
        .collect();
    if ids != assigned {
        issues.push(StudyRunnerIssue::PackageIdentityMismatch {
            field: "presentation_order".into(),
        });
    }
    let schedule_map: BTreeMap<_, _> = schedule
        .presentations
        .iter()
        .map(|presentation| (presentation.presentation_id.as_str(), presentation))
        .collect();
    for presentation in &package.presentations {
        match schedule_map.get(presentation.presentation_id.as_str()) {
            Some(entry)
                if entry.anonymous_code == presentation.anonymous_code
                    && entry.audio_sha256 == presentation.audio_sha256 => {}
            Some(_) => issues.push(StudyRunnerIssue::PackageIdentityMismatch {
                field: format!("schedule_presentation:{}", presentation.presentation_id),
            }),
            None => issues.push(StudyRunnerIssue::UnknownPresentation {
                presentation_id: presentation.presentation_id.clone(),
            }),
        }
    }
    match canonical_json_sha256(schedule) {
        Ok(value) => {
            if value != artifacts.schedule_sha256 {
                issues.push(StudyRunnerIssue::DigestMismatch {
                    field: "artifact_schedule_sha256".into(),
                });
            }
            if value != participant_schedule.base_schedule_sha256 {
                issues.push(StudyRunnerIssue::DigestMismatch {
                    field: "participant_base_schedule_sha256".into(),
                });
            }
        }
        Err(_) => issues.push(StudyRunnerIssue::SerializationFailed {
            field: "schedule".into(),
        }),
    }
    let artifact_map: BTreeMap<_, _> = artifacts
        .records
        .iter()
        .map(|record| (record.presentation_id.as_str(), record))
        .collect();
    for presentation in &package.presentations {
        match artifact_map.get(presentation.presentation_id.as_str()) {
            Some(record)
                if record.audio.sha256 == presentation.audio_sha256
                    && record.audio.relative_path == presentation.audio_relative_path
                    && record.wav_audit.duration_ms == presentation.duration_ms => {}
            Some(_) => issues.push(StudyRunnerIssue::ArtifactDigestMismatch {
                presentation_id: presentation.presentation_id.clone(),
            }),
            None => issues.push(StudyRunnerIssue::MissingPresentation {
                presentation_id: presentation.presentation_id.clone(),
            }),
        }
    }
    match canonical_json_sha256(participant_schedule) {
        Ok(value) if value == package.participant_schedule_sha256 => {}
        Ok(_) => issues.push(StudyRunnerIssue::DigestMismatch {
            field: "participant_schedule_sha256".into(),
        }),
        Err(_) => issues.push(StudyRunnerIssue::SerializationFailed {
            field: "participant_schedule".into(),
        }),
    }
    if package.artifact_bundle_sha256 != artifacts.bundle_sha256 {
        issues.push(StudyRunnerIssue::DigestMismatch {
            field: "artifact_bundle_sha256".into(),
        });
    }
    match runner_package_commitment(package) {
        Ok(value) if value == package.package_sha256 => {}
        Ok(_) => issues.push(StudyRunnerIssue::DigestMismatch {
            field: "package_sha256".into(),
        }),
        Err(_) => issues.push(StudyRunnerIssue::SerializationFailed {
            field: "runner_package".into(),
        }),
    }
    issues
}

pub fn validate_pilot_runner_package(
    package: &StudyRunnerPackage,
    schedule: &BlindedSchedule,
    pilot_schedule: &PilotParticipantScheduleBook,
    artifacts: &StudyArtifactBundle,
) -> Vec<StudyRunnerIssue> {
    if package.package_version != STUDY_RUNNER_PACKAGE_VERSION {
        return vec![StudyRunnerIssue::WrongPackageVersion {
            found: package.package_version.clone(),
        }];
    }
    let mut issues = validate_protocol(&package.protocol);
    match build_pilot_runner_package(
        schedule,
        pilot_schedule,
        artifacts,
        &package.block_id,
        package.protocol.clone(),
    ) {
        Ok(rebuilt) if rebuilt == *package => {}
        Ok(_) => issues.push(StudyRunnerIssue::PackageIdentityMismatch {
            field: "rebuilt_pilot_package".into(),
        }),
        Err(found) => issues.extend(found),
    }
    let Some(assignment) = pilot_schedule
        .blocks
        .iter()
        .find(|block| block.block_id == package.block_id)
    else {
        issues.push(StudyRunnerIssue::UnknownBlock {
            block_id: package.block_id.clone(),
        });
        return issues;
    };
    if package.participant_token != assignment.participant_token {
        issues.push(StudyRunnerIssue::PackageIdentityMismatch {
            field: "participant_token".into(),
        });
    }
    if package.key != assignment.key {
        issues.push(StudyRunnerIssue::PackageIdentityMismatch {
            field: "key".into(),
        });
    }
    let ids: Vec<_> = package
        .presentations
        .iter()
        .map(|presentation| presentation.presentation_id.as_str())
        .collect();
    let assigned: Vec<_> = assignment
        .ordered_presentation_ids
        .iter()
        .map(String::as_str)
        .collect();
    if ids != assigned {
        issues.push(StudyRunnerIssue::PackageIdentityMismatch {
            field: "presentation_order".into(),
        });
    }
    match canonical_json_sha256(pilot_schedule) {
        Ok(value) if value == package.participant_schedule_sha256 => {}
        Ok(_) => issues.push(StudyRunnerIssue::DigestMismatch {
            field: "pilot_participant_schedule_sha256".into(),
        }),
        Err(_) => issues.push(StudyRunnerIssue::SerializationFailed {
            field: "pilot_participant_schedule".into(),
        }),
    }
    if package.artifact_bundle_sha256 != artifacts.bundle_sha256 {
        issues.push(StudyRunnerIssue::DigestMismatch {
            field: "artifact_bundle_sha256".into(),
        });
    }
    match runner_package_commitment(package) {
        Ok(value) if value == package.package_sha256 => {}
        Ok(_) => issues.push(StudyRunnerIssue::DigestMismatch {
            field: "package_sha256".into(),
        }),
        Err(_) => issues.push(StudyRunnerIssue::SerializationFailed {
            field: "runner_package".into(),
        }),
    }
    issues
}

pub fn new_session_log(package: &StudyRunnerPackage) -> StudySessionLog {
    StudySessionLog {
        log_version: STUDY_SESSION_LOG_VERSION.into(),
        package_sha256: package.package_sha256.clone(),
        block_id: package.block_id.clone(),
        participant_token: package.participant_token.clone(),
        events: Vec::new(),
        log_sha256: String::new(),
    }
}

pub fn append_session_event(
    package: &StudyRunnerPackage,
    log: &mut StudySessionLog,
    server_received_unix_ms: u64,
    client_elapsed_ms: u64,
    event: StudySessionEvent,
) -> Result<(), Vec<StudyRunnerIssue>> {
    let mut trial = log.clone();
    let previous_event_sha256 = trial.events.last().map_or_else(
        || GENESIS_EVENT_DIGEST.into(),
        |entry| entry.event_sha256.clone(),
    );
    let mut envelope = StudySessionEventEnvelope {
        sequence: trial.events.len() as u64,
        previous_event_sha256,
        server_received_unix_ms,
        client_elapsed_ms,
        event,
        event_sha256: String::new(),
    };
    envelope.event_sha256 =
        session_event_commitment(&package.package_sha256, &envelope).map_err(|_| {
            vec![StudyRunnerIssue::SerializationFailed {
                field: "session_event".into(),
            }]
        })?;
    trial.events.push(envelope);
    trial.log_sha256 = session_log_commitment(&trial).map_err(|_| {
        vec![StudyRunnerIssue::SerializationFailed {
            field: "session_log".into(),
        }]
    })?;
    let issues = validate_session_log(package, &trial, false);
    if issues.is_empty() {
        *log = trial;
        Ok(())
    } else {
        Err(issues)
    }
}

pub fn validate_session_log(
    package: &StudyRunnerPackage,
    log: &StudySessionLog,
    require_finalized: bool,
) -> Vec<StudyRunnerIssue> {
    let mut issues = Vec::new();
    if log.log_version != STUDY_SESSION_LOG_VERSION {
        issues.push(StudyRunnerIssue::WrongLogVersion {
            found: log.log_version.clone(),
        });
    }
    for (field, left, right) in [
        (
            "package_sha256",
            log.package_sha256.as_str(),
            package.package_sha256.as_str(),
        ),
        ("block_id", log.block_id.as_str(), package.block_id.as_str()),
        (
            "participant_token",
            log.participant_token.as_str(),
            package.participant_token.as_str(),
        ),
    ] {
        if left != right {
            issues.push(StudyRunnerIssue::PackageIdentityMismatch {
                field: field.into(),
            });
        }
    }
    validate_event_chain(package, log, &mut issues);
    validate_protocol_state(package, log, require_finalized, &mut issues);
    match session_log_commitment(log) {
        Ok(value) if value == log.log_sha256 => {}
        Ok(_) if !log.events.is_empty() => issues.push(StudyRunnerIssue::DigestMismatch {
            field: "log_sha256".into(),
        }),
        Ok(_) => {}
        Err(_) => issues.push(StudyRunnerIssue::SerializationFailed {
            field: "session_log".into(),
        }),
    }
    issues
}

pub fn compile_listener_block(
    package: &StudyRunnerPackage,
    log: &StudySessionLog,
) -> Result<ListenerResponseBlock, Vec<StudyRunnerIssue>> {
    let issues = validate_session_log(package, log, true);
    if !issues.is_empty() {
        return Err(issues);
    }
    let exclusion = log.events.iter().find_map(|entry| match &entry.event {
        StudySessionEvent::BlockExcluded { reason } => Some(reason.clone()),
        _ => None,
    });
    let rankings: BTreeMap<_, _> = log
        .events
        .iter()
        .find_map(|entry| match &entry.event {
            StudySessionEvent::RankingsSubmitted {
                presentation_ids_best_to_worst,
            } => Some(
                presentation_ids_best_to_worst
                    .iter()
                    .enumerate()
                    .map(|(index, id)| (id.as_str(), (index + 1) as u8))
                    .collect(),
            ),
            _ => None,
        })
        .unwrap_or_default();
    let responses = log
        .events
        .iter()
        .filter_map(|entry| match &entry.event {
            StudySessionEvent::ResponseRecorded {
                presentation_id,
                return_recognized,
                development_instability,
                earned_recapitulation,
                attention_check_response,
                elapsed_ms,
            } => Some(ListenerPresentationResponse {
                presentation_id: presentation_id.clone(),
                return_recognized: *return_recognized,
                development_instability: *development_instability,
                earned_recapitulation: *earned_recapitulation,
                preference_rank: *rankings.get(presentation_id.as_str()).unwrap_or(&0),
                playback_completed: true,
                attention_check_passed: attention_check_response
                    .is_some_and(|value| value == package.protocol.attention_check_expected_index),
                elapsed_ms: *elapsed_ms,
            }),
            _ => None,
        })
        .collect();
    Ok(ListenerResponseBlock {
        block_id: package.block_id.clone(),
        listener_id: package.participant_token.clone(),
        key: package.key.clone(),
        status: exclusion.map_or(EvidenceBlockStatus::Included, |reason| {
            EvidenceBlockStatus::Excluded { reason }
        }),
        responses,
    })
}

fn validate_event_chain(
    package: &StudyRunnerPackage,
    log: &StudySessionLog,
    issues: &mut Vec<StudyRunnerIssue>,
) {
    let mut previous_digest = GENESIS_EVENT_DIGEST;
    let mut previous_server = 0u64;
    let mut previous_client = 0u64;
    for (index, envelope) in log.events.iter().enumerate() {
        if envelope.sequence != index as u64 {
            issues.push(StudyRunnerIssue::SequenceMismatch {
                found: envelope.sequence,
                expected: index as u64,
            });
        }
        if envelope.previous_event_sha256 != previous_digest {
            issues.push(StudyRunnerIssue::PreviousEventMismatch {
                sequence: envelope.sequence,
            });
        }
        match session_event_commitment(&package.package_sha256, envelope) {
            Ok(value) if value == envelope.event_sha256 => {}
            _ => issues.push(StudyRunnerIssue::EventDigestMismatch {
                sequence: envelope.sequence,
            }),
        }
        if index > 0 && envelope.server_received_unix_ms < previous_server {
            issues.push(StudyRunnerIssue::NonMonotonicServerTime {
                sequence: envelope.sequence,
            });
        }
        if index > 0 && envelope.client_elapsed_ms < previous_client {
            issues.push(StudyRunnerIssue::NonMonotonicClientTime {
                sequence: envelope.sequence,
            });
        }
        previous_digest = &envelope.event_sha256;
        previous_server = envelope.server_received_unix_ms;
        previous_client = envelope.client_elapsed_ms;
    }
}

#[derive(Default)]
struct ProtocolState {
    consented: bool,
    instructions_acknowledged: bool,
    finalized: bool,
    excluded: bool,
    exclusion_reason: Option<PreregisteredExclusion>,
    current_index: usize,
    started_replays: BTreeMap<String, BTreeSet<u8>>,
    started_server_ms: BTreeMap<(String, u8), u64>,
    completed_replays: BTreeMap<String, BTreeSet<u8>>,
    responses: BTreeSet<String>,
    ranking_submitted: bool,
    attention_failed: bool,
}

fn validate_protocol_state(
    package: &StudyRunnerPackage,
    log: &StudySessionLog,
    require_finalized: bool,
    issues: &mut Vec<StudyRunnerIssue>,
) {
    let assigned: Vec<_> = package
        .presentations
        .iter()
        .map(|presentation| presentation.presentation_id.as_str())
        .collect();
    let presentation_map: BTreeMap<_, _> = package
        .presentations
        .iter()
        .map(|presentation| (presentation.presentation_id.as_str(), presentation))
        .collect();
    let mut state = ProtocolState::default();
    for envelope in &log.events {
        if state.finalized {
            issues.push(StudyRunnerIssue::UnexpectedEventAfterFinalization);
            break;
        }
        match &envelope.event {
            StudySessionEvent::ConsentAccepted {
                consent_document_sha256,
            } => {
                if state.consented
                    || consent_document_sha256 != &package.protocol.consent_document_sha256
                {
                    issues.push(StudyRunnerIssue::ConsentDigestMismatch);
                }
                state.consented = true;
            }
            event if !state.consented => {
                issues.push(StudyRunnerIssue::ConsentRequired);
                if matches!(event, StudySessionEvent::BlockFinalized) {
                    state.finalized = true;
                }
            }
            event if state.excluded && !matches!(event, StudySessionEvent::BlockFinalized) => {
                issues.push(StudyRunnerIssue::ExclusionConflict);
            }
            StudySessionEvent::InstructionsAcknowledged {
                instructions_sha256,
            } => {
                if state.instructions_acknowledged
                    || instructions_sha256 != &package.protocol.instructions_sha256
                {
                    issues.push(StudyRunnerIssue::InstructionsDigestMismatch);
                }
                state.instructions_acknowledged = true;
            }
            StudySessionEvent::PlaybackStarted {
                presentation_id,
                replay_index,
            } if !state.instructions_acknowledged => {
                let _ = replay_index;
                issues.push(StudyRunnerIssue::InstructionsRequired);
                issues.push(StudyRunnerIssue::PresentationOrderViolation {
                    presentation_id: presentation_id.clone(),
                });
            }
            StudySessionEvent::PlaybackStarted {
                presentation_id,
                replay_index,
            } => {
                let expected = assigned
                    .get(state.current_index)
                    .copied()
                    .unwrap_or_default();
                if package.protocol.prevent_backtracking && presentation_id != expected {
                    issues.push(StudyRunnerIssue::PresentationOrderViolation {
                        presentation_id: presentation_id.clone(),
                    });
                }
                if *replay_index > package.protocol.maximum_replays_per_presentation {
                    issues.push(StudyRunnerIssue::ReplayLimitExceeded {
                        presentation_id: presentation_id.clone(),
                    });
                }
                if !state
                    .started_replays
                    .entry(presentation_id.clone())
                    .or_default()
                    .insert(*replay_index)
                {
                    issues.push(StudyRunnerIssue::DuplicateReplay {
                        presentation_id: presentation_id.clone(),
                        replay_index: *replay_index,
                    });
                } else {
                    state.started_server_ms.insert(
                        (presentation_id.clone(), *replay_index),
                        envelope.server_received_unix_ms,
                    );
                }
            }
            StudySessionEvent::PlaybackCompleted {
                presentation_id,
                replay_index,
                listened_ms,
                media_duration_ms,
            } => {
                let Some(presentation) = presentation_map.get(presentation_id.as_str()) else {
                    issues.push(StudyRunnerIssue::UnknownPresentation {
                        presentation_id: presentation_id.clone(),
                    });
                    continue;
                };
                if !state
                    .started_replays
                    .get(presentation_id)
                    .is_some_and(|replays| replays.contains(replay_index))
                {
                    issues.push(StudyRunnerIssue::PlaybackNotStarted {
                        presentation_id: presentation_id.clone(),
                    });
                }
                if *media_duration_ms != presentation.duration_ms {
                    issues.push(StudyRunnerIssue::MediaDurationMismatch {
                        presentation_id: presentation_id.clone(),
                    });
                }
                let minimum = (presentation.duration_ms as f64
                    * package.protocol.minimum_playback_fraction)
                    .ceil() as u64;
                let server_elapsed_ms = state
                    .started_server_ms
                    .get(&(presentation_id.clone(), *replay_index))
                    .map(|started| envelope.server_received_unix_ms.saturating_sub(*started))
                    .unwrap_or_default();
                if *listened_ms < minimum || server_elapsed_ms < minimum {
                    issues.push(StudyRunnerIssue::PlaybackIncomplete {
                        presentation_id: presentation_id.clone(),
                    });
                }
                state
                    .completed_replays
                    .entry(presentation_id.clone())
                    .or_default()
                    .insert(*replay_index);
            }
            StudySessionEvent::ResponseRecorded {
                presentation_id,
                development_instability,
                earned_recapitulation,
                attention_check_response,
                elapsed_ms,
                ..
            } => {
                let expected = assigned
                    .get(state.current_index)
                    .copied()
                    .unwrap_or_default();
                if presentation_id != expected {
                    issues.push(StudyRunnerIssue::PresentationOrderViolation {
                        presentation_id: presentation_id.clone(),
                    });
                }
                if !state
                    .completed_replays
                    .get(presentation_id)
                    .is_some_and(|replays| !replays.is_empty())
                {
                    issues.push(StudyRunnerIssue::ResponseBeforePlayback {
                        presentation_id: presentation_id.clone(),
                    });
                }
                if !state.responses.insert(presentation_id.clone()) {
                    issues.push(StudyRunnerIssue::DuplicateResponse {
                        presentation_id: presentation_id.clone(),
                    });
                }
                for (field, value) in [
                    ("development_instability", *development_instability),
                    ("earned_recapitulation", *earned_recapitulation),
                ] {
                    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                        issues.push(StudyRunnerIssue::InvalidResponse {
                            presentation_id: presentation_id.clone(),
                            field: field.into(),
                        });
                    }
                }
                if *elapsed_ms > envelope.client_elapsed_ms {
                    issues.push(StudyRunnerIssue::InvalidResponse {
                        presentation_id: presentation_id.clone(),
                        field: "elapsed_ms".into(),
                    });
                }
                let attention_passed = attention_check_response
                    .is_some_and(|value| value == package.protocol.attention_check_expected_index);
                if package.protocol.require_attention_check && !attention_passed {
                    // Preserve the failed response, but require the explicitly
                    // preregistered exclusion before finalization.
                    state.attention_failed = true;
                }
                state.current_index = state.current_index.saturating_add(1);
            }
            StudySessionEvent::RankingsSubmitted {
                presentation_ids_best_to_worst,
            } => {
                let ranked: BTreeSet<_> = presentation_ids_best_to_worst.iter().collect();
                let expected: BTreeSet<_> = package
                    .presentations
                    .iter()
                    .map(|presentation| &presentation.presentation_id)
                    .collect();
                if state.responses.len() != assigned.len()
                    || presentation_ids_best_to_worst.len() != assigned.len()
                    || ranked != expected
                {
                    issues.push(StudyRunnerIssue::InvalidRanking);
                }
                if state.ranking_submitted {
                    issues.push(StudyRunnerIssue::InvalidRanking);
                }
                state.ranking_submitted = true;
            }
            StudySessionEvent::BlockExcluded { reason } => {
                if state.excluded || state.finalized {
                    issues.push(StudyRunnerIssue::ExclusionConflict);
                }
                state.excluded = true;
                state.exclusion_reason = Some(reason.clone());
            }
            StudySessionEvent::BlockFinalized => {
                if !state.excluded {
                    if state.attention_failed {
                        issues.push(StudyRunnerIssue::FailedAttentionCheckNotExcluded);
                    }
                    if state.responses.len() != assigned.len() {
                        issues.push(StudyRunnerIssue::IncompleteResponses);
                    }
                    if !state.ranking_submitted {
                        issues.push(StudyRunnerIssue::MissingRanking);
                    }
                } else if state.attention_failed
                    && state.exclusion_reason.as_ref()
                        != Some(&PreregisteredExclusion::FailedAttentionCheck)
                {
                    issues.push(StudyRunnerIssue::FailedAttentionCheckNotExcluded);
                }
                state.finalized = true;
            }
        }
    }
    if require_finalized && !state.finalized {
        issues.push(StudyRunnerIssue::FinalizationRequired);
    }
}

fn validate_protocol(protocol: &RunnerProtocol) -> Vec<StudyRunnerIssue> {
    let mut issues = Vec::new();
    for (field, text, digest) in [
        (
            "consent_document",
            &protocol.consent_document_text,
            &protocol.consent_document_sha256,
        ),
        (
            "instructions",
            &protocol.instructions_text,
            &protocol.instructions_sha256,
        ),
    ] {
        if text.trim().is_empty()
            || digest.len() != 64
            || !digest.bytes().all(|byte| byte.is_ascii_hexdigit())
            || sha256_hex(text.as_bytes()) != *digest
        {
            issues.push(StudyRunnerIssue::InvalidProtocol {
                field: field.into(),
            });
        }
    }
    if !protocol.minimum_playback_fraction.is_finite()
        || !(0.5..=1.0).contains(&protocol.minimum_playback_fraction)
    {
        issues.push(StudyRunnerIssue::InvalidProtocol {
            field: "minimum_playback_fraction".into(),
        });
    }
    if protocol.maximum_replays_per_presentation > 10 {
        issues.push(StudyRunnerIssue::InvalidProtocol {
            field: "maximum_replays_per_presentation".into(),
        });
    }
    if protocol.require_attention_check {
        if protocol.attention_check_prompt.trim().is_empty()
            || !(2..=6).contains(&protocol.attention_check_options.len())
            || protocol
                .attention_check_options
                .iter()
                .any(|option| option.trim().is_empty())
            || usize::from(protocol.attention_check_expected_index)
                >= protocol.attention_check_options.len()
        {
            issues.push(StudyRunnerIssue::InvalidProtocol {
                field: "attention_check".into(),
            });
        }
    } else if !protocol.attention_check_prompt.is_empty()
        || !protocol.attention_check_options.is_empty()
        || protocol.attention_check_expected_index != 0
    {
        issues.push(StudyRunnerIssue::InvalidProtocol {
            field: "disabled_attention_check".into(),
        });
    }
    issues
}

pub fn assigned_block<'a>(
    schedule: &'a ParticipantScheduleBook,
    block_id: &str,
) -> Option<&'a ParticipantBlockAssignment> {
    schedule
        .blocks
        .iter()
        .find(|block| block.block_id == block_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn package() -> StudyRunnerPackage {
        let mut package = StudyRunnerPackage {
            package_version: STUDY_RUNNER_PACKAGE_VERSION.into(),
            participant_schedule_sha256: "a".repeat(64),
            artifact_bundle_sha256: "b".repeat(64),
            block_id: "block".into(),
            participant_token: "participant".into(),
            key: FrozenTrialKey {
                fixture_id: "fixture".into(),
                seed: 7,
            },
            protocol: RunnerProtocol {
                consent_document_text: "I consent to participate.".into(),
                consent_document_sha256: sha256_hex(b"I consent to participate."),
                instructions_text: "Listen fully, then answer each question.".into(),
                instructions_sha256: sha256_hex(b"Listen fully, then answer each question."),
                minimum_playback_fraction: 0.9,
                maximum_replays_per_presentation: 1,
                require_attention_check: true,
                attention_check_prompt: "Select the second option.".into(),
                attention_check_options: vec!["First".into(), "Second".into()],
                attention_check_expected_index: 1,
                prevent_backtracking: true,
            },
            presentations: (0..4)
                .map(|index| RunnerPresentation {
                    presentation_id: format!("p{index}"),
                    anonymous_code: format!("M{index}"),
                    display_position: index,
                    audio_relative_path: format!("audio/p{index}.wav"),
                    audio_sha256: format!("{index}").repeat(64),
                    duration_ms: 1_000,
                })
                .collect(),
            package_sha256: String::new(),
        };
        package.package_sha256 = runner_package_commitment(&package).unwrap();
        package
    }

    #[test]
    fn rejects_response_before_playback_completion() {
        let package = package();
        let mut log = new_session_log(&package);
        append_session_event(
            &package,
            &mut log,
            1,
            0,
            StudySessionEvent::ConsentAccepted {
                consent_document_sha256: package.protocol.consent_document_sha256.clone(),
            },
        )
        .unwrap();
        append_session_event(
            &package,
            &mut log,
            2,
            1,
            StudySessionEvent::InstructionsAcknowledged {
                instructions_sha256: package.protocol.instructions_sha256.clone(),
            },
        )
        .unwrap();
        let result = append_session_event(
            &package,
            &mut log,
            2,
            10,
            StudySessionEvent::ResponseRecorded {
                presentation_id: "p0".into(),
                return_recognized: true,
                development_instability: 0.5,
                earned_recapitulation: 0.5,
                attention_check_response: Some(1),
                elapsed_ms: 10,
            },
        );
        assert!(result.is_err());
        assert_eq!(log.events.len(), 2);
    }

    #[test]
    fn complete_session_compiles_ranked_listener_block() {
        let package = package();
        let mut log = new_session_log(&package);
        let mut server = 1u64;
        let mut client = 0u64;
        append_session_event(
            &package,
            &mut log,
            server,
            client,
            StudySessionEvent::ConsentAccepted {
                consent_document_sha256: package.protocol.consent_document_sha256.clone(),
            },
        )
        .unwrap();
        append_session_event(
            &package,
            &mut log,
            2,
            1,
            StudySessionEvent::InstructionsAcknowledged {
                instructions_sha256: package.protocol.instructions_sha256.clone(),
            },
        )
        .unwrap();
        for presentation in &package.presentations {
            server += 1;
            client += 10;
            append_session_event(
                &package,
                &mut log,
                server,
                client,
                StudySessionEvent::PlaybackStarted {
                    presentation_id: presentation.presentation_id.clone(),
                    replay_index: 0,
                },
            )
            .unwrap();
            server += 1_000;
            client += 1_000;
            append_session_event(
                &package,
                &mut log,
                server,
                client,
                StudySessionEvent::PlaybackCompleted {
                    presentation_id: presentation.presentation_id.clone(),
                    replay_index: 0,
                    listened_ms: 1_000,
                    media_duration_ms: 1_000,
                },
            )
            .unwrap();
            server += 1;
            client += 10;
            append_session_event(
                &package,
                &mut log,
                server,
                client,
                StudySessionEvent::ResponseRecorded {
                    presentation_id: presentation.presentation_id.clone(),
                    return_recognized: true,
                    development_instability: 0.4,
                    earned_recapitulation: 0.7,
                    attention_check_response: Some(1),
                    elapsed_ms: client,
                },
            )
            .unwrap();
        }
        server += 1;
        client += 10;
        append_session_event(
            &package,
            &mut log,
            server,
            client,
            StudySessionEvent::RankingsSubmitted {
                presentation_ids_best_to_worst: package
                    .presentations
                    .iter()
                    .map(|presentation| presentation.presentation_id.clone())
                    .collect(),
            },
        )
        .unwrap();
        append_session_event(
            &package,
            &mut log,
            server + 1,
            client + 1,
            StudySessionEvent::BlockFinalized,
        )
        .unwrap();
        assert!(validate_session_log(&package, &log, true).is_empty());
        let block = compile_listener_block(&package, &log).unwrap();
        assert_eq!(block.responses.len(), 4);
        assert_eq!(block.responses[0].preference_rank, 1);
        assert_eq!(block.responses[3].preference_rank, 4);
    }

    #[test]
    fn failed_attention_check_requires_explicit_exclusion() {
        let package = package();
        let mut log = new_session_log(&package);
        append_session_event(
            &package,
            &mut log,
            1,
            0,
            StudySessionEvent::ConsentAccepted {
                consent_document_sha256: package.protocol.consent_document_sha256.clone(),
            },
        )
        .unwrap();
        append_session_event(
            &package,
            &mut log,
            2,
            1,
            StudySessionEvent::InstructionsAcknowledged {
                instructions_sha256: package.protocol.instructions_sha256.clone(),
            },
        )
        .unwrap();
        append_session_event(
            &package,
            &mut log,
            2,
            10,
            StudySessionEvent::PlaybackStarted {
                presentation_id: "p0".into(),
                replay_index: 0,
            },
        )
        .unwrap();
        append_session_event(
            &package,
            &mut log,
            1_002,
            1_010,
            StudySessionEvent::PlaybackCompleted {
                presentation_id: "p0".into(),
                replay_index: 0,
                listened_ms: 1_000,
                media_duration_ms: 1_000,
            },
        )
        .unwrap();
        append_session_event(
            &package,
            &mut log,
            1_003,
            1_020,
            StudySessionEvent::ResponseRecorded {
                presentation_id: "p0".into(),
                return_recognized: false,
                development_instability: 0.5,
                earned_recapitulation: 0.5,
                attention_check_response: Some(0),
                elapsed_ms: 1_020,
            },
        )
        .unwrap();
        assert!(
            append_session_event(
                &package,
                &mut log,
                1_004,
                1_021,
                StudySessionEvent::BlockFinalized,
            )
            .is_err()
        );
        append_session_event(
            &package,
            &mut log,
            1_004,
            1_021,
            StudySessionEvent::BlockExcluded {
                reason: PreregisteredExclusion::FailedAttentionCheck,
            },
        )
        .unwrap();
        append_session_event(
            &package,
            &mut log,
            1_005,
            1_022,
            StudySessionEvent::BlockFinalized,
        )
        .unwrap();
        let block = compile_listener_block(&package, &log).unwrap();
        assert!(matches!(
            block.status,
            EvidenceBlockStatus::Excluded {
                reason: PreregisteredExclusion::FailedAttentionCheck
            }
        ));
    }

    #[test]
    fn server_timing_rejects_forged_client_playback_duration() {
        let package = package();
        let mut log = new_session_log(&package);
        append_session_event(
            &package,
            &mut log,
            1,
            0,
            StudySessionEvent::ConsentAccepted {
                consent_document_sha256: package.protocol.consent_document_sha256.clone(),
            },
        )
        .unwrap();
        append_session_event(
            &package,
            &mut log,
            2,
            1,
            StudySessionEvent::InstructionsAcknowledged {
                instructions_sha256: package.protocol.instructions_sha256.clone(),
            },
        )
        .unwrap();
        append_session_event(
            &package,
            &mut log,
            10,
            10,
            StudySessionEvent::PlaybackStarted {
                presentation_id: "p0".into(),
                replay_index: 0,
            },
        )
        .unwrap();
        let result = append_session_event(
            &package,
            &mut log,
            100,
            1_010,
            StudySessionEvent::PlaybackCompleted {
                presentation_id: "p0".into(),
                replay_index: 0,
                listened_ms: 1_000,
                media_duration_ms: 1_000,
            },
        );
        assert!(result.is_err());
        assert_eq!(log.events.len(), 3);
    }

    #[test]
    fn failed_attention_requires_matching_exclusion_reason() {
        let package = package();
        let mut log = new_session_log(&package);
        append_session_event(
            &package,
            &mut log,
            1,
            0,
            StudySessionEvent::ConsentAccepted {
                consent_document_sha256: package.protocol.consent_document_sha256.clone(),
            },
        )
        .unwrap();
        append_session_event(
            &package,
            &mut log,
            2,
            1,
            StudySessionEvent::InstructionsAcknowledged {
                instructions_sha256: package.protocol.instructions_sha256.clone(),
            },
        )
        .unwrap();
        append_session_event(
            &package,
            &mut log,
            10,
            10,
            StudySessionEvent::PlaybackStarted {
                presentation_id: "p0".into(),
                replay_index: 0,
            },
        )
        .unwrap();
        append_session_event(
            &package,
            &mut log,
            1_010,
            1_010,
            StudySessionEvent::PlaybackCompleted {
                presentation_id: "p0".into(),
                replay_index: 0,
                listened_ms: 1_000,
                media_duration_ms: 1_000,
            },
        )
        .unwrap();
        append_session_event(
            &package,
            &mut log,
            1_020,
            1_020,
            StudySessionEvent::ResponseRecorded {
                presentation_id: "p0".into(),
                return_recognized: false,
                development_instability: 0.5,
                earned_recapitulation: 0.5,
                attention_check_response: Some(0),
                elapsed_ms: 1_020,
            },
        )
        .unwrap();
        append_session_event(
            &package,
            &mut log,
            1_021,
            1_021,
            StudySessionEvent::BlockExcluded {
                reason: PreregisteredExclusion::TechnicalPlaybackFailure,
            },
        )
        .unwrap();
        assert!(
            append_session_event(
                &package,
                &mut log,
                1_022,
                1_022,
                StudySessionEvent::BlockFinalized,
            )
            .is_err()
        );
    }

    #[test]
    fn hash_chain_detects_event_tampering() {
        let package = package();
        let mut log = new_session_log(&package);
        append_session_event(
            &package,
            &mut log,
            1,
            0,
            StudySessionEvent::ConsentAccepted {
                consent_document_sha256: package.protocol.consent_document_sha256.clone(),
            },
        )
        .unwrap();
        append_session_event(
            &package,
            &mut log,
            2,
            1,
            StudySessionEvent::InstructionsAcknowledged {
                instructions_sha256: package.protocol.instructions_sha256.clone(),
            },
        )
        .unwrap();
        log.events[0].client_elapsed_ms = 99;
        assert!(
            validate_session_log(&package, &log, false)
                .contains(&StudyRunnerIssue::EventDigestMismatch { sequence: 0 })
        );
    }
}
