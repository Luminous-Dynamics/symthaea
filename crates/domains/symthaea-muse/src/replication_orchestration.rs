// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hash-chained lifecycle for an independent replication program.
//!
//! The lifecycle prevents source-outcome reveal or synthesis publication before
//! prospective collection is closed and each site has completed its frozen analysis.

use crate::evidence_digest::canonical_json_sha256;
use crate::replication_protocol::FrozenReplicationProtocol;
use serde::{Deserialize, Serialize};

pub const REPLICATION_ORCHESTRATION_VERSION: &str = "symthaea-muse-replication-orchestration-v1";
pub const REPLICATION_ORCHESTRATION_GENESIS: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplicationLifecyclePhase {
    Draft,
    ProtocolFrozen,
    SitesRegistered,
    PackagesIssued,
    CollectionOpen,
    CollectionClosed,
    SourceOutcomeRevealAuthorized,
    AnalysisComplete,
    SynthesisPublished,
    StewardshipReleased,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplicationLifecycleEvent {
    pub sequence: u32,
    pub from: ReplicationLifecyclePhase,
    pub to: ReplicationLifecyclePhase,
    pub actor_id: String,
    pub authority_sha256: String,
    pub recorded_at_utc: String,
    pub previous_event_sha256: String,
    pub event_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplicationOrchestrationLog {
    pub orchestration_version: String,
    pub replication_id: String,
    pub source_final_release_sha256: String,
    pub protocol_sha256: String,
    pub current_phase: ReplicationLifecyclePhase,
    pub events: Vec<ReplicationLifecycleEvent>,
    pub log_sha256: String,
}

#[derive(Serialize)]
struct EventCommitment<'a> {
    replication_id: &'a str,
    sequence: u32,
    from: ReplicationLifecyclePhase,
    to: ReplicationLifecyclePhase,
    actor_id: &'a str,
    authority_sha256: &'a str,
    recorded_at_utc: &'a str,
    previous_event_sha256: &'a str,
}

#[derive(Serialize)]
struct LogCommitment<'a> {
    orchestration_version: &'a str,
    replication_id: &'a str,
    source_final_release_sha256: &'a str,
    protocol_sha256: &'a str,
    current_phase: ReplicationLifecyclePhase,
    events: &'a [ReplicationLifecycleEvent],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplicationOrchestrationIssue {
    WrongVersion {
        found: String,
    },
    EmptyField {
        sequence: Option<u32>,
        field: String,
    },
    InvalidDigest {
        sequence: Option<u32>,
        field: String,
    },
    UnexpectedSequence {
        expected: u32,
        found: u32,
    },
    InvalidTransition {
        from: ReplicationLifecyclePhase,
        to: ReplicationLifecyclePhase,
    },
    PhaseMismatch,
    ChainBroken {
        sequence: u32,
    },
    EventDigestMismatch {
        sequence: u32,
    },
    SerializationFailed,
    LogDigestMismatch,
}

pub fn new_replication_orchestration(
    protocol: &FrozenReplicationProtocol,
) -> Result<ReplicationOrchestrationLog, serde_json::Error> {
    let mut log = ReplicationOrchestrationLog {
        orchestration_version: REPLICATION_ORCHESTRATION_VERSION.into(),
        replication_id: protocol.replication_id.clone(),
        source_final_release_sha256: protocol.source_final_release_sha256.clone(),
        protocol_sha256: protocol.protocol_sha256.clone(),
        current_phase: ReplicationLifecyclePhase::Draft,
        events: Vec::new(),
        log_sha256: String::new(),
    };
    log.log_sha256 = replication_orchestration_commitment(&log)?;
    Ok(log)
}

pub fn replication_lifecycle_event_commitment(
    replication_id: &str,
    event: &ReplicationLifecycleEvent,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&EventCommitment {
        replication_id,
        sequence: event.sequence,
        from: event.from,
        to: event.to,
        actor_id: &event.actor_id,
        authority_sha256: &event.authority_sha256,
        recorded_at_utc: &event.recorded_at_utc,
        previous_event_sha256: &event.previous_event_sha256,
    })
}

pub fn replication_orchestration_commitment(
    log: &ReplicationOrchestrationLog,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&LogCommitment {
        orchestration_version: &log.orchestration_version,
        replication_id: &log.replication_id,
        source_final_release_sha256: &log.source_final_release_sha256,
        protocol_sha256: &log.protocol_sha256,
        current_phase: log.current_phase,
        events: &log.events,
    })
}

pub fn append_replication_transition(
    log: &mut ReplicationOrchestrationLog,
    to: ReplicationLifecyclePhase,
    actor_id: String,
    authority_sha256: String,
    recorded_at_utc: String,
) -> Result<(), Vec<ReplicationOrchestrationIssue>> {
    let current_issues = validate_replication_orchestration(log);
    if !current_issues.is_empty() {
        return Err(current_issues);
    }
    if !valid_transition(log.current_phase, to) {
        return Err(vec![ReplicationOrchestrationIssue::InvalidTransition {
            from: log.current_phase,
            to,
        }]);
    }
    let mut event = ReplicationLifecycleEvent {
        sequence: log.events.len() as u32 + 1,
        from: log.current_phase,
        to,
        actor_id,
        authority_sha256,
        recorded_at_utc,
        previous_event_sha256: log
            .events
            .last()
            .map_or(REPLICATION_ORCHESTRATION_GENESIS, |event| {
                event.event_sha256.as_str()
            })
            .to_string(),
        event_sha256: String::new(),
    };
    event.event_sha256 = replication_lifecycle_event_commitment(&log.replication_id, &event)
        .map_err(|_| vec![ReplicationOrchestrationIssue::SerializationFailed])?;
    log.events.push(event);
    log.current_phase = to;
    log.log_sha256 = replication_orchestration_commitment(log)
        .map_err(|_| vec![ReplicationOrchestrationIssue::SerializationFailed])?;
    let issues = validate_replication_orchestration(log);
    if issues.is_empty() {
        Ok(())
    } else {
        Err(issues)
    }
}

pub fn validate_replication_orchestration(
    log: &ReplicationOrchestrationLog,
) -> Vec<ReplicationOrchestrationIssue> {
    let mut issues = Vec::new();
    if log.orchestration_version != REPLICATION_ORCHESTRATION_VERSION {
        issues.push(ReplicationOrchestrationIssue::WrongVersion {
            found: log.orchestration_version.clone(),
        });
    }
    for (field, value) in [("replication_id", log.replication_id.as_str())] {
        if value.trim().is_empty() {
            issues.push(ReplicationOrchestrationIssue::EmptyField {
                sequence: None,
                field: field.into(),
            });
        }
    }
    for (field, digest) in [
        (
            "source_final_release_sha256",
            log.source_final_release_sha256.as_str(),
        ),
        ("protocol_sha256", log.protocol_sha256.as_str()),
        ("log_sha256", log.log_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            issues.push(ReplicationOrchestrationIssue::InvalidDigest {
                sequence: None,
                field: field.into(),
            });
        }
    }
    let mut previous = REPLICATION_ORCHESTRATION_GENESIS.to_string();
    let mut phase = ReplicationLifecyclePhase::Draft;
    for (index, event) in log.events.iter().enumerate() {
        let expected_sequence = index as u32 + 1;
        if event.sequence != expected_sequence {
            issues.push(ReplicationOrchestrationIssue::UnexpectedSequence {
                expected: expected_sequence,
                found: event.sequence,
            });
        }
        if event.from != phase || !valid_transition(event.from, event.to) {
            issues.push(ReplicationOrchestrationIssue::InvalidTransition {
                from: event.from,
                to: event.to,
            });
        }
        if event.previous_event_sha256 != previous {
            issues.push(ReplicationOrchestrationIssue::ChainBroken {
                sequence: event.sequence,
            });
        }
        for (field, value) in [
            ("actor_id", event.actor_id.as_str()),
            ("recorded_at_utc", event.recorded_at_utc.as_str()),
        ] {
            if value.trim().is_empty() {
                issues.push(ReplicationOrchestrationIssue::EmptyField {
                    sequence: Some(event.sequence),
                    field: field.into(),
                });
            }
        }
        for (field, digest) in [
            ("authority_sha256", event.authority_sha256.as_str()),
            (
                "previous_event_sha256",
                event.previous_event_sha256.as_str(),
            ),
            ("event_sha256", event.event_sha256.as_str()),
        ] {
            if !is_sha256(digest) {
                issues.push(ReplicationOrchestrationIssue::InvalidDigest {
                    sequence: Some(event.sequence),
                    field: field.into(),
                });
            }
        }
        match replication_lifecycle_event_commitment(&log.replication_id, event) {
            Ok(digest) if digest == event.event_sha256 => {}
            Ok(_) => issues.push(ReplicationOrchestrationIssue::EventDigestMismatch {
                sequence: event.sequence,
            }),
            Err(_) => issues.push(ReplicationOrchestrationIssue::SerializationFailed),
        }
        previous = event.event_sha256.clone();
        phase = event.to;
    }
    if phase != log.current_phase {
        issues.push(ReplicationOrchestrationIssue::PhaseMismatch);
    }
    match replication_orchestration_commitment(log) {
        Ok(digest) if digest == log.log_sha256 => {}
        Ok(_) => issues.push(ReplicationOrchestrationIssue::LogDigestMismatch),
        Err(_) => issues.push(ReplicationOrchestrationIssue::SerializationFailed),
    }
    issues
}

fn valid_transition(from: ReplicationLifecyclePhase, to: ReplicationLifecyclePhase) -> bool {
    matches!(
        (from, to),
        (
            ReplicationLifecyclePhase::Draft,
            ReplicationLifecyclePhase::ProtocolFrozen
        ) | (
            ReplicationLifecyclePhase::ProtocolFrozen,
            ReplicationLifecyclePhase::SitesRegistered
        ) | (
            ReplicationLifecyclePhase::SitesRegistered,
            ReplicationLifecyclePhase::PackagesIssued
        ) | (
            ReplicationLifecyclePhase::PackagesIssued,
            ReplicationLifecyclePhase::CollectionOpen
        ) | (
            ReplicationLifecyclePhase::CollectionOpen,
            ReplicationLifecyclePhase::CollectionClosed
        ) | (
            ReplicationLifecyclePhase::CollectionClosed,
            ReplicationLifecyclePhase::SourceOutcomeRevealAuthorized
        ) | (
            ReplicationLifecyclePhase::SourceOutcomeRevealAuthorized,
            ReplicationLifecyclePhase::AnalysisComplete
        ) | (
            ReplicationLifecyclePhase::AnalysisComplete,
            ReplicationLifecyclePhase::SynthesisPublished
        ) | (
            ReplicationLifecyclePhase::SynthesisPublished,
            ReplicationLifecyclePhase::StewardshipReleased
        )
    )
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_outcomes_cannot_be_revealed_before_close() {
        assert!(!valid_transition(
            ReplicationLifecyclePhase::CollectionOpen,
            ReplicationLifecyclePhase::SourceOutcomeRevealAuthorized,
        ));
        assert!(valid_transition(
            ReplicationLifecyclePhase::CollectionClosed,
            ReplicationLifecyclePhase::SourceOutcomeRevealAuthorized,
        ));
    }
}
