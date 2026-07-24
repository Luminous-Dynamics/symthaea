// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sealed pilot session collection.
//!
//! The envelope binds runner packages and logs without joining the private arm
//! codebook. Operational monitoring may inspect only derived pilot records.

use crate::blinded_study::BlindedSchedule;
use crate::cohort_registry::PilotCohortRegistry;
use crate::evidence_digest::canonical_json_sha256;
use crate::pilot_monitoring::{PilotOperationalRecord, compile_pilot_operational_record};
use crate::pilot_protocol::FrozenPilotProtocol;
use crate::pilot_schedule::PilotParticipantScheduleBook;
use crate::study_artifact::StudyArtifactBundle;
use crate::study_runner::{
    StudyRunnerIssue, StudyRunnerPackage, StudySessionLog, validate_pilot_runner_package,
    validate_session_log,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const PILOT_COLLECTION_VERSION: &str = "symthaea-muse-pilot-collection-v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PilotSessionSubmission {
    pub package: StudyRunnerPackage,
    pub log: StudySessionLog,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PilotCollectionEnvelope {
    pub collection_version: String,
    pub pilot_protocol_sha256: String,
    pub pilot_schedule_sha256: String,
    pub artifact_bundle_sha256: String,
    pub cohort_registry_sha256: String,
    pub collected_at_utc: String,
    pub sessions: Vec<PilotSessionSubmission>,
    pub operational_records: Vec<PilotOperationalRecord>,
    pub collection_sha256: String,
}

#[derive(Serialize)]
struct PilotCollectionCommitment<'a> {
    collection_version: &'a str,
    pilot_protocol_sha256: &'a str,
    pilot_schedule_sha256: &'a str,
    artifact_bundle_sha256: &'a str,
    cohort_registry_sha256: &'a str,
    collected_at_utc: &'a str,
    sessions: &'a [PilotSessionSubmission],
    operational_records: &'a [PilotOperationalRecord],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PilotCollectionIssue {
    WrongVersion {
        found: String,
    },
    EmptyCollectionTime,
    SerializationFailed {
        field: String,
    },
    DigestMismatch {
        field: String,
    },
    DuplicateSession {
        block_id: String,
    },
    UnknownSession {
        block_id: String,
    },
    MissingSession {
        block_id: String,
    },
    RunnerPackage {
        block_id: String,
        issues: Vec<StudyRunnerIssue>,
    },
    RunnerSession {
        block_id: String,
        issues: Vec<StudyRunnerIssue>,
    },
    OperationalRecordFailed {
        block_id: String,
    },
    OperationalRecordMismatch {
        block_id: String,
    },
    CollectionDigestMismatch,
}

pub fn pilot_collection_commitment(
    envelope: &PilotCollectionEnvelope,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&PilotCollectionCommitment {
        collection_version: &envelope.collection_version,
        pilot_protocol_sha256: &envelope.pilot_protocol_sha256,
        pilot_schedule_sha256: &envelope.pilot_schedule_sha256,
        artifact_bundle_sha256: &envelope.artifact_bundle_sha256,
        cohort_registry_sha256: &envelope.cohort_registry_sha256,
        collected_at_utc: &envelope.collected_at_utc,
        sessions: &envelope.sessions,
        operational_records: &envelope.operational_records,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn seal_pilot_collection(
    protocol: &FrozenPilotProtocol,
    base_schedule: &BlindedSchedule,
    pilot_schedule: &PilotParticipantScheduleBook,
    artifacts: &StudyArtifactBundle,
    registry: &PilotCohortRegistry,
    collected_at_utc: String,
    mut sessions: Vec<PilotSessionSubmission>,
) -> Result<PilotCollectionEnvelope, Vec<PilotCollectionIssue>> {
    if collected_at_utc.trim().is_empty() {
        return Err(vec![PilotCollectionIssue::EmptyCollectionTime]);
    }
    sessions.sort_by(|left, right| left.package.block_id.cmp(&right.package.block_id));
    let mut operational_records = Vec::with_capacity(sessions.len());
    for submission in &sessions {
        match compile_pilot_operational_record(&submission.package, &submission.log) {
            Ok(record) => operational_records.push(record),
            Err(_) => {
                return Err(vec![PilotCollectionIssue::OperationalRecordFailed {
                    block_id: submission.package.block_id.clone(),
                }]);
            }
        }
    }
    operational_records.sort_by(|left, right| left.block_id.cmp(&right.block_id));
    let mut envelope = PilotCollectionEnvelope {
        collection_version: PILOT_COLLECTION_VERSION.into(),
        pilot_protocol_sha256: canonical_json_sha256(protocol).map_err(|_| {
            vec![PilotCollectionIssue::SerializationFailed {
                field: "pilot_protocol".into(),
            }]
        })?,
        pilot_schedule_sha256: canonical_json_sha256(pilot_schedule).map_err(|_| {
            vec![PilotCollectionIssue::SerializationFailed {
                field: "pilot_schedule".into(),
            }]
        })?,
        artifact_bundle_sha256: artifacts.bundle_sha256.clone(),
        cohort_registry_sha256: registry.registry_sha256.clone(),
        collected_at_utc,
        sessions,
        operational_records,
        collection_sha256: String::new(),
    };
    envelope.collection_sha256 = pilot_collection_commitment(&envelope).map_err(|_| {
        vec![PilotCollectionIssue::SerializationFailed {
            field: "pilot_collection".into(),
        }]
    })?;
    let issues = validate_pilot_collection(
        protocol,
        base_schedule,
        pilot_schedule,
        artifacts,
        registry,
        &envelope,
    );
    if issues.is_empty() {
        Ok(envelope)
    } else {
        Err(issues)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn validate_pilot_collection(
    protocol: &FrozenPilotProtocol,
    base_schedule: &BlindedSchedule,
    pilot_schedule: &PilotParticipantScheduleBook,
    artifacts: &StudyArtifactBundle,
    registry: &PilotCohortRegistry,
    envelope: &PilotCollectionEnvelope,
) -> Vec<PilotCollectionIssue> {
    let mut issues = Vec::new();
    if envelope.collection_version != PILOT_COLLECTION_VERSION {
        issues.push(PilotCollectionIssue::WrongVersion {
            found: envelope.collection_version.clone(),
        });
    }
    if envelope.collected_at_utc.trim().is_empty() {
        issues.push(PilotCollectionIssue::EmptyCollectionTime);
    }
    verify_digest(
        protocol,
        &envelope.pilot_protocol_sha256,
        "pilot_protocol",
        &mut issues,
    );
    verify_digest(
        pilot_schedule,
        &envelope.pilot_schedule_sha256,
        "pilot_schedule",
        &mut issues,
    );
    if envelope.artifact_bundle_sha256 != artifacts.bundle_sha256 {
        issues.push(PilotCollectionIssue::DigestMismatch {
            field: "artifact_bundle_sha256".into(),
        });
    }
    if envelope.cohort_registry_sha256 != registry.registry_sha256 {
        issues.push(PilotCollectionIssue::DigestMismatch {
            field: "cohort_registry_sha256".into(),
        });
    }

    let assignments: BTreeSet<_> = pilot_schedule
        .blocks
        .iter()
        .map(|block| block.block_id.as_str())
        .collect();
    let mut sessions = BTreeMap::new();
    for submission in &envelope.sessions {
        let block_id = submission.package.block_id.clone();
        if !assignments.contains(block_id.as_str()) {
            issues.push(PilotCollectionIssue::UnknownSession {
                block_id: block_id.clone(),
            });
        }
        if sessions.insert(block_id.clone(), submission).is_some() {
            issues.push(PilotCollectionIssue::DuplicateSession { block_id });
        }
    }
    for block in &pilot_schedule.blocks {
        if !sessions.contains_key(&block.block_id) {
            issues.push(PilotCollectionIssue::MissingSession {
                block_id: block.block_id.clone(),
            });
        }
    }
    let operational_by_block: BTreeMap<_, _> = envelope
        .operational_records
        .iter()
        .map(|record| (record.block_id.as_str(), record))
        .collect();
    for (block_id, submission) in sessions {
        let package_issues = validate_pilot_runner_package(
            &submission.package,
            base_schedule,
            pilot_schedule,
            artifacts,
        );
        if !package_issues.is_empty() {
            issues.push(PilotCollectionIssue::RunnerPackage {
                block_id: block_id.clone(),
                issues: package_issues,
            });
        }
        let session_issues = validate_session_log(&submission.package, &submission.log, true);
        if !session_issues.is_empty() {
            issues.push(PilotCollectionIssue::RunnerSession {
                block_id: block_id.clone(),
                issues: session_issues,
            });
        }
        match (
            compile_pilot_operational_record(&submission.package, &submission.log),
            operational_by_block.get(block_id.as_str()),
        ) {
            (Ok(expected), Some(found)) if *found == &expected => {}
            (Ok(_), _) => issues.push(PilotCollectionIssue::OperationalRecordMismatch { block_id }),
            (Err(_), _) => issues.push(PilotCollectionIssue::OperationalRecordFailed { block_id }),
        }
    }
    match pilot_collection_commitment(envelope) {
        Ok(value) if value == envelope.collection_sha256 => {}
        Ok(_) => issues.push(PilotCollectionIssue::CollectionDigestMismatch),
        Err(_) => issues.push(PilotCollectionIssue::SerializationFailed {
            field: "pilot_collection".into(),
        }),
    }
    issues
}

fn verify_digest<T: Serialize>(
    value: &T,
    expected: &str,
    field: &str,
    issues: &mut Vec<PilotCollectionIssue>,
) {
    match canonical_json_sha256(value) {
        Ok(value) if value == expected => {}
        Ok(_) => issues.push(PilotCollectionIssue::DigestMismatch {
            field: field.into(),
        }),
        Err(_) => issues.push(PilotCollectionIssue::SerializationFailed {
            field: field.into(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collection_version_is_explicit() {
        assert!(PILOT_COLLECTION_VERSION.ends_with("v1"));
    }
}
