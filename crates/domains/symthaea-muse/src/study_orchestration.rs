// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hash-chained study lifecycle orchestration.
//!
//! This state machine prevents collection before authorities are sealed,
//! prevents confirmatory freezing before the pilot is closed, and prevents
//! unblinding before confirmatory collection is irreversibly closed.

use crate::evidence_digest::canonical_json_sha256;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const STUDY_ORCHESTRATION_VERSION_V1: &str = "symthaea-muse-study-orchestration-v1";
const STUDY_ORCHESTRATION_VERSION_V2: &str = "symthaea-muse-study-orchestration-v2";
pub const STUDY_ORCHESTRATION_VERSION: &str = "symthaea-muse-study-orchestration-v3";
const ORCHESTRATION_GENESIS_SHA256: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum StudyAuthorityRole {
    FrozenManifest,
    FrozenMethodology,
    PilotProtocol,
    PilotRandomizationCommitment,
    PilotArtifactBundle,
    PilotParticipantSchedule,
    PilotCohortRegistry,
    PilotCollection,
    PilotAmendmentLedger,
    PilotReport,
    ExternalPilotReceipt,
    ExternalReviewProtocol,
    ExternalReviewEvidenceIndex,
    ExternalReviewCompletion,
    ConfirmatoryAuthoritySnapshot,
    ConfirmatoryAmendmentLedger,
    WorkspaceValidation,
    HumanStudyGovernance,
    ConfirmatoryDryRun,
    IndependentReproductionReadiness,
    ConfirmatoryReadinessReport,
    ConfirmatoryReadinessRelease,
    ConfirmatoryPreregistration,
    ConfirmatoryArtifactBundle,
    ConfirmatoryParticipantSchedule,
    ConfirmatoryCohortRegistry,
    ConfirmatoryCollectionProtocol,
    ConfirmatoryCollectionMonitor,
    ConfirmatoryCollection,
    ConfirmatoryCollectionCloseReceipt,
    BlindingCodebook,
    RandomizationKeyReveal,
    ConfirmatoryUnblindingReceipt,
    PrimaryAnalysisReport,
    IndependentAnalysisReport,
    ReproducibilityAttestation,
    ConfirmatoryAnalysisExecution,
    ConfirmatoryPublicationRecord,
    PostPublicationAuditLedger,
    StudyReleaseBundle,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StudyAuthorityBinding {
    pub role: StudyAuthorityRole,
    pub sha256: String,
    pub external_uri: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum StudyLifecyclePhase {
    Draft,
    PilotRegistered,
    PilotArtifactsSealed,
    PilotCollectionOpen,
    PilotCollectionClosed,
    PilotReviewed,
    ExternalReviewOpen,
    ExternalReviewComplete,
    ConfirmatoryReady,
    ConfirmatoryFrozen,
    ConfirmatoryArtifactsSealed,
    ConfirmatoryCollectionOpen,
    ConfirmatoryCollectionClosed,
    Unblinded,
    Analyzed,
    Published,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StudyLifecycleTransition {
    pub sequence: u32,
    pub from: StudyLifecyclePhase,
    pub to: StudyLifecyclePhase,
    pub recorded_at_utc: String,
    pub operator_id: String,
    pub authorization_sha256: String,
    pub added_authorities: Vec<StudyAuthorityBinding>,
    pub previous_transition_sha256: String,
    pub transition_sha256: String,
}

#[derive(Serialize)]
struct TransitionCommitment<'a> {
    orchestration_id: &'a str,
    sequence: u32,
    from: StudyLifecyclePhase,
    to: StudyLifecyclePhase,
    recorded_at_utc: &'a str,
    operator_id: &'a str,
    authorization_sha256: &'a str,
    added_authorities: &'a [StudyAuthorityBinding],
    previous_transition_sha256: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StudyOrchestrationLog {
    pub orchestration_version: String,
    pub orchestration_id: String,
    pub current_phase: StudyLifecyclePhase,
    pub authorities: Vec<StudyAuthorityBinding>,
    pub transitions: Vec<StudyLifecycleTransition>,
    pub log_sha256: String,
}

#[derive(Serialize)]
struct OrchestrationCommitment<'a> {
    orchestration_version: &'a str,
    orchestration_id: &'a str,
    current_phase: StudyLifecyclePhase,
    authorities: &'a [StudyAuthorityBinding],
    transitions: &'a [StudyLifecycleTransition],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StudyOrchestrationIssue {
    WrongVersion {
        found: String,
    },
    EmptyField {
        field: String,
    },
    InvalidDigest {
        field: String,
    },
    DuplicateAuthority {
        role: StudyAuthorityRole,
    },
    ConflictingAuthority {
        role: StudyAuthorityRole,
    },
    IllegalTransition {
        from: StudyLifecyclePhase,
        to: StudyLifecyclePhase,
    },
    MissingAuthority {
        phase: StudyLifecyclePhase,
        role: StudyAuthorityRole,
    },
    UnexpectedSequence {
        expected: u32,
        found: u32,
    },
    TransitionFromMismatch {
        sequence: u32,
    },
    TransitionChainBroken {
        sequence: u32,
    },
    TransitionDigestMismatch {
        sequence: u32,
    },
    SerializationFailed {
        field: String,
    },
    CurrentPhaseMismatch,
    AuthoritySnapshotMismatch,
    LogDigestMismatch,
    PilotAmendmentAfterConfirmatoryFreeze,
    UnblindingBeforeCollectionClose,
    LegacyOrchestrationReadOnly,
    LegacyUpgradeAfterConfirmatoryStart,
}

pub fn new_study_orchestration(orchestration_id: impl Into<String>) -> StudyOrchestrationLog {
    let mut log = StudyOrchestrationLog {
        orchestration_version: STUDY_ORCHESTRATION_VERSION.into(),
        orchestration_id: orchestration_id.into(),
        current_phase: StudyLifecyclePhase::Draft,
        authorities: Vec::new(),
        transitions: Vec::new(),
        log_sha256: String::new(),
    };
    log.log_sha256 =
        study_orchestration_commitment(&log).expect("empty orchestration log is serializable");
    log
}

pub fn study_transition_commitment(
    orchestration_id: &str,
    transition: &StudyLifecycleTransition,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&TransitionCommitment {
        orchestration_id,
        sequence: transition.sequence,
        from: transition.from,
        to: transition.to,
        recorded_at_utc: &transition.recorded_at_utc,
        operator_id: &transition.operator_id,
        authorization_sha256: &transition.authorization_sha256,
        added_authorities: &transition.added_authorities,
        previous_transition_sha256: &transition.previous_transition_sha256,
    })
}

pub fn study_orchestration_commitment(
    log: &StudyOrchestrationLog,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&OrchestrationCommitment {
        orchestration_version: &log.orchestration_version,
        orchestration_id: &log.orchestration_id,
        current_phase: log.current_phase,
        authorities: &log.authorities,
        transitions: &log.transitions,
    })
}

pub fn upgrade_legacy_study_orchestration(
    log: &mut StudyOrchestrationLog,
) -> Result<(), Vec<StudyOrchestrationIssue>> {
    let issues = validate_study_orchestration(log);
    if !issues.is_empty() {
        return Err(issues);
    }
    if log.orchestration_version == STUDY_ORCHESTRATION_VERSION {
        return Ok(());
    }
    if log.current_phase >= StudyLifecyclePhase::ConfirmatoryCollectionOpen {
        return Err(vec![
            StudyOrchestrationIssue::LegacyUpgradeAfterConfirmatoryStart,
        ]);
    }
    log.orchestration_version = STUDY_ORCHESTRATION_VERSION.into();
    log.log_sha256 = study_orchestration_commitment(log).map_err(|_| {
        vec![StudyOrchestrationIssue::SerializationFailed {
            field: "orchestration_log".into(),
        }]
    })?;
    Ok(())
}

pub fn append_study_transition(
    log: &mut StudyOrchestrationLog,
    to: StudyLifecyclePhase,
    recorded_at_utc: String,
    operator_id: String,
    authorization_sha256: String,
    mut added_authorities: Vec<StudyAuthorityBinding>,
) -> Result<(), Vec<StudyOrchestrationIssue>> {
    let mut issues = validate_study_orchestration(log);
    if !issues.is_empty() {
        return Err(issues);
    }
    if log.orchestration_version != STUDY_ORCHESTRATION_VERSION {
        return Err(vec![StudyOrchestrationIssue::LegacyOrchestrationReadOnly]);
    }
    if !legal_transition(&log.orchestration_version, log.current_phase, to) {
        issues.push(StudyOrchestrationIssue::IllegalTransition {
            from: log.current_phase,
            to,
        });
    }
    for (field, value) in [
        ("recorded_at_utc", recorded_at_utc.as_str()),
        ("operator_id", operator_id.as_str()),
    ] {
        if value.trim().is_empty() {
            issues.push(StudyOrchestrationIssue::EmptyField {
                field: field.into(),
            });
        }
    }
    if !is_sha256(&authorization_sha256) {
        issues.push(StudyOrchestrationIssue::InvalidDigest {
            field: "authorization_sha256".into(),
        });
    }
    let existing: BTreeMap<_, _> = log
        .authorities
        .iter()
        .map(|binding| (binding.role, binding.sha256.as_str()))
        .collect();
    let mut added_roles = BTreeSet::new();
    for binding in &added_authorities {
        if !is_sha256(&binding.sha256) {
            issues.push(StudyOrchestrationIssue::InvalidDigest {
                field: format!("authority.{:?}", binding.role),
            });
        }
        if !added_roles.insert(binding.role) {
            issues.push(StudyOrchestrationIssue::DuplicateAuthority { role: binding.role });
        }
        if existing
            .get(&binding.role)
            .is_some_and(|digest| *digest != binding.sha256.as_str())
        {
            issues.push(StudyOrchestrationIssue::ConflictingAuthority { role: binding.role });
        }
        if log.current_phase >= StudyLifecyclePhase::ConfirmatoryFrozen
            && binding.role == StudyAuthorityRole::PilotAmendmentLedger
        {
            issues.push(StudyOrchestrationIssue::PilotAmendmentAfterConfirmatoryFreeze);
        }
    }
    let mut prospective = log.authorities.clone();
    prospective.extend(added_authorities.iter().cloned());
    deduplicate_authorities(&mut prospective);
    for role in required_authorities(&log.orchestration_version, to) {
        if !prospective.iter().any(|binding| binding.role == *role) {
            issues.push(StudyOrchestrationIssue::MissingAuthority {
                phase: to,
                role: *role,
            });
        }
    }
    if to >= StudyLifecyclePhase::Unblinded
        && log.current_phase < StudyLifecyclePhase::ConfirmatoryCollectionClosed
    {
        issues.push(StudyOrchestrationIssue::UnblindingBeforeCollectionClose);
    }
    if !issues.is_empty() {
        return Err(issues);
    }

    added_authorities.sort_by_key(|binding| binding.role);
    let previous_transition_sha256 = log
        .transitions
        .last()
        .map_or(ORCHESTRATION_GENESIS_SHA256, |transition| {
            transition.transition_sha256.as_str()
        })
        .to_string();
    let mut transition = StudyLifecycleTransition {
        sequence: log.transitions.len() as u32 + 1,
        from: log.current_phase,
        to,
        recorded_at_utc,
        operator_id,
        authorization_sha256,
        added_authorities,
        previous_transition_sha256,
        transition_sha256: String::new(),
    };
    transition.transition_sha256 = study_transition_commitment(&log.orchestration_id, &transition)
        .map_err(|_| {
            vec![StudyOrchestrationIssue::SerializationFailed {
                field: "transition".into(),
            }]
        })?;
    log.authorities = prospective;
    log.authorities.sort_by_key(|binding| binding.role);
    log.current_phase = to;
    log.transitions.push(transition);
    log.log_sha256 = study_orchestration_commitment(log).map_err(|_| {
        vec![StudyOrchestrationIssue::SerializationFailed {
            field: "orchestration_log".into(),
        }]
    })?;
    Ok(())
}

pub fn validate_study_orchestration(log: &StudyOrchestrationLog) -> Vec<StudyOrchestrationIssue> {
    let mut issues = Vec::new();
    if log.orchestration_version != STUDY_ORCHESTRATION_VERSION
        && log.orchestration_version != STUDY_ORCHESTRATION_VERSION_V2
        && log.orchestration_version != STUDY_ORCHESTRATION_VERSION_V1
    {
        issues.push(StudyOrchestrationIssue::WrongVersion {
            found: log.orchestration_version.clone(),
        });
    }
    if log.orchestration_id.trim().is_empty() {
        issues.push(StudyOrchestrationIssue::EmptyField {
            field: "orchestration_id".into(),
        });
    }
    let mut phase = StudyLifecyclePhase::Draft;
    let mut authorities = Vec::new();
    let mut previous = ORCHESTRATION_GENESIS_SHA256.to_string();
    for (index, transition) in log.transitions.iter().enumerate() {
        let expected = index as u32 + 1;
        if transition.sequence != expected {
            issues.push(StudyOrchestrationIssue::UnexpectedSequence {
                expected,
                found: transition.sequence,
            });
        }
        if transition.from != phase {
            issues.push(StudyOrchestrationIssue::TransitionFromMismatch {
                sequence: transition.sequence,
            });
        }
        if !legal_transition(&log.orchestration_version, transition.from, transition.to) {
            issues.push(StudyOrchestrationIssue::IllegalTransition {
                from: transition.from,
                to: transition.to,
            });
        }
        if transition.previous_transition_sha256 != previous {
            issues.push(StudyOrchestrationIssue::TransitionChainBroken {
                sequence: transition.sequence,
            });
        }
        match study_transition_commitment(&log.orchestration_id, transition) {
            Ok(value) if value == transition.transition_sha256 => {}
            Ok(_) => issues.push(StudyOrchestrationIssue::TransitionDigestMismatch {
                sequence: transition.sequence,
            }),
            Err(_) => issues.push(StudyOrchestrationIssue::SerializationFailed {
                field: format!("transition.{}", transition.sequence),
            }),
        }
        for binding in &transition.added_authorities {
            if authorities.iter().any(|existing: &StudyAuthorityBinding| {
                existing.role == binding.role && existing.sha256 != binding.sha256
            }) {
                issues.push(StudyOrchestrationIssue::ConflictingAuthority { role: binding.role });
            }
            authorities.push(binding.clone());
        }
        deduplicate_authorities(&mut authorities);
        for role in required_authorities(&log.orchestration_version, transition.to) {
            if !authorities.iter().any(|binding| binding.role == *role) {
                issues.push(StudyOrchestrationIssue::MissingAuthority {
                    phase: transition.to,
                    role: *role,
                });
            }
        }
        previous = transition.transition_sha256.clone();
        phase = transition.to;
    }
    authorities.sort_by_key(|binding| binding.role);
    let mut snapshot = log.authorities.clone();
    snapshot.sort_by_key(|binding| binding.role);
    if phase != log.current_phase {
        issues.push(StudyOrchestrationIssue::CurrentPhaseMismatch);
    }
    if authorities != snapshot {
        issues.push(StudyOrchestrationIssue::AuthoritySnapshotMismatch);
    }
    match study_orchestration_commitment(log) {
        Ok(value) if value == log.log_sha256 => {}
        Ok(_) => issues.push(StudyOrchestrationIssue::LogDigestMismatch),
        Err(_) => issues.push(StudyOrchestrationIssue::SerializationFailed {
            field: "orchestration_log".into(),
        }),
    }
    issues
}

fn legal_transition(version: &str, from: StudyLifecyclePhase, to: StudyLifecyclePhase) -> bool {
    if version == STUDY_ORCHESTRATION_VERSION_V1 {
        return matches!(
            (from, to),
            (
                StudyLifecyclePhase::Draft,
                StudyLifecyclePhase::PilotRegistered
            ) | (
                StudyLifecyclePhase::PilotRegistered,
                StudyLifecyclePhase::PilotArtifactsSealed
            ) | (
                StudyLifecyclePhase::PilotArtifactsSealed,
                StudyLifecyclePhase::PilotCollectionOpen
            ) | (
                StudyLifecyclePhase::PilotCollectionOpen,
                StudyLifecyclePhase::PilotCollectionClosed
            ) | (
                StudyLifecyclePhase::PilotCollectionClosed,
                StudyLifecyclePhase::PilotReviewed
            ) | (
                StudyLifecyclePhase::PilotReviewed,
                StudyLifecyclePhase::ConfirmatoryFrozen
            ) | (
                StudyLifecyclePhase::ConfirmatoryFrozen,
                StudyLifecyclePhase::ConfirmatoryArtifactsSealed
            ) | (
                StudyLifecyclePhase::ConfirmatoryArtifactsSealed,
                StudyLifecyclePhase::ConfirmatoryCollectionOpen
            ) | (
                StudyLifecyclePhase::ConfirmatoryCollectionOpen,
                StudyLifecyclePhase::ConfirmatoryCollectionClosed
            ) | (
                StudyLifecyclePhase::ConfirmatoryCollectionClosed,
                StudyLifecyclePhase::Unblinded
            ) | (
                StudyLifecyclePhase::Unblinded,
                StudyLifecyclePhase::Analyzed
            ) | (
                StudyLifecyclePhase::Analyzed,
                StudyLifecyclePhase::Published
            )
        );
    }
    matches!(
        (from, to),
        (
            StudyLifecyclePhase::Draft,
            StudyLifecyclePhase::PilotRegistered
        ) | (
            StudyLifecyclePhase::PilotRegistered,
            StudyLifecyclePhase::PilotArtifactsSealed
        ) | (
            StudyLifecyclePhase::PilotArtifactsSealed,
            StudyLifecyclePhase::PilotCollectionOpen
        ) | (
            StudyLifecyclePhase::PilotCollectionOpen,
            StudyLifecyclePhase::PilotCollectionClosed
        ) | (
            StudyLifecyclePhase::PilotCollectionClosed,
            StudyLifecyclePhase::PilotReviewed
        ) | (
            StudyLifecyclePhase::PilotReviewed,
            StudyLifecyclePhase::ExternalReviewOpen
        ) | (
            StudyLifecyclePhase::ExternalReviewOpen,
            StudyLifecyclePhase::ExternalReviewComplete
        ) | (
            StudyLifecyclePhase::ExternalReviewComplete,
            StudyLifecyclePhase::ConfirmatoryReady
        ) | (
            StudyLifecyclePhase::ConfirmatoryReady,
            StudyLifecyclePhase::ConfirmatoryFrozen
        ) | (
            StudyLifecyclePhase::ConfirmatoryFrozen,
            StudyLifecyclePhase::ConfirmatoryArtifactsSealed
        ) | (
            StudyLifecyclePhase::ConfirmatoryArtifactsSealed,
            StudyLifecyclePhase::ConfirmatoryCollectionOpen
        ) | (
            StudyLifecyclePhase::ConfirmatoryCollectionOpen,
            StudyLifecyclePhase::ConfirmatoryCollectionClosed
        ) | (
            StudyLifecyclePhase::ConfirmatoryCollectionClosed,
            StudyLifecyclePhase::Unblinded
        ) | (
            StudyLifecyclePhase::Unblinded,
            StudyLifecyclePhase::Analyzed
        ) | (
            StudyLifecyclePhase::Analyzed,
            StudyLifecyclePhase::Published
        )
    )
}

fn required_authorities(
    version: &str,
    phase: StudyLifecyclePhase,
) -> &'static [StudyAuthorityRole] {
    use StudyAuthorityRole as R;
    if version == STUDY_ORCHESTRATION_VERSION_V1 {
        return match phase {
            StudyLifecyclePhase::Draft => &[],
            StudyLifecyclePhase::PilotRegistered => &[
                R::FrozenManifest,
                R::FrozenMethodology,
                R::PilotProtocol,
                R::PilotRandomizationCommitment,
                R::ExternalPilotReceipt,
            ],
            StudyLifecyclePhase::PilotArtifactsSealed => {
                &[R::PilotArtifactBundle, R::PilotParticipantSchedule]
            }
            StudyLifecyclePhase::PilotCollectionOpen => &[R::PilotCohortRegistry],
            StudyLifecyclePhase::PilotCollectionClosed => &[R::PilotCollection],
            StudyLifecyclePhase::PilotReviewed => &[R::PilotAmendmentLedger, R::PilotReport],
            StudyLifecyclePhase::ConfirmatoryFrozen => &[R::ConfirmatoryPreregistration],
            StudyLifecyclePhase::ConfirmatoryArtifactsSealed => &[
                R::ConfirmatoryArtifactBundle,
                R::ConfirmatoryParticipantSchedule,
            ],
            StudyLifecyclePhase::ConfirmatoryCollectionOpen => &[R::ConfirmatoryCohortRegistry],
            StudyLifecyclePhase::ConfirmatoryCollectionClosed => &[R::ConfirmatoryCollection],
            StudyLifecyclePhase::Unblinded => &[R::BlindingCodebook, R::RandomizationKeyReveal],
            StudyLifecyclePhase::Analyzed => &[
                R::PrimaryAnalysisReport,
                R::IndependentAnalysisReport,
                R::ReproducibilityAttestation,
            ],
            StudyLifecyclePhase::Published => &[R::StudyReleaseBundle],
            StudyLifecyclePhase::ExternalReviewOpen
            | StudyLifecyclePhase::ExternalReviewComplete
            | StudyLifecyclePhase::ConfirmatoryReady => &[],
        };
    }
    if version == STUDY_ORCHESTRATION_VERSION_V2 {
        return match phase {
            StudyLifecyclePhase::Draft => &[],
            StudyLifecyclePhase::PilotRegistered => &[
                R::FrozenManifest,
                R::FrozenMethodology,
                R::PilotProtocol,
                R::PilotRandomizationCommitment,
                R::ExternalPilotReceipt,
            ],
            StudyLifecyclePhase::PilotArtifactsSealed => {
                &[R::PilotArtifactBundle, R::PilotParticipantSchedule]
            }
            StudyLifecyclePhase::PilotCollectionOpen => &[R::PilotCohortRegistry],
            StudyLifecyclePhase::PilotCollectionClosed => &[R::PilotCollection],
            StudyLifecyclePhase::PilotReviewed => &[R::PilotAmendmentLedger, R::PilotReport],
            StudyLifecyclePhase::ExternalReviewOpen => {
                &[R::ExternalReviewProtocol, R::ExternalReviewEvidenceIndex]
            }
            StudyLifecyclePhase::ExternalReviewComplete => {
                &[R::ExternalReviewCompletion, R::ConfirmatoryAmendmentLedger]
            }
            StudyLifecyclePhase::ConfirmatoryReady => &[
                R::ConfirmatoryAuthoritySnapshot,
                R::WorkspaceValidation,
                R::HumanStudyGovernance,
                R::ConfirmatoryDryRun,
                R::IndependentReproductionReadiness,
                R::ConfirmatoryReadinessReport,
                R::ConfirmatoryReadinessRelease,
                R::ConfirmatoryPreregistration,
            ],
            StudyLifecyclePhase::ConfirmatoryFrozen => &[
                R::ConfirmatoryAuthoritySnapshot,
                R::ConfirmatoryPreregistration,
                R::ConfirmatoryReadinessReport,
                R::ConfirmatoryReadinessRelease,
            ],
            StudyLifecyclePhase::ConfirmatoryArtifactsSealed => &[
                R::ConfirmatoryArtifactBundle,
                R::ConfirmatoryParticipantSchedule,
            ],
            StudyLifecyclePhase::ConfirmatoryCollectionOpen => &[R::ConfirmatoryCohortRegistry],
            StudyLifecyclePhase::ConfirmatoryCollectionClosed => &[R::ConfirmatoryCollection],
            StudyLifecyclePhase::Unblinded => &[R::BlindingCodebook, R::RandomizationKeyReveal],
            StudyLifecyclePhase::Analyzed => &[
                R::PrimaryAnalysisReport,
                R::IndependentAnalysisReport,
                R::ReproducibilityAttestation,
            ],
            StudyLifecyclePhase::Published => &[R::StudyReleaseBundle],
        };
    }
    match phase {
        StudyLifecyclePhase::Draft => &[],
        StudyLifecyclePhase::PilotRegistered => &[
            R::FrozenManifest,
            R::FrozenMethodology,
            R::PilotProtocol,
            R::PilotRandomizationCommitment,
            R::ExternalPilotReceipt,
        ],
        StudyLifecyclePhase::PilotArtifactsSealed => {
            &[R::PilotArtifactBundle, R::PilotParticipantSchedule]
        }
        StudyLifecyclePhase::PilotCollectionOpen => &[R::PilotCohortRegistry],
        StudyLifecyclePhase::PilotCollectionClosed => &[R::PilotCollection],
        StudyLifecyclePhase::PilotReviewed => &[R::PilotAmendmentLedger, R::PilotReport],
        StudyLifecyclePhase::ExternalReviewOpen => {
            &[R::ExternalReviewProtocol, R::ExternalReviewEvidenceIndex]
        }
        StudyLifecyclePhase::ExternalReviewComplete => {
            &[R::ExternalReviewCompletion, R::ConfirmatoryAmendmentLedger]
        }
        StudyLifecyclePhase::ConfirmatoryReady => &[
            R::ConfirmatoryAuthoritySnapshot,
            R::WorkspaceValidation,
            R::HumanStudyGovernance,
            R::ConfirmatoryDryRun,
            R::IndependentReproductionReadiness,
            R::ConfirmatoryReadinessReport,
            R::ConfirmatoryReadinessRelease,
            R::ConfirmatoryPreregistration,
        ],
        StudyLifecyclePhase::ConfirmatoryFrozen => &[
            R::ConfirmatoryAuthoritySnapshot,
            R::ConfirmatoryPreregistration,
            R::ConfirmatoryReadinessReport,
            R::ConfirmatoryReadinessRelease,
        ],
        StudyLifecyclePhase::ConfirmatoryArtifactsSealed => &[
            R::ConfirmatoryArtifactBundle,
            R::ConfirmatoryParticipantSchedule,
        ],
        StudyLifecyclePhase::ConfirmatoryCollectionOpen => &[
            R::ConfirmatoryCohortRegistry,
            R::ConfirmatoryCollectionProtocol,
        ],
        StudyLifecyclePhase::ConfirmatoryCollectionClosed => &[
            R::ConfirmatoryCollection,
            R::ConfirmatoryCollectionMonitor,
            R::ConfirmatoryCollectionCloseReceipt,
        ],
        StudyLifecyclePhase::Unblinded => &[
            R::BlindingCodebook,
            R::RandomizationKeyReveal,
            R::ConfirmatoryCollectionCloseReceipt,
            R::ConfirmatoryUnblindingReceipt,
        ],
        StudyLifecyclePhase::Analyzed => &[
            R::PrimaryAnalysisReport,
            R::IndependentAnalysisReport,
            R::ReproducibilityAttestation,
            R::ConfirmatoryAnalysisExecution,
        ],
        StudyLifecyclePhase::Published => &[
            R::StudyReleaseBundle,
            R::ConfirmatoryPublicationRecord,
            R::PostPublicationAuditLedger,
        ],
    }
}

fn deduplicate_authorities(authorities: &mut Vec<StudyAuthorityBinding>) {
    authorities.sort_by_key(|binding| binding.role);
    authorities.dedup_by(|left, right| left.role == right.role && left.sha256 == right.sha256);
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding(role: StudyAuthorityRole) -> StudyAuthorityBinding {
        StudyAuthorityBinding {
            role,
            sha256: format!("{:064x}", role as u8 + 1),
            external_uri: None,
        }
    }

    #[test]
    fn pilot_cannot_open_before_authorities_exist() {
        let mut log = new_study_orchestration("study-1");
        let result = append_study_transition(
            &mut log,
            StudyLifecyclePhase::PilotCollectionOpen,
            "2026-07-14T00:00:00Z".into(),
            "operator".into(),
            "a".repeat(64),
            vec![binding(StudyAuthorityRole::PilotCohortRegistry)],
        );
        assert!(result.is_err());
        assert_eq!(log.current_phase, StudyLifecyclePhase::Draft);
    }

    #[test]
    fn unblinding_has_no_early_transition() {
        assert!(!legal_transition(
            STUDY_ORCHESTRATION_VERSION,
            StudyLifecyclePhase::ConfirmatoryCollectionOpen,
            StudyLifecyclePhase::Unblinded
        ));
    }
    #[test]
    fn v2_requires_external_review_before_confirmatory_freeze() {
        assert!(!legal_transition(
            STUDY_ORCHESTRATION_VERSION,
            StudyLifecyclePhase::PilotReviewed,
            StudyLifecyclePhase::ConfirmatoryFrozen,
        ));
        assert!(legal_transition(
            STUDY_ORCHESTRATION_VERSION,
            StudyLifecyclePhase::PilotReviewed,
            StudyLifecyclePhase::ExternalReviewOpen,
        ));
    }

    #[test]
    fn legacy_v1_transition_remains_verifiable() {
        assert!(legal_transition(
            STUDY_ORCHESTRATION_VERSION_V1,
            StudyLifecyclePhase::PilotReviewed,
            StudyLifecyclePhase::ConfirmatoryFrozen,
        ));
    }

    #[test]
    fn legacy_logs_are_read_only_until_upgraded() {
        let mut log = new_study_orchestration("legacy");
        log.orchestration_version = STUDY_ORCHESTRATION_VERSION_V1.into();
        log.log_sha256 = study_orchestration_commitment(&log).unwrap();
        let result = append_study_transition(
            &mut log,
            StudyLifecyclePhase::PilotRegistered,
            "2026-07-14T00:00:00Z".into(),
            "operator".into(),
            "a".repeat(64),
            vec![],
        );
        assert_eq!(
            result,
            Err(vec![StudyOrchestrationIssue::LegacyOrchestrationReadOnly])
        );
    }

    #[test]
    fn preconfirmatory_legacy_log_can_upgrade() {
        let mut log = new_study_orchestration("legacy");
        log.orchestration_version = STUDY_ORCHESTRATION_VERSION_V1.into();
        log.log_sha256 = study_orchestration_commitment(&log).unwrap();
        upgrade_legacy_study_orchestration(&mut log).unwrap();
        assert_eq!(log.orchestration_version, STUDY_ORCHESTRATION_VERSION);
        assert!(validate_study_orchestration(&log).is_empty());
    }

    #[test]
    fn legacy_v2_log_is_read_only_until_upgraded() {
        let mut log = new_study_orchestration("legacy-v2");
        log.orchestration_version = STUDY_ORCHESTRATION_VERSION_V2.into();
        log.log_sha256 = study_orchestration_commitment(&log).unwrap();
        let result = append_study_transition(
            &mut log,
            StudyLifecyclePhase::PilotRegistered,
            "2026-07-14T00:00:00Z".into(),
            "operator".into(),
            "a".repeat(64),
            vec![],
        );
        assert_eq!(
            result,
            Err(vec![StudyOrchestrationIssue::LegacyOrchestrationReadOnly])
        );
        upgrade_legacy_study_orchestration(&mut log).unwrap();
        assert_eq!(log.orchestration_version, STUDY_ORCHESTRATION_VERSION);
    }

    #[test]
    fn v3_collection_open_requires_protocol_and_registry() {
        let required = required_authorities(
            STUDY_ORCHESTRATION_VERSION,
            StudyLifecyclePhase::ConfirmatoryCollectionOpen,
        );
        assert!(required.contains(&StudyAuthorityRole::ConfirmatoryCollectionProtocol));
        assert!(required.contains(&StudyAuthorityRole::ConfirmatoryCohortRegistry));
    }

    #[test]
    fn v3_unblinding_requires_close_and_unblinding_receipts() {
        let required =
            required_authorities(STUDY_ORCHESTRATION_VERSION, StudyLifecyclePhase::Unblinded);
        assert!(required.contains(&StudyAuthorityRole::ConfirmatoryCollectionCloseReceipt));
        assert!(required.contains(&StudyAuthorityRole::ConfirmatoryUnblindingReceipt));
    }
}
