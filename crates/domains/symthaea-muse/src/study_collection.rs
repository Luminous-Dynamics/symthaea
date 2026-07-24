// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic compilation of runner sessions into sealed participant evidence.

use crate::blinded_study::BlindedSchedule;
use crate::evidence_digest::canonical_json_sha256;
use crate::experiment_manifest::FrozenStudyManifest;
use crate::participant_evidence::{ParticipantEvidenceEnvelope, seal_participant_evidence};
use crate::participant_schedule::ParticipantScheduleBook;
use crate::study_artifact::StudyArtifactBundle;
use crate::study_evidence::{
    ArtistWorkflowBlock, RawStudyEvidence, StructuralPresentationOutcome, seal_raw_evidence,
};
use crate::study_runner::{
    StudyRunnerIssue, StudyRunnerPackage, StudySessionLog, compile_listener_block,
    validate_runner_package,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const STUDY_COLLECTION_VERSION: &str = "symthaea-muse-runner-collection-v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunnerSessionSubmission {
    pub package: StudyRunnerPackage,
    pub log: StudySessionLog,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StudyCollectionDraft {
    pub collection_version: String,
    pub artifact_bundle_sha256: String,
    pub structural: Vec<StructuralPresentationOutcome>,
    pub workflow_blocks: Vec<ArtistWorkflowBlock>,
    pub sessions: Vec<RunnerSessionSubmission>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StudyCollectionIssue {
    WrongVersion {
        found: String,
    },
    ArtifactBundleMismatch,
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
    DuplicateStructuralPresentation {
        presentation_id: String,
    },
    UnknownStructuralPresentation {
        presentation_id: String,
    },
    MissingStructuralPresentation {
        presentation_id: String,
    },
    SerializationFailed {
        field: String,
    },
}

pub fn seal_runner_collection(
    manifest: &FrozenStudyManifest,
    schedule: &BlindedSchedule,
    participant_schedule: &ParticipantScheduleBook,
    artifacts: &StudyArtifactBundle,
    draft: &StudyCollectionDraft,
) -> Result<ParticipantEvidenceEnvelope, Vec<StudyCollectionIssue>> {
    let mut issues = Vec::new();
    if draft.collection_version != STUDY_COLLECTION_VERSION {
        issues.push(StudyCollectionIssue::WrongVersion {
            found: draft.collection_version.clone(),
        });
    }
    if draft.artifact_bundle_sha256 != artifacts.bundle_sha256 {
        issues.push(StudyCollectionIssue::ArtifactBundleMismatch);
    }

    let assignments: BTreeSet<_> = participant_schedule
        .blocks
        .iter()
        .map(|block| block.block_id.as_str())
        .collect();
    let mut sessions = BTreeMap::new();
    for submission in &draft.sessions {
        let block_id = submission.package.block_id.clone();
        if !assignments.contains(block_id.as_str()) {
            issues.push(StudyCollectionIssue::UnknownSession {
                block_id: block_id.clone(),
            });
        }
        if sessions.insert(block_id.clone(), submission).is_some() {
            issues.push(StudyCollectionIssue::DuplicateSession { block_id });
        }
    }
    for block in &participant_schedule.blocks {
        if !sessions.contains_key(&block.block_id) {
            issues.push(StudyCollectionIssue::MissingSession {
                block_id: block.block_id.clone(),
            });
        }
    }

    let schedule_presentations: BTreeSet<_> = schedule
        .presentations
        .iter()
        .map(|presentation| presentation.presentation_id.as_str())
        .collect();
    let mut structural_seen = BTreeSet::new();
    for outcome in &draft.structural {
        if !structural_seen.insert(outcome.presentation_id.as_str()) {
            issues.push(StudyCollectionIssue::DuplicateStructuralPresentation {
                presentation_id: outcome.presentation_id.clone(),
            });
        }
        if !schedule_presentations.contains(outcome.presentation_id.as_str()) {
            issues.push(StudyCollectionIssue::UnknownStructuralPresentation {
                presentation_id: outcome.presentation_id.clone(),
            });
        }
    }
    for presentation in &schedule.presentations {
        if !structural_seen.contains(presentation.presentation_id.as_str()) {
            issues.push(StudyCollectionIssue::MissingStructuralPresentation {
                presentation_id: presentation.presentation_id.clone(),
            });
        }
    }

    let mut listener_blocks = Vec::with_capacity(draft.sessions.len());
    for (block_id, submission) in sessions {
        let package_issues = validate_runner_package(
            &submission.package,
            schedule,
            participant_schedule,
            artifacts,
        );
        if !package_issues.is_empty() {
            issues.push(StudyCollectionIssue::RunnerPackage {
                block_id: block_id.clone(),
                issues: package_issues,
            });
            continue;
        }
        match compile_listener_block(&submission.package, &submission.log) {
            Ok(block) => listener_blocks.push(block),
            Err(found) => issues.push(StudyCollectionIssue::RunnerSession {
                block_id,
                issues: found,
            }),
        }
    }
    if !issues.is_empty() {
        return Err(issues);
    }
    listener_blocks.sort_by(|left, right| left.block_id.cmp(&right.block_id));

    let mut evidence = RawStudyEvidence {
        manifest_sha256: canonical_json_sha256(manifest).map_err(|_| {
            vec![StudyCollectionIssue::SerializationFailed {
                field: "manifest".into(),
            }]
        })?,
        schedule_sha256: canonical_json_sha256(schedule).map_err(|_| {
            vec![StudyCollectionIssue::SerializationFailed {
                field: "schedule".into(),
            }]
        })?,
        raw_evidence_sha256: String::new(),
        structural: draft.structural.clone(),
        listener_blocks,
        workflow_blocks: draft.workflow_blocks.clone(),
    };
    seal_raw_evidence(&mut evidence).map_err(|_| {
        vec![StudyCollectionIssue::SerializationFailed {
            field: "raw_evidence".into(),
        }]
    })?;
    seal_participant_evidence(manifest, schedule, participant_schedule, evidence).map_err(|_| {
        vec![StudyCollectionIssue::SerializationFailed {
            field: "participant_envelope".into(),
        }]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collection_version_is_explicit() {
        let draft = StudyCollectionDraft {
            collection_version: STUDY_COLLECTION_VERSION.into(),
            artifact_bundle_sha256: "a".repeat(64),
            structural: Vec::new(),
            workflow_blocks: Vec::new(),
            sessions: Vec::new(),
        };
        assert_eq!(draft.collection_version, STUDY_COLLECTION_VERSION);
    }
}
