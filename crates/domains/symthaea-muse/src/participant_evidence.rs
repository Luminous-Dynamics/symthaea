// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bind raw listener responses to their assigned participant-specific order.

use crate::blinded_study::{BlindedSchedule, BlindingCodebook};
use crate::evidence_digest::canonical_json_sha256;
use crate::experiment_manifest::FrozenStudyManifest;
use crate::participant_schedule::{
    ParticipantCohortSpec, ParticipantScheduleBook, ParticipantScheduleIssue,
    validate_participant_schedule,
};
use crate::study_evidence::{
    CompiledStudyDataset, EvidenceBlockStatus, RawStudyEvidence, StudyEvidenceIssue,
    compile_study_dataset, raw_evidence_commitment, validate_raw_study_evidence,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const PARTICIPANT_EVIDENCE_VERSION: &str = "symthaea-muse-participant-evidence-v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParticipantEvidenceEnvelope {
    pub evidence_version: String,
    pub manifest_sha256: String,
    pub base_schedule_sha256: String,
    pub participant_schedule_sha256: String,
    pub raw_evidence_sha256: String,
    pub envelope_sha256: String,
    pub evidence: RawStudyEvidence,
}

#[derive(Serialize)]
struct ParticipantEvidenceCommitment<'a> {
    evidence_version: &'a str,
    manifest_sha256: &'a str,
    base_schedule_sha256: &'a str,
    participant_schedule_sha256: &'a str,
    raw_evidence_sha256: &'a str,
    evidence: &'a RawStudyEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParticipantEvidenceIssue {
    WrongVersion { found: String },
    ParticipantSchedule { issue: ParticipantScheduleIssue },
    RawEvidence { issue: StudyEvidenceIssue },
    SerializationFailed { field: String },
    DigestMismatch { field: String },
    UnknownAssignedBlock { block_id: String },
    ParticipantMismatch { block_id: String },
    FixtureMismatch { block_id: String },
    PresentationOrderMismatch { block_id: String },
    DuplicateIncludedAssignment { block_id: String },
    MissingAssignment { block_id: String },
}

pub fn participant_evidence_commitment(
    envelope: &ParticipantEvidenceEnvelope,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&ParticipantEvidenceCommitment {
        evidence_version: &envelope.evidence_version,
        manifest_sha256: &envelope.manifest_sha256,
        base_schedule_sha256: &envelope.base_schedule_sha256,
        participant_schedule_sha256: &envelope.participant_schedule_sha256,
        raw_evidence_sha256: &envelope.raw_evidence_sha256,
        evidence: &envelope.evidence,
    })
}

pub fn seal_participant_evidence(
    manifest: &FrozenStudyManifest,
    base_schedule: &BlindedSchedule,
    participant_schedule: &ParticipantScheduleBook,
    evidence: RawStudyEvidence,
) -> Result<ParticipantEvidenceEnvelope, serde_json::Error> {
    let mut envelope = ParticipantEvidenceEnvelope {
        evidence_version: PARTICIPANT_EVIDENCE_VERSION.into(),
        manifest_sha256: canonical_json_sha256(manifest)?,
        base_schedule_sha256: canonical_json_sha256(base_schedule)?,
        participant_schedule_sha256: canonical_json_sha256(participant_schedule)?,
        raw_evidence_sha256: raw_evidence_commitment(&evidence)?,
        envelope_sha256: String::new(),
        evidence,
    };
    envelope.envelope_sha256 = participant_evidence_commitment(&envelope)?;
    Ok(envelope)
}

pub fn validate_participant_evidence(
    manifest: &FrozenStudyManifest,
    base_schedule: &BlindedSchedule,
    codebook: &BlindingCodebook,
    cohort: &ParticipantCohortSpec,
    participant_schedule: &ParticipantScheduleBook,
    envelope: &ParticipantEvidenceEnvelope,
) -> Vec<ParticipantEvidenceIssue> {
    let mut issues: Vec<_> =
        validate_participant_schedule(manifest, base_schedule, cohort, participant_schedule, None)
            .into_iter()
            .map(|issue| ParticipantEvidenceIssue::ParticipantSchedule { issue })
            .collect();
    issues.extend(
        validate_raw_study_evidence(manifest, base_schedule, codebook, &envelope.evidence)
            .into_iter()
            .map(|issue| ParticipantEvidenceIssue::RawEvidence { issue }),
    );
    if envelope.evidence_version != PARTICIPANT_EVIDENCE_VERSION {
        issues.push(ParticipantEvidenceIssue::WrongVersion {
            found: envelope.evidence_version.clone(),
        });
    }
    verify_digest(
        "manifest_sha256",
        canonical_json_sha256(manifest),
        &envelope.manifest_sha256,
        &mut issues,
    );
    verify_digest(
        "base_schedule_sha256",
        canonical_json_sha256(base_schedule),
        &envelope.base_schedule_sha256,
        &mut issues,
    );
    verify_digest(
        "participant_schedule_sha256",
        canonical_json_sha256(participant_schedule),
        &envelope.participant_schedule_sha256,
        &mut issues,
    );
    verify_digest(
        "raw_evidence_sha256",
        raw_evidence_commitment(&envelope.evidence),
        &envelope.raw_evidence_sha256,
        &mut issues,
    );
    verify_digest(
        "envelope_sha256",
        participant_evidence_commitment(envelope),
        &envelope.envelope_sha256,
        &mut issues,
    );

    let assignments: BTreeMap<_, _> = participant_schedule
        .blocks
        .iter()
        .map(|block| (block.block_id.as_str(), block))
        .collect();
    let mut observed_assignments = BTreeSet::new();
    for block in &envelope.evidence.listener_blocks {
        let Some(assignment) = assignments.get(block.block_id.as_str()) else {
            issues.push(ParticipantEvidenceIssue::UnknownAssignedBlock {
                block_id: block.block_id.clone(),
            });
            continue;
        };
        if !observed_assignments.insert(block.block_id.clone()) {
            issues.push(ParticipantEvidenceIssue::DuplicateIncludedAssignment {
                block_id: block.block_id.clone(),
            });
        }
        if block.listener_id != assignment.participant_token {
            issues.push(ParticipantEvidenceIssue::ParticipantMismatch {
                block_id: block.block_id.clone(),
            });
        }
        if block.key != assignment.key {
            issues.push(ParticipantEvidenceIssue::FixtureMismatch {
                block_id: block.block_id.clone(),
            });
        }
        let response_order: Vec<_> = block
            .responses
            .iter()
            .map(|response| response.presentation_id.as_str())
            .collect();
        let assigned_order: Vec<_> = assignment
            .ordered_presentation_ids
            .iter()
            .map(String::as_str)
            .collect();
        let order_valid = match &block.status {
            EvidenceBlockStatus::Included => response_order == assigned_order,
            EvidenceBlockStatus::Excluded { .. } => {
                assigned_order
                    .iter()
                    .filter(|presentation_id| response_order.contains(presentation_id))
                    .copied()
                    .collect::<Vec<_>>()
                    == response_order
            }
        };
        if !order_valid {
            issues.push(ParticipantEvidenceIssue::PresentationOrderMismatch {
                block_id: block.block_id.clone(),
            });
        }
    }
    for assignment in &participant_schedule.blocks {
        if !observed_assignments.contains(&assignment.block_id) {
            issues.push(ParticipantEvidenceIssue::MissingAssignment {
                block_id: assignment.block_id.clone(),
            });
        }
    }
    issues
}

pub fn compile_participant_dataset(
    manifest: &FrozenStudyManifest,
    base_schedule: &BlindedSchedule,
    codebook: &BlindingCodebook,
    cohort: &ParticipantCohortSpec,
    participant_schedule: &ParticipantScheduleBook,
    envelope: &ParticipantEvidenceEnvelope,
) -> Result<CompiledStudyDataset, Vec<ParticipantEvidenceIssue>> {
    let issues = validate_participant_evidence(
        manifest,
        base_schedule,
        codebook,
        cohort,
        participant_schedule,
        envelope,
    );
    if !issues.is_empty() {
        return Err(issues);
    }
    compile_study_dataset(manifest, base_schedule, codebook, &envelope.evidence).map_err(|found| {
        found
            .into_iter()
            .map(|issue| ParticipantEvidenceIssue::RawEvidence { issue })
            .collect()
    })
}

fn verify_digest(
    field: &str,
    result: Result<String, serde_json::Error>,
    expected: &str,
    issues: &mut Vec<ParticipantEvidenceIssue>,
) {
    match result {
        Ok(value) if value == expected => {}
        Ok(_) => issues.push(ParticipantEvidenceIssue::DigestMismatch {
            field: field.into(),
        }),
        Err(_) => issues.push(ParticipantEvidenceIssue::SerializationFailed {
            field: field.into(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIGEST: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    #[test]
    fn envelope_commitment_detects_response_tampering() {
        let evidence = RawStudyEvidence {
            manifest_sha256: DIGEST.into(),
            schedule_sha256: DIGEST.into(),
            raw_evidence_sha256: DIGEST.into(),
            structural: Vec::new(),
            listener_blocks: Vec::new(),
            workflow_blocks: Vec::new(),
        };
        let mut envelope = ParticipantEvidenceEnvelope {
            evidence_version: PARTICIPANT_EVIDENCE_VERSION.into(),
            manifest_sha256: DIGEST.into(),
            base_schedule_sha256: DIGEST.into(),
            participant_schedule_sha256: DIGEST.into(),
            raw_evidence_sha256: DIGEST.into(),
            envelope_sha256: String::new(),
            evidence,
        };
        envelope.envelope_sha256 = participant_evidence_commitment(&envelope).unwrap();
        let sealed = envelope.envelope_sha256.clone();
        envelope.evidence.raw_evidence_sha256 = "b".repeat(64);
        assert_ne!(participant_evidence_commitment(&envelope).unwrap(), sealed);
    }
}
