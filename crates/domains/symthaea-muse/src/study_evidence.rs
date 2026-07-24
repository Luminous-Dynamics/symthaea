// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Raw blinded evidence and deterministic compilation into experiment records.
//!
//! Listener and workflow inputs refer only to anonymous presentation IDs. Arm
//! labels enter only during compilation with the private codebook.

use crate::blinded_study::{
    BlindedSchedule, BlindingCodebook, BlindingIssue, validate_blinded_schedule,
};
use crate::cognitive_experiment::{
    CognitiveTrialRecord, FrozenTrialKey, PerceptualTrialOutcome, StructuralTrialOutcome,
    WorkflowTrialOutcome,
};
use crate::evidence_digest::canonical_json_sha256;
use crate::experiment_manifest::FrozenStudyManifest;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PreregisteredExclusion {
    FailedAttentionCheck,
    TechnicalPlaybackFailure,
    IncompleteBlock,
    DuplicateParticipation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceBlockStatus {
    Included,
    Excluded { reason: PreregisteredExclusion },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ListenerPresentationResponse {
    pub presentation_id: String,
    pub return_recognized: bool,
    pub development_instability: f32,
    pub earned_recapitulation: f32,
    /// One is most preferred and four is least preferred within the block.
    pub preference_rank: u8,
    pub playback_completed: bool,
    pub attention_check_passed: bool,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ListenerResponseBlock {
    pub block_id: String,
    pub listener_id: String,
    pub key: FrozenTrialKey,
    pub status: EvidenceBlockStatus,
    pub responses: Vec<ListenerPresentationResponse>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArtistDisposition {
    Kept,
    Edited,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtistPresentationResponse {
    pub presentation_id: String,
    pub disposition: ArtistDisposition,
    pub time_to_commit_seconds: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtistWorkflowBlock {
    pub block_id: String,
    pub artist_id: String,
    pub key: FrozenTrialKey,
    pub status: EvidenceBlockStatus,
    pub responses: Vec<ArtistPresentationResponse>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuralPresentationOutcome {
    pub presentation_id: String,
    pub outcome: StructuralTrialOutcome,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawStudyEvidence {
    pub manifest_sha256: String,
    pub schedule_sha256: String,
    pub raw_evidence_sha256: String,
    pub structural: Vec<StructuralPresentationOutcome>,
    pub listener_blocks: Vec<ListenerResponseBlock>,
    pub workflow_blocks: Vec<ArtistWorkflowBlock>,
}

#[derive(Serialize)]
struct RawEvidenceCommitment<'a> {
    manifest_sha256: &'a str,
    schedule_sha256: &'a str,
    structural: &'a [StructuralPresentationOutcome],
    listener_blocks: &'a [ListenerResponseBlock],
    workflow_blocks: &'a [ArtistWorkflowBlock],
}

pub fn raw_evidence_commitment(evidence: &RawStudyEvidence) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&RawEvidenceCommitment {
        manifest_sha256: &evidence.manifest_sha256,
        schedule_sha256: &evidence.schedule_sha256,
        structural: &evidence.structural,
        listener_blocks: &evidence.listener_blocks,
        workflow_blocks: &evidence.workflow_blocks,
    })
}

pub fn seal_raw_evidence(evidence: &mut RawStudyEvidence) -> Result<(), serde_json::Error> {
    evidence.raw_evidence_sha256 = raw_evidence_commitment(evidence)?;
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompiledStudyDataset {
    pub manifest_sha256: String,
    pub schedule_sha256: String,
    pub codebook_sha256: String,
    pub raw_evidence_sha256: String,
    pub included_listener_blocks: usize,
    pub excluded_listener_blocks: usize,
    pub included_workflow_blocks: usize,
    pub excluded_workflow_blocks: usize,
    pub records: Vec<CognitiveTrialRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StudyEvidenceIssue {
    Blinding {
        issue: BlindingIssue,
    },
    InvalidDigest {
        field: String,
    },
    RawEvidenceDigestMismatch,
    ManifestDigestMismatch,
    ScheduleDigestMismatch,
    EvidenceSerializationFailed {
        field: String,
    },
    EmptyBlockId {
        kind: String,
        index: usize,
    },
    DuplicateBlockId {
        block_id: String,
    },
    EmptyParticipantId {
        block_id: String,
    },
    UnknownFixture {
        key: FrozenTrialKey,
    },
    DuplicateParticipantFixture {
        participant_id: String,
        key: FrozenTrialKey,
    },
    MultipleIncludedWorkflowBlocks {
        key: FrozenTrialKey,
    },
    UnknownPresentation {
        presentation_id: String,
    },
    PresentationFixtureMismatch {
        block_id: String,
        presentation_id: String,
    },
    DuplicatePresentationInBlock {
        block_id: String,
        presentation_id: String,
    },
    IncompleteIncludedBlock {
        block_id: String,
        found: usize,
    },
    InvalidPreferenceRank {
        block_id: String,
        presentation_id: String,
    },
    DuplicatePreferenceRank {
        block_id: String,
        rank: u8,
    },
    InvalidListenerRating {
        block_id: String,
        presentation_id: String,
        field: String,
    },
    IncompletePlayback {
        block_id: String,
        presentation_id: String,
    },
    FailedIncludedAttentionCheck {
        block_id: String,
        presentation_id: String,
    },
    ZeroResponseTime {
        block_id: String,
        presentation_id: String,
    },
    ZeroWorkflowTime {
        block_id: String,
        presentation_id: String,
    },
    MissingStructuralOutcome {
        presentation_id: String,
    },
    DuplicateStructuralOutcome {
        presentation_id: String,
    },
    InvalidStructuralOutcome {
        presentation_id: String,
    },
}

pub fn validate_raw_study_evidence(
    manifest: &FrozenStudyManifest,
    schedule: &BlindedSchedule,
    codebook: &BlindingCodebook,
    evidence: &RawStudyEvidence,
) -> Vec<StudyEvidenceIssue> {
    let mut issues: Vec<StudyEvidenceIssue> =
        validate_blinded_schedule(manifest, schedule, Some(codebook))
            .into_iter()
            .map(|issue| StudyEvidenceIssue::Blinding { issue })
            .collect();
    for (field, value) in [
        ("manifest_sha256", &evidence.manifest_sha256),
        ("schedule_sha256", &evidence.schedule_sha256),
        ("raw_evidence_sha256", &evidence.raw_evidence_sha256),
    ] {
        if !is_sha256(value) {
            issues.push(StudyEvidenceIssue::InvalidDigest {
                field: field.into(),
            });
        }
    }
    match raw_evidence_commitment(evidence) {
        Ok(digest) if digest == evidence.raw_evidence_sha256 => {}
        Ok(_) => issues.push(StudyEvidenceIssue::RawEvidenceDigestMismatch),
        Err(_) => issues.push(StudyEvidenceIssue::EvidenceSerializationFailed {
            field: "raw_evidence".into(),
        }),
    }
    match canonical_json_sha256(manifest) {
        Ok(digest) if digest == evidence.manifest_sha256 && digest == schedule.manifest_sha256 => {}
        Ok(_) => issues.push(StudyEvidenceIssue::ManifestDigestMismatch),
        Err(_) => issues.push(StudyEvidenceIssue::EvidenceSerializationFailed {
            field: "manifest".into(),
        }),
    }
    match canonical_json_sha256(schedule) {
        Ok(digest) if digest == evidence.schedule_sha256 => {}
        Ok(_) => issues.push(StudyEvidenceIssue::ScheduleDigestMismatch),
        Err(_) => issues.push(StudyEvidenceIssue::EvidenceSerializationFailed {
            field: "schedule".into(),
        }),
    }

    let presentation_map: BTreeMap<_, _> = schedule
        .presentations
        .iter()
        .map(|presentation| (presentation.presentation_id.as_str(), presentation))
        .collect();

    let mut structural_ids = BTreeSet::new();
    for structural in &evidence.structural {
        if !structural_ids.insert(structural.presentation_id.clone()) {
            issues.push(StudyEvidenceIssue::DuplicateStructuralOutcome {
                presentation_id: structural.presentation_id.clone(),
            });
        }
        if !presentation_map.contains_key(structural.presentation_id.as_str()) {
            issues.push(StudyEvidenceIssue::UnknownPresentation {
                presentation_id: structural.presentation_id.clone(),
            });
        }
        if !structural_valid(&structural.outcome) {
            issues.push(StudyEvidenceIssue::InvalidStructuralOutcome {
                presentation_id: structural.presentation_id.clone(),
            });
        }
    }
    for presentation in &schedule.presentations {
        if !structural_ids.contains(&presentation.presentation_id) {
            issues.push(StudyEvidenceIssue::MissingStructuralOutcome {
                presentation_id: presentation.presentation_id.clone(),
            });
        }
    }

    let mut block_ids = BTreeSet::new();
    let mut listener_fixture = BTreeSet::new();
    for (index, block) in evidence.listener_blocks.iter().enumerate() {
        validate_block_identity(
            "listener",
            index,
            &block.block_id,
            &block.listener_id,
            &block.key,
            manifest,
            &mut block_ids,
            &mut listener_fixture,
            &mut issues,
        );
        validate_listener_block(block, &presentation_map, &mut issues);
    }

    let mut workflow_fixture = BTreeSet::new();
    let mut included_workflow_fixtures = BTreeSet::new();
    for (index, block) in evidence.workflow_blocks.iter().enumerate() {
        validate_block_identity(
            "workflow",
            index,
            &block.block_id,
            &block.artist_id,
            &block.key,
            manifest,
            &mut block_ids,
            &mut workflow_fixture,
            &mut issues,
        );
        if block.status == EvidenceBlockStatus::Included
            && !included_workflow_fixtures.insert(block.key.clone())
        {
            issues.push(StudyEvidenceIssue::MultipleIncludedWorkflowBlocks {
                key: block.key.clone(),
            });
        }
        validate_workflow_block(block, &presentation_map, &mut issues);
    }
    issues
}

#[allow(clippy::too_many_arguments)]
fn validate_block_identity(
    kind: &str,
    index: usize,
    block_id: &str,
    participant_id: &str,
    key: &FrozenTrialKey,
    manifest: &FrozenStudyManifest,
    block_ids: &mut BTreeSet<String>,
    participant_fixture: &mut BTreeSet<(String, FrozenTrialKey)>,
    issues: &mut Vec<StudyEvidenceIssue>,
) {
    if block_id.trim().is_empty() {
        issues.push(StudyEvidenceIssue::EmptyBlockId {
            kind: kind.into(),
            index,
        });
    } else if !block_ids.insert(block_id.to_owned()) {
        issues.push(StudyEvidenceIssue::DuplicateBlockId {
            block_id: block_id.to_owned(),
        });
    }
    if participant_id.trim().is_empty() {
        issues.push(StudyEvidenceIssue::EmptyParticipantId {
            block_id: block_id.to_owned(),
        });
    }
    if manifest.fixture(key).is_none() {
        issues.push(StudyEvidenceIssue::UnknownFixture { key: key.clone() });
    }
    if !participant_fixture.insert((participant_id.to_owned(), key.clone())) {
        issues.push(StudyEvidenceIssue::DuplicateParticipantFixture {
            participant_id: participant_id.to_owned(),
            key: key.clone(),
        });
    }
}

fn validate_listener_block(
    block: &ListenerResponseBlock,
    presentation_map: &BTreeMap<&str, &crate::blinded_study::BlindedPresentation>,
    issues: &mut Vec<StudyEvidenceIssue>,
) {
    if block.status == EvidenceBlockStatus::Included && block.responses.len() != 4 {
        issues.push(StudyEvidenceIssue::IncompleteIncludedBlock {
            block_id: block.block_id.clone(),
            found: block.responses.len(),
        });
    }
    let mut presentations = BTreeSet::new();
    let mut ranks = BTreeSet::new();
    for response in &block.responses {
        match presentation_map.get(response.presentation_id.as_str()) {
            None => issues.push(StudyEvidenceIssue::UnknownPresentation {
                presentation_id: response.presentation_id.clone(),
            }),
            Some(presentation) if presentation.key != block.key => {
                issues.push(StudyEvidenceIssue::PresentationFixtureMismatch {
                    block_id: block.block_id.clone(),
                    presentation_id: response.presentation_id.clone(),
                });
            }
            Some(_) => {}
        }
        if !presentations.insert(response.presentation_id.clone()) {
            issues.push(StudyEvidenceIssue::DuplicatePresentationInBlock {
                block_id: block.block_id.clone(),
                presentation_id: response.presentation_id.clone(),
            });
        }
        if block.status == EvidenceBlockStatus::Included {
            if !(1..=4).contains(&response.preference_rank) {
                issues.push(StudyEvidenceIssue::InvalidPreferenceRank {
                    block_id: block.block_id.clone(),
                    presentation_id: response.presentation_id.clone(),
                });
            } else if !ranks.insert(response.preference_rank) {
                issues.push(StudyEvidenceIssue::DuplicatePreferenceRank {
                    block_id: block.block_id.clone(),
                    rank: response.preference_rank,
                });
            }
            for (field, value) in [
                ("development_instability", response.development_instability),
                ("earned_recapitulation", response.earned_recapitulation),
            ] {
                if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                    issues.push(StudyEvidenceIssue::InvalidListenerRating {
                        block_id: block.block_id.clone(),
                        presentation_id: response.presentation_id.clone(),
                        field: field.into(),
                    });
                }
            }
            if !response.playback_completed {
                issues.push(StudyEvidenceIssue::IncompletePlayback {
                    block_id: block.block_id.clone(),
                    presentation_id: response.presentation_id.clone(),
                });
            }
            if !response.attention_check_passed {
                issues.push(StudyEvidenceIssue::FailedIncludedAttentionCheck {
                    block_id: block.block_id.clone(),
                    presentation_id: response.presentation_id.clone(),
                });
            }
            if response.elapsed_ms == 0 {
                issues.push(StudyEvidenceIssue::ZeroResponseTime {
                    block_id: block.block_id.clone(),
                    presentation_id: response.presentation_id.clone(),
                });
            }
        }
    }
}

fn validate_workflow_block(
    block: &ArtistWorkflowBlock,
    presentation_map: &BTreeMap<&str, &crate::blinded_study::BlindedPresentation>,
    issues: &mut Vec<StudyEvidenceIssue>,
) {
    if block.status == EvidenceBlockStatus::Included && block.responses.len() != 4 {
        issues.push(StudyEvidenceIssue::IncompleteIncludedBlock {
            block_id: block.block_id.clone(),
            found: block.responses.len(),
        });
    }
    let mut presentations = BTreeSet::new();
    for response in &block.responses {
        match presentation_map.get(response.presentation_id.as_str()) {
            None => issues.push(StudyEvidenceIssue::UnknownPresentation {
                presentation_id: response.presentation_id.clone(),
            }),
            Some(presentation) if presentation.key != block.key => {
                issues.push(StudyEvidenceIssue::PresentationFixtureMismatch {
                    block_id: block.block_id.clone(),
                    presentation_id: response.presentation_id.clone(),
                });
            }
            Some(_) => {}
        }
        if !presentations.insert(response.presentation_id.clone()) {
            issues.push(StudyEvidenceIssue::DuplicatePresentationInBlock {
                block_id: block.block_id.clone(),
                presentation_id: response.presentation_id.clone(),
            });
        }
        if block.status == EvidenceBlockStatus::Included && response.time_to_commit_seconds == 0 {
            issues.push(StudyEvidenceIssue::ZeroWorkflowTime {
                block_id: block.block_id.clone(),
                presentation_id: response.presentation_id.clone(),
            });
        }
    }
}

pub fn compile_study_dataset(
    manifest: &FrozenStudyManifest,
    schedule: &BlindedSchedule,
    codebook: &BlindingCodebook,
    evidence: &RawStudyEvidence,
) -> Result<CompiledStudyDataset, Vec<StudyEvidenceIssue>> {
    let issues = validate_raw_study_evidence(manifest, schedule, codebook, evidence);
    if !issues.is_empty() {
        return Err(issues);
    }

    let codebook_map: BTreeMap<_, _> = codebook
        .entries
        .iter()
        .map(|entry| (entry.presentation_id.as_str(), entry))
        .collect();
    let structural_map: BTreeMap<_, _> = evidence
        .structural
        .iter()
        .map(|value| (value.presentation_id.as_str(), value))
        .collect();

    let included_listener: Vec<_> = evidence
        .listener_blocks
        .iter()
        .filter(|block| block.status == EvidenceBlockStatus::Included)
        .collect();
    let included_workflow: Vec<_> = evidence
        .workflow_blocks
        .iter()
        .filter(|block| block.status == EvidenceBlockStatus::Included)
        .collect();

    let mut records = Vec::with_capacity(schedule.presentations.len());
    for presentation in &schedule.presentations {
        let code = codebook_map[presentation.presentation_id.as_str()];
        let fixture = manifest
            .fixture(&presentation.key)
            .expect("validated fixture");
        let listener_responses: Vec<_> = included_listener
            .iter()
            .filter(|block| block.key == presentation.key)
            .flat_map(|block| block.responses.iter())
            .filter(|response| response.presentation_id == presentation.presentation_id)
            .collect();
        let perceptual = if listener_responses.is_empty() {
            None
        } else {
            let count = listener_responses.len() as f32;
            Some(PerceptualTrialOutcome {
                listener_count: listener_responses.len(),
                return_recognition_rate: Some(
                    listener_responses
                        .iter()
                        .filter(|response| response.return_recognized)
                        .count() as f32
                        / count,
                ),
                development_instability: Some(
                    listener_responses
                        .iter()
                        .map(|response| response.development_instability)
                        .sum::<f32>()
                        / count,
                ),
                earned_recapitulation: Some(
                    listener_responses
                        .iter()
                        .map(|response| response.earned_recapitulation)
                        .sum::<f32>()
                        / count,
                ),
                preference_rate: Some(
                    listener_responses
                        .iter()
                        .map(|response| f32::from(4 - response.preference_rank) / 3.0)
                        .sum::<f32>()
                        / count,
                ),
            })
        };
        let workflow = included_workflow
            .iter()
            .find(|block| block.key == presentation.key)
            .and_then(|block| {
                block
                    .responses
                    .iter()
                    .find(|response| response.presentation_id == presentation.presentation_id)
            })
            .map(|response| WorkflowTrialOutcome {
                kept: response.disposition == ArtistDisposition::Kept,
                edited: response.disposition == ArtistDisposition::Edited,
                rejected: response.disposition == ArtistDisposition::Rejected,
                time_to_commit_seconds: Some(response.time_to_commit_seconds),
            });
        records.push(CognitiveTrialRecord {
            key: presentation.key.clone(),
            arm: code.arm,
            frozen_input_sha256: fixture.frozen_input_sha256.clone(),
            policy_version: manifest.policy_versions[&code.arm].clone(),
            structural: structural_map[presentation.presentation_id.as_str()]
                .outcome
                .clone(),
            perceptual,
            workflow,
        });
    }
    records.sort_by(|left, right| left.key.cmp(&right.key).then(left.arm.cmp(&right.arm)));
    Ok(CompiledStudyDataset {
        manifest_sha256: evidence.manifest_sha256.clone(),
        schedule_sha256: evidence.schedule_sha256.clone(),
        codebook_sha256: canonical_json_sha256(codebook)
            .expect("validated codebook serialization must succeed"),
        raw_evidence_sha256: evidence.raw_evidence_sha256.clone(),
        included_listener_blocks: included_listener.len(),
        excluded_listener_blocks: evidence.listener_blocks.len() - included_listener.len(),
        included_workflow_blocks: included_workflow.len(),
        excluded_workflow_blocks: evidence.workflow_blocks.len() - included_workflow.len(),
        records,
    })
}

fn structural_valid(outcome: &StructuralTrialOutcome) -> bool {
    outcome.obligations_total > 0
        && outcome.obligations_fulfilled <= outcome.obligations_total
        && outcome
            .motif_return_similarity
            .is_some_and(|value| value.is_finite() && (0.0..=1.0).contains(&value))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blinded_study::{ArmArtifactBinding, build_blinded_schedule};
    use crate::cognitive_experiment::CognitivePolicyArm;
    use crate::evidence_digest::sha256_hex;
    use crate::experiment_manifest::{
        ConfirmatoryEndpoint, FrozenStudyFixture, MIN_CONFIRMATORY_FIXTURES, MIN_PILOT_FIXTURES,
        STUDY_MANIFEST_VERSION, StudySplit,
    };

    const DIGEST: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SECRET: [u8; 32] = [0x42; 32];

    fn manifest() -> FrozenStudyManifest {
        let mut fixtures = Vec::new();
        for index in 0..MIN_PILOT_FIXTURES + MIN_CONFIRMATORY_FIXTURES {
            fixtures.push(FrozenStudyFixture {
                key: FrozenTrialKey {
                    fixture_id: format!("fixture-{index}"),
                    seed: index as u64 + 1,
                },
                family_id: format!("family-{index}"),
                split: if index < MIN_PILOT_FIXTURES {
                    StudySplit::Pilot
                } else {
                    StudySplit::Confirmatory
                },
                frozen_input_sha256: format!("{:064x}", index + 1),
                subject_sha256: DIGEST.into(),
                renderer_sha256: DIGEST.into(),
                soundfont_sha256: DIGEST.into(),
                theory_constraints_sha256: DIGEST.into(),
                tonic: "C".into(),
                meter: "4/4".into(),
                orchestration: "piano".into(),
            });
        }
        FrozenStudyManifest {
            manifest_version: STUDY_MANIFEST_VERSION.into(),
            preregistration_sha256: DIGEST.into(),
            analysis_plan_sha256: DIGEST.into(),
            randomization_commitment_sha256: sha256_hex(&SECRET),
            policy_versions: CognitivePolicyArm::ALL
                .into_iter()
                .map(|arm| (arm, "policy-v1".into()))
                .collect(),
            primary_endpoints: vec![ConfirmatoryEndpoint::Preference],
            alpha: 0.05,
            fixtures,
        }
    }

    fn setup() -> (FrozenStudyManifest, BlindedSchedule, BlindingCodebook) {
        let manifest = manifest();
        let artifacts: Vec<_> = manifest
            .fixtures
            .iter()
            .flat_map(|fixture| {
                CognitivePolicyArm::ALL
                    .into_iter()
                    .map(move |arm| ArmArtifactBinding {
                        key: fixture.key.clone(),
                        arm,
                        audio_sha256: DIGEST.into(),
                        recipe_sha256: DIGEST.into(),
                    })
            })
            .collect();
        let (schedule, codebook) = build_blinded_schedule(&manifest, &artifacts, SECRET).unwrap();
        (manifest, schedule, codebook)
    }

    fn evidence(schedule: &BlindedSchedule) -> RawStudyEvidence {
        let mut evidence = RawStudyEvidence {
            manifest_sha256: schedule.manifest_sha256.clone(),
            schedule_sha256: canonical_json_sha256(schedule).unwrap(),
            raw_evidence_sha256: String::new(),
            structural: schedule
                .presentations
                .iter()
                .map(|presentation| StructuralPresentationOutcome {
                    presentation_id: presentation.presentation_id.clone(),
                    outcome: StructuralTrialOutcome {
                        hard_constraints_valid: true,
                        obligations_total: 4,
                        obligations_fulfilled: 4,
                        voice_leading_violations: 0,
                        motif_return_similarity: Some(0.98),
                        tonic_returned: true,
                    },
                })
                .collect(),
            listener_blocks: Vec::new(),
            workflow_blocks: Vec::new(),
        };
        seal_raw_evidence(&mut evidence).unwrap();
        evidence
    }

    #[test]
    fn structural_only_dataset_compiles_without_inventing_listener_evidence() {
        let (manifest, schedule, codebook) = setup();
        let dataset =
            compile_study_dataset(&manifest, &schedule, &codebook, &evidence(&schedule)).unwrap();
        assert_eq!(dataset.records.len(), schedule.presentations.len());
        assert!(
            dataset
                .records
                .iter()
                .all(|record| record.perceptual.is_none())
        );
    }

    #[test]
    fn duplicate_listener_ranks_are_rejected() {
        let (manifest, schedule, codebook) = setup();
        let mut evidence = evidence(&schedule);
        let key = manifest.fixtures[0].key.clone();
        let presentations: Vec<_> = schedule
            .presentations
            .iter()
            .filter(|presentation| presentation.key == key)
            .collect();
        evidence.listener_blocks.push(ListenerResponseBlock {
            block_id: "block-1".into(),
            listener_id: "listener-1".into(),
            key,
            status: EvidenceBlockStatus::Included,
            responses: presentations
                .iter()
                .map(|presentation| ListenerPresentationResponse {
                    presentation_id: presentation.presentation_id.clone(),
                    return_recognized: true,
                    development_instability: 0.5,
                    earned_recapitulation: 0.5,
                    preference_rank: 1,
                    playback_completed: true,
                    attention_check_passed: true,
                    elapsed_ms: 1_000,
                })
                .collect(),
        });
        seal_raw_evidence(&mut evidence).unwrap();
        let issues = validate_raw_study_evidence(&manifest, &schedule, &codebook, &evidence);
        assert!(
            issues
                .iter()
                .any(|issue| matches!(issue, StudyEvidenceIssue::DuplicatePreferenceRank { .. }))
        );
    }

    #[test]
    fn raw_evidence_tampering_is_detected() {
        let (manifest, schedule, codebook) = setup();
        let mut evidence = evidence(&schedule);
        evidence.structural[0].outcome.tonic_returned = false;
        let issues = validate_raw_study_evidence(&manifest, &schedule, &codebook, &evidence);
        assert!(issues.contains(&StudyEvidenceIssue::RawEvidenceDigestMismatch));
    }

    #[test]
    fn arm_labels_are_added_only_during_private_compilation() {
        let (manifest, schedule, codebook) = setup();
        let dataset =
            compile_study_dataset(&manifest, &schedule, &codebook, &evidence(&schedule)).unwrap();
        let arms: BTreeSet<_> = dataset
            .records
            .iter()
            .filter(|record| record.key == manifest.fixtures[0].key)
            .map(|record| record.arm)
            .collect();
        assert_eq!(arms, CognitivePolicyArm::ALL.into_iter().collect());
    }
}
