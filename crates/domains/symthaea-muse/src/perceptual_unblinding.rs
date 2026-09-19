// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003P1F: post-close unblinding and deterministic outcome derivation.
//!
//! This layer is intentionally non-statistical. It verifies the frozen P1E
//! collection close, verifies the P1D randomization-key reveal, then compiles
//! blinded raw choices into exactly the two registered binary outcomes:
//! ABX correctness and directional intervention selection. The resulting
//! dataset does not retain private arm mappings or the revealed secret.

use crate::evidence_digest::{
    canonical_json_sha256,
    perceptual_analysis_plan::FrozenPerceptualAnalysisSpecV1,
    perceptual_collection_evidence::{
        FrozenPerceptualCollectionAuthorityV1, PerceptualCollectionCloseV1,
        RawPerceptualChoiceV1, RawPerceptualCollectionV1, SessionStatusV1,
        validate_collection_close,
    },
    perceptual_participant_schedule::{
        PerceptualCohortSlotsV1, PerceptualParticipantScheduleAuditV1,
        PerceptualParticipantScheduleBookV1, PrivateTrialMappingV1,
    },
    perceptual_schedule_reveal::verify_perceptual_schedule_key_reveal,
    perceptual_stimulus_pack::{
        FrozenC6fRenderSubjectBindingV1, FrozenPerceptualStimulusPackV1, StimulusArmV1,
    },
    perceptual_study_protocol::{
        FrozenPerceptualStudyProtocolV1, MEL003_FIXED_SEEDS, PerceptualTaskV1,
    },
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const PERCEPTUAL_DERIVED_DATASET_VERSION: &str =
    "mel003-perceptual-derived-dataset-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DerivedPerceptualOutcomeV1 {
    AbxCorrect { correct: bool },
    DirectionalInterventionSelected { intervention_selected: bool },
}

impl DerivedPerceptualOutcomeV1 {
    pub fn binary_response(self) -> bool {
        match self {
            Self::AbxCorrect { correct } => correct,
            Self::DirectionalInterventionSelected {
                intervention_selected,
            } => intervention_selected,
        }
    }

    pub fn task(self) -> PerceptualTaskV1 {
        match self {
            Self::AbxCorrect { .. } => PerceptualTaskV1::AbxDiscrimination,
            Self::DirectionalInterventionSelected { .. } => {
                PerceptualTaskV1::DirectionalRearticulation2Afc
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DerivedPerceptualTrialV1 {
    pub participant_token: String,
    pub source_session_sha256: String,
    pub source_record_sha256: String,
    pub sequence: u32,
    pub trial_id: String,
    pub item_id: String,
    pub seed: u64,
    pub task: PerceptualTaskV1,
    pub outcome: DerivedPerceptualOutcomeV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenPerceptualDerivedDatasetV1 {
    pub dataset_version: String,
    pub protocol_sha256: String,
    pub analysis_spec_sha256: String,
    pub stimulus_pack_sha256: String,
    pub participant_schedule_sha256: String,
    pub private_audit_sha256: String,
    pub raw_dataset_sha256: String,
    pub collection_close_sha256: String,
    pub randomization_commitment_sha256: String,
    pub schedule_reveal_sha256: String,
    pub cohort_id: String,
    pub completed_participant_count: usize,
    pub excluded_aborted_session_count: usize,
    pub withdrawn_and_deleted_session_count: usize,
    pub rows: Vec<DerivedPerceptualTrialV1>,
    pub dataset_sha256: String,
}

#[derive(Serialize)]
struct DerivedDatasetCommitment<'a> {
    dataset_version: &'a str,
    protocol_sha256: &'a str,
    analysis_spec_sha256: &'a str,
    stimulus_pack_sha256: &'a str,
    participant_schedule_sha256: &'a str,
    private_audit_sha256: &'a str,
    raw_dataset_sha256: &'a str,
    collection_close_sha256: &'a str,
    randomization_commitment_sha256: &'a str,
    schedule_reveal_sha256: &'a str,
    cohort_id: &'a str,
    completed_participant_count: usize,
    excluded_aborted_session_count: usize,
    withdrawn_and_deleted_session_count: usize,
    rows: &'a [DerivedPerceptualTrialV1],
}

pub fn derived_dataset_commitment(
    dataset: &FrozenPerceptualDerivedDatasetV1,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&DerivedDatasetCommitment {
        dataset_version: &dataset.dataset_version,
        protocol_sha256: &dataset.protocol_sha256,
        analysis_spec_sha256: &dataset.analysis_spec_sha256,
        stimulus_pack_sha256: &dataset.stimulus_pack_sha256,
        participant_schedule_sha256: &dataset.participant_schedule_sha256,
        private_audit_sha256: &dataset.private_audit_sha256,
        raw_dataset_sha256: &dataset.raw_dataset_sha256,
        collection_close_sha256: &dataset.collection_close_sha256,
        randomization_commitment_sha256: &dataset.randomization_commitment_sha256,
        schedule_reveal_sha256: &dataset.schedule_reveal_sha256,
        cohort_id: &dataset.cohort_id,
        completed_participant_count: dataset.completed_participant_count,
        excluded_aborted_session_count: dataset.excluded_aborted_session_count,
        withdrawn_and_deleted_session_count: dataset.withdrawn_and_deleted_session_count,
        rows: &dataset.rows,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerceptualUnblindingIssueV1 {
    InvalidAnalysisSpec,
    AnalysisSpecSerializationFailed,
    AnalysisSpecDigestMismatch,
    InvalidCollectionClose,
    ScheduleRevealFailed,
    ScheduleRevealSerializationFailed,
    MissingAuditParticipant { participant_token: String },
    MissingAuditTrial { participant_token: String, seed: u64 },
    TrialIdentityMismatch { participant_token: String, trial_id: String },
    ChoiceTaskMismatch { participant_token: String, trial_id: String },
    DuplicateDerivedTrial { participant_token: String, trial_id: String },
    CompleteSessionCountMismatch { found: usize, expected: usize },
    DerivedRowCountMismatch { found: usize, expected: usize },
    WrongDatasetVersion,
    InvalidDigest { field: String },
    EmptyCohortId,
    EmptyRowField { row_index: usize, field: String },
    OutcomeTaskMismatch { row_index: usize },
    CompletedParticipantCountMismatch { found: usize, expected: usize },
    TotalRowCountMismatch { found: usize, expected: usize },
    ParticipantRowCountMismatch { participant_token: String, found: usize, expected: usize },
    ParticipantSequencePanelMismatch { participant_token: String },
    ParticipantTaskSeedPanelMismatch { participant_token: String, task: PerceptualTaskV1 },
    ParticipantSessionDigestMismatch { participant_token: String },
    DuplicateSourceRecordDigest { digest: String },
    DatasetDigestMismatch,
    SerializationFailed,
}

pub fn derive_perceptual_analysis_dataset(
    protocol: &FrozenPerceptualStudyProtocolV1,
    analysis_spec: &FrozenPerceptualAnalysisSpecV1,
    stimulus_pack: &FrozenPerceptualStimulusPackV1,
    render_binding: &FrozenC6fRenderSubjectBindingV1,
    cohort: &PerceptualCohortSlotsV1,
    schedule: &PerceptualParticipantScheduleBookV1,
    private_audit: &PerceptualParticipantScheduleAuditV1,
    authority: &FrozenPerceptualCollectionAuthorityV1,
    collection: &RawPerceptualCollectionV1,
    close: &PerceptualCollectionCloseV1,
    secret_key: [u8; 32],
) -> Result<FrozenPerceptualDerivedDatasetV1, Vec<PerceptualUnblindingIssueV1>> {
    let mut issues = Vec::new();

    if !analysis_spec.validate().is_empty() {
        issues.push(PerceptualUnblindingIssueV1::InvalidAnalysisSpec);
    }
    let analysis_spec_sha256 = match canonical_json_sha256(analysis_spec) {
        Ok(value) => {
            if value != protocol.analysis_spec_sha256 {
                issues.push(PerceptualUnblindingIssueV1::AnalysisSpecDigestMismatch);
            }
            value
        }
        Err(_) => {
            issues.push(PerceptualUnblindingIssueV1::AnalysisSpecSerializationFailed);
            String::new()
        }
    };

    if !validate_collection_close(
        protocol,
        stimulus_pack,
        render_binding,
        schedule,
        authority,
        collection,
        close,
    )
    .is_empty()
    {
        issues.push(PerceptualUnblindingIssueV1::InvalidCollectionClose);
    }

    let reveal = verify_perceptual_schedule_key_reveal(
        protocol,
        stimulus_pack,
        render_binding,
        cohort,
        schedule,
        private_audit,
        secret_key,
    );
    if !reveal.success() {
        issues.push(PerceptualUnblindingIssueV1::ScheduleRevealFailed);
    }
    let schedule_reveal_sha256 = match canonical_json_sha256(&reveal) {
        Ok(value) => value,
        Err(_) => {
            issues.push(PerceptualUnblindingIssueV1::ScheduleRevealSerializationFailed);
            String::new()
        }
    };

    if !issues.is_empty() {
        return Err(issues);
    }

    let audit_by_participant: BTreeMap<_, _> = private_audit
        .participants
        .iter()
        .map(|participant| (participant.participant_token.as_str(), participant))
        .collect();

    let complete_sessions: Vec<_> = collection
        .sessions
        .iter()
        .filter(|session| session.status == SessionStatusV1::Complete)
        .collect();
    if complete_sessions.len() != close.completed_sessions {
        issues.push(PerceptualUnblindingIssueV1::CompleteSessionCountMismatch {
            found: complete_sessions.len(),
            expected: close.completed_sessions,
        });
    }

    let expected_rows: usize = complete_sessions
        .iter()
        .map(|session| session.records.len())
        .sum();
    let mut rows = Vec::with_capacity(expected_rows);
    let mut seen_trials = BTreeSet::new();

    for session in complete_sessions {
        let Some(audit_participant) =
            audit_by_participant.get(session.participant_token.as_str())
        else {
            issues.push(PerceptualUnblindingIssueV1::MissingAuditParticipant {
                participant_token: session.participant_token.clone(),
            });
            continue;
        };
        let audit_by_seed: BTreeMap<_, _> = audit_participant
            .trials
            .iter()
            .map(|trial| (trial.seed, trial))
            .collect();

        for record in &session.records {
            let Some(mapping) = audit_by_seed.get(&record.seed) else {
                issues.push(PerceptualUnblindingIssueV1::MissingAuditTrial {
                    participant_token: session.participant_token.clone(),
                    seed: record.seed,
                });
                continue;
            };
            let expected_trial_id = match record.task {
                PerceptualTaskV1::AbxDiscrimination => &mapping.abx_trial_id,
                PerceptualTaskV1::DirectionalRearticulation2Afc => {
                    &mapping.directional_trial_id
                }
            };
            if record.trial_id != *expected_trial_id
                || record.item_id != mapping.item_id
                || record.seed != mapping.seed
            {
                issues.push(PerceptualUnblindingIssueV1::TrialIdentityMismatch {
                    participant_token: session.participant_token.clone(),
                    trial_id: record.trial_id.clone(),
                });
                continue;
            }
            let outcome = match derive_outcome(mapping, record.task, record.choice) {
                Some(outcome) => outcome,
                None => {
                    issues.push(PerceptualUnblindingIssueV1::ChoiceTaskMismatch {
                        participant_token: session.participant_token.clone(),
                        trial_id: record.trial_id.clone(),
                    });
                    continue;
                }
            };
            let trial_key = (
                session.participant_token.clone(),
                record.trial_id.clone(),
            );
            if !seen_trials.insert(trial_key) {
                issues.push(PerceptualUnblindingIssueV1::DuplicateDerivedTrial {
                    participant_token: session.participant_token.clone(),
                    trial_id: record.trial_id.clone(),
                });
                continue;
            }
            rows.push(DerivedPerceptualTrialV1 {
                participant_token: session.participant_token.clone(),
                source_session_sha256: session.session_sha256.clone(),
                source_record_sha256: record.record_sha256.clone(),
                sequence: record.sequence,
                trial_id: record.trial_id.clone(),
                item_id: record.item_id.clone(),
                seed: record.seed,
                task: record.task,
                outcome,
            });
        }
    }

    if rows.len() != expected_rows {
        issues.push(PerceptualUnblindingIssueV1::DerivedRowCountMismatch {
            found: rows.len(),
            expected: expected_rows,
        });
    }
    if !issues.is_empty() {
        return Err(issues);
    }

    rows.sort_by(|left, right| {
        left.participant_token
            .cmp(&right.participant_token)
            .then_with(|| left.sequence.cmp(&right.sequence))
            .then_with(|| left.trial_id.cmp(&right.trial_id))
    });

    let mut dataset = FrozenPerceptualDerivedDatasetV1 {
        dataset_version: PERCEPTUAL_DERIVED_DATASET_VERSION.into(),
        protocol_sha256: canonical_json_sha256(protocol)
            .map_err(|_| vec![PerceptualUnblindingIssueV1::SerializationFailed])?,
        analysis_spec_sha256,
        stimulus_pack_sha256: canonical_json_sha256(stimulus_pack)
            .map_err(|_| vec![PerceptualUnblindingIssueV1::SerializationFailed])?,
        participant_schedule_sha256: canonical_json_sha256(schedule)
            .map_err(|_| vec![PerceptualUnblindingIssueV1::SerializationFailed])?,
        private_audit_sha256: schedule.private_audit_sha256.clone(),
        raw_dataset_sha256: collection.raw_dataset_sha256.clone(),
        collection_close_sha256: close.close_sha256.clone(),
        randomization_commitment_sha256: protocol
            .blinding
            .randomization_commitment_sha256
            .clone(),
        schedule_reveal_sha256,
        cohort_id: cohort.cohort_id.clone(),
        completed_participant_count: close.completed_sessions,
        excluded_aborted_session_count: close.aborted_sessions,
        withdrawn_and_deleted_session_count: close.withdrawn_and_deleted_sessions,
        rows,
        dataset_sha256: String::new(),
    };
    dataset.dataset_sha256 = derived_dataset_commitment(&dataset)
        .map_err(|_| vec![PerceptualUnblindingIssueV1::SerializationFailed])?;

    let dataset_issues = validate_derived_perceptual_dataset(&dataset);
    if dataset_issues.is_empty() {
        Ok(dataset)
    } else {
        Err(dataset_issues)
    }
}

fn derive_outcome(
    mapping: &PrivateTrialMappingV1,
    task: PerceptualTaskV1,
    choice: RawPerceptualChoiceV1,
) -> Option<DerivedPerceptualOutcomeV1> {
    match (task, choice) {
        (PerceptualTaskV1::AbxDiscrimination, RawPerceptualChoiceV1::AbxA) => {
            Some(DerivedPerceptualOutcomeV1::AbxCorrect {
                correct: mapping.a_arm == mapping.x_arm,
            })
        }
        (PerceptualTaskV1::AbxDiscrimination, RawPerceptualChoiceV1::AbxB) => {
            Some(DerivedPerceptualOutcomeV1::AbxCorrect {
                correct: mapping.b_arm == mapping.x_arm,
            })
        }
        (
            PerceptualTaskV1::DirectionalRearticulation2Afc,
            RawPerceptualChoiceV1::DirectionLeft,
        ) => Some(
            DerivedPerceptualOutcomeV1::DirectionalInterventionSelected {
                intervention_selected: mapping.left_arm == StimulusArmV1::Intervention,
            },
        ),
        (
            PerceptualTaskV1::DirectionalRearticulation2Afc,
            RawPerceptualChoiceV1::DirectionRight,
        ) => Some(
            DerivedPerceptualOutcomeV1::DirectionalInterventionSelected {
                intervention_selected: mapping.right_arm == StimulusArmV1::Intervention,
            },
        ),
        _ => None,
    }
}

pub fn validate_derived_perceptual_dataset(
    dataset: &FrozenPerceptualDerivedDatasetV1,
) -> Vec<PerceptualUnblindingIssueV1> {
    let mut issues = Vec::new();
    if dataset.dataset_version != PERCEPTUAL_DERIVED_DATASET_VERSION {
        issues.push(PerceptualUnblindingIssueV1::WrongDatasetVersion);
    }
    for (field, digest) in [
        ("protocol_sha256", dataset.protocol_sha256.as_str()),
        ("analysis_spec_sha256", dataset.analysis_spec_sha256.as_str()),
        ("stimulus_pack_sha256", dataset.stimulus_pack_sha256.as_str()),
        (
            "participant_schedule_sha256",
            dataset.participant_schedule_sha256.as_str(),
        ),
        ("private_audit_sha256", dataset.private_audit_sha256.as_str()),
        ("raw_dataset_sha256", dataset.raw_dataset_sha256.as_str()),
        (
            "collection_close_sha256",
            dataset.collection_close_sha256.as_str(),
        ),
        (
            "randomization_commitment_sha256",
            dataset.randomization_commitment_sha256.as_str(),
        ),
        ("schedule_reveal_sha256", dataset.schedule_reveal_sha256.as_str()),
        ("dataset_sha256", dataset.dataset_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            issues.push(PerceptualUnblindingIssueV1::InvalidDigest {
                field: field.into(),
            });
        }
    }
    if dataset.cohort_id.trim().is_empty() {
        issues.push(PerceptualUnblindingIssueV1::EmptyCohortId);
    }

    let expected_rows_per_participant = MEL003_FIXED_SEEDS.len() * 2;
    let expected_total_rows = dataset
        .completed_participant_count
        .saturating_mul(expected_rows_per_participant);
    if dataset.rows.len() != expected_total_rows {
        issues.push(PerceptualUnblindingIssueV1::TotalRowCountMismatch {
            found: dataset.rows.len(),
            expected: expected_total_rows,
        });
    }

    let mut rows_by_participant: BTreeMap<&str, Vec<(usize, &DerivedPerceptualTrialV1)>> =
        BTreeMap::new();
    let mut source_record_digests = BTreeSet::new();
    let mut seen_trials = BTreeSet::new();
    for (index, row) in dataset.rows.iter().enumerate() {
        for (field, value) in [
            ("participant_token", row.participant_token.as_str()),
            ("trial_id", row.trial_id.as_str()),
            ("item_id", row.item_id.as_str()),
        ] {
            if value.trim().is_empty() {
                issues.push(PerceptualUnblindingIssueV1::EmptyRowField {
                    row_index: index,
                    field: field.into(),
                });
            }
        }
        for (field, digest) in [
            ("source_session_sha256", row.source_session_sha256.as_str()),
            ("source_record_sha256", row.source_record_sha256.as_str()),
        ] {
            if !is_sha256(digest) {
                issues.push(PerceptualUnblindingIssueV1::InvalidDigest {
                    field: format!("rows[{index}].{field}"),
                });
            }
        }
        if row.outcome.task() != row.task {
            issues.push(PerceptualUnblindingIssueV1::OutcomeTaskMismatch {
                row_index: index,
            });
        }
        if !seen_trials.insert((row.participant_token.as_str(), row.trial_id.as_str())) {
            issues.push(PerceptualUnblindingIssueV1::DuplicateDerivedTrial {
                participant_token: row.participant_token.clone(),
                trial_id: row.trial_id.clone(),
            });
        }
        if !source_record_digests.insert(row.source_record_sha256.as_str()) {
            issues.push(PerceptualUnblindingIssueV1::DuplicateSourceRecordDigest {
                digest: row.source_record_sha256.clone(),
            });
        }
        rows_by_participant
            .entry(row.participant_token.as_str())
            .or_default()
            .push((index, row));
    }

    if rows_by_participant.len() != dataset.completed_participant_count {
        issues.push(PerceptualUnblindingIssueV1::CompletedParticipantCountMismatch {
            found: rows_by_participant.len(),
            expected: dataset.completed_participant_count,
        });
    }

    let expected_sequences: BTreeSet<u32> =
        (0..expected_rows_per_participant as u32).collect();
    let expected_seeds: BTreeSet<u64> = MEL003_FIXED_SEEDS.into_iter().collect();
    for (participant_token, participant_rows) in rows_by_participant {
        if participant_rows.len() != expected_rows_per_participant {
            issues.push(PerceptualUnblindingIssueV1::ParticipantRowCountMismatch {
                participant_token: participant_token.into(),
                found: participant_rows.len(),
                expected: expected_rows_per_participant,
            });
        }
        let sequences: BTreeSet<_> = participant_rows
            .iter()
            .map(|(_, row)| row.sequence)
            .collect();
        if sequences != expected_sequences {
            issues.push(
                PerceptualUnblindingIssueV1::ParticipantSequencePanelMismatch {
                    participant_token: participant_token.into(),
                },
            );
        }
        let session_digests: BTreeSet<_> = participant_rows
            .iter()
            .map(|(_, row)| row.source_session_sha256.as_str())
            .collect();
        if session_digests.len() != 1 {
            issues.push(PerceptualUnblindingIssueV1::ParticipantSessionDigestMismatch {
                participant_token: participant_token.into(),
            });
        }
        for task in [
            PerceptualTaskV1::AbxDiscrimination,
            PerceptualTaskV1::DirectionalRearticulation2Afc,
        ] {
            let task_rows: Vec<_> = participant_rows
                .iter()
                .filter(|(_, row)| row.task == task)
                .collect();
            let seeds: BTreeSet<_> = task_rows.iter().map(|(_, row)| row.seed).collect();
            if task_rows.len() != MEL003_FIXED_SEEDS.len() || seeds != expected_seeds {
                issues.push(
                    PerceptualUnblindingIssueV1::ParticipantTaskSeedPanelMismatch {
                        participant_token: participant_token.into(),
                        task,
                    },
                );
            }
        }
    }

    match derived_dataset_commitment(dataset) {
        Ok(value) if value == dataset.dataset_sha256 => {}
        Ok(_) => issues.push(PerceptualUnblindingIssueV1::DatasetDigestMismatch),
        Err(_) => issues.push(PerceptualUnblindingIssueV1::SerializationFailed),
    }
    issues
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence_digest::perceptual_participant_schedule::FactorialAssignmentV1;

    const DIGEST: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn mapping() -> PrivateTrialMappingV1 {
        PrivateTrialMappingV1 {
            participant_token: "p-1".into(),
            item_id: "sonata-seed-3".into(),
            seed: 3,
            factorial_assignment: FactorialAssignmentV1::from_index(0).unwrap(),
            abx_position: 0,
            directional_position: 4,
            baseline_output_sha256: DIGEST.into(),
            intervention_output_sha256: "b".repeat(64),
            a_arm: StimulusArmV1::Baseline,
            b_arm: StimulusArmV1::Intervention,
            x_arm: StimulusArmV1::Baseline,
            left_arm: StimulusArmV1::Intervention,
            right_arm: StimulusArmV1::Baseline,
            a_clip_id: "a".into(),
            b_clip_id: "b".into(),
            x_clip_id: "x".into(),
            left_clip_id: "left".into(),
            right_clip_id: "right".into(),
            abx_trial_id: "abx-trial".into(),
            directional_trial_id: "direction-trial".into(),
        }
    }

    fn valid_derived_dataset() -> FrozenPerceptualDerivedDatasetV1 {
        let mut rows = Vec::new();
        for (index, seed) in MEL003_FIXED_SEEDS.iter().copied().enumerate() {
            rows.push(DerivedPerceptualTrialV1 {
                participant_token: "p-1".into(),
                source_session_sha256: DIGEST.into(),
                source_record_sha256: format!("{index:064x}"),
                sequence: index as u32,
                trial_id: format!("abx-{seed}"),
                item_id: format!("sonata-seed-{seed}"),
                seed,
                task: PerceptualTaskV1::AbxDiscrimination,
                outcome: DerivedPerceptualOutcomeV1::AbxCorrect {
                    correct: index % 2 == 0,
                },
            });
        }
        for (index, seed) in MEL003_FIXED_SEEDS.iter().copied().enumerate() {
            let sequence = index + MEL003_FIXED_SEEDS.len();
            rows.push(DerivedPerceptualTrialV1 {
                participant_token: "p-1".into(),
                source_session_sha256: DIGEST.into(),
                source_record_sha256: format!("{sequence:064x}"),
                sequence: sequence as u32,
                trial_id: format!("direction-{seed}"),
                item_id: format!("sonata-seed-{seed}"),
                seed,
                task: PerceptualTaskV1::DirectionalRearticulation2Afc,
                outcome: DerivedPerceptualOutcomeV1::DirectionalInterventionSelected {
                    intervention_selected: index % 2 == 0,
                },
            });
        }
        let mut dataset = FrozenPerceptualDerivedDatasetV1 {
            dataset_version: PERCEPTUAL_DERIVED_DATASET_VERSION.into(),
            protocol_sha256: DIGEST.into(),
            analysis_spec_sha256: DIGEST.into(),
            stimulus_pack_sha256: DIGEST.into(),
            participant_schedule_sha256: DIGEST.into(),
            private_audit_sha256: DIGEST.into(),
            raw_dataset_sha256: DIGEST.into(),
            collection_close_sha256: DIGEST.into(),
            randomization_commitment_sha256: DIGEST.into(),
            schedule_reveal_sha256: DIGEST.into(),
            cohort_id: "cohort".into(),
            completed_participant_count: 1,
            excluded_aborted_session_count: 0,
            withdrawn_and_deleted_session_count: 0,
            rows,
            dataset_sha256: String::new(),
        };
        dataset.dataset_sha256 = derived_dataset_commitment(&dataset).unwrap();
        dataset
    }

    #[test]
    fn abx_correctness_is_derived_only_from_frozen_mapping_and_raw_choice() {
        let mapping = mapping();
        assert_eq!(
            derive_outcome(
                &mapping,
                PerceptualTaskV1::AbxDiscrimination,
                RawPerceptualChoiceV1::AbxA,
            ),
            Some(DerivedPerceptualOutcomeV1::AbxCorrect { correct: true })
        );
        assert_eq!(
            derive_outcome(
                &mapping,
                PerceptualTaskV1::AbxDiscrimination,
                RawPerceptualChoiceV1::AbxB,
            ),
            Some(DerivedPerceptualOutcomeV1::AbxCorrect { correct: false })
        );
    }

    #[test]
    fn directional_outcome_is_intervention_selection_not_preference() {
        let mapping = mapping();
        assert_eq!(
            derive_outcome(
                &mapping,
                PerceptualTaskV1::DirectionalRearticulation2Afc,
                RawPerceptualChoiceV1::DirectionLeft,
            ),
            Some(
                DerivedPerceptualOutcomeV1::DirectionalInterventionSelected {
                    intervention_selected: true,
                }
            )
        );
        assert_eq!(
            derive_outcome(
                &mapping,
                PerceptualTaskV1::DirectionalRearticulation2Afc,
                RawPerceptualChoiceV1::DirectionRight,
            ),
            Some(
                DerivedPerceptualOutcomeV1::DirectionalInterventionSelected {
                    intervention_selected: false,
                }
            )
        );
    }

    #[test]
    fn task_mismatched_raw_choice_is_rejected() {
        let mapping = mapping();
        assert_eq!(
            derive_outcome(
                &mapping,
                PerceptualTaskV1::AbxDiscrimination,
                RawPerceptualChoiceV1::DirectionLeft,
            ),
            None
        );
    }

    #[test]
    fn self_validation_rejects_selectively_pruned_complete_session() {
        let dataset = valid_derived_dataset();
        assert!(validate_derived_perceptual_dataset(&dataset).is_empty());

        let mut pruned = dataset.clone();
        pruned.rows.pop();
        pruned.dataset_sha256 = derived_dataset_commitment(&pruned).unwrap();
        let issues = validate_derived_perceptual_dataset(&pruned);
        assert!(issues.iter().any(|issue| matches!(
            issue,
            PerceptualUnblindingIssueV1::TotalRowCountMismatch { .. }
                | PerceptualUnblindingIssueV1::ParticipantRowCountMismatch { .. }
                | PerceptualUnblindingIssueV1::ParticipantTaskSeedPanelMismatch { .. }
        )));
    }

    #[test]
    fn dataset_commitment_detects_derived_outcome_tampering() {
        let dataset = valid_derived_dataset();
        let sealed = dataset.dataset_sha256.clone();
        let mut tampered = dataset;
        tampered.rows[0].outcome = DerivedPerceptualOutcomeV1::AbxCorrect { correct: false };
        assert_ne!(derived_dataset_commitment(&tampered).unwrap(), sealed);
    }
}
