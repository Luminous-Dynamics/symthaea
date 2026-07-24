// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Participant-specific Williams counterbalancing for blinded study blocks.
//!
//! V8 balanced arm position across fixtures but gave every listener the same
//! order inside a fixture. V8.2 assigns one of four Williams sequences to each
//! participant-fixture block. Every presentation appears equally often in every
//! position and every ordered first-order carryover pair is balanced.

use crate::blinded_study::{BlindedSchedule, BlindingCodebook, validate_blinded_schedule};
use crate::cognitive_experiment::{CognitivePolicyArm, FrozenTrialKey};
use crate::evidence_digest::{canonical_json_sha256, sha256_hex};
use crate::experiment_manifest::{FrozenStudyManifest, StudySplit};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const PARTICIPANT_SCHEDULE_VERSION: &str = "symthaea-muse-participant-williams-schedule-v1";
pub const MIN_COUNTERBALANCED_PARTICIPANTS: usize = 12;

/// Four-sequence Williams square. Across one complete cycle, every treatment
/// appears once in every position and every ordered pair of distinct treatments
/// appears exactly once as an adjacent carryover.
const WILLIAMS_FOUR: [[usize; 4]; 4] = [[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1]];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParticipantCohortSpec {
    pub cohort_id: String,
    /// Pseudonymous participant tokens. Raw names or contact details do not
    /// belong in the public schedule.
    pub participant_tokens: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParticipantBlockAssignment {
    pub block_id: String,
    pub participant_token: String,
    pub key: FrozenTrialKey,
    pub ordered_presentation_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParticipantScheduleBook {
    pub schedule_version: String,
    pub manifest_sha256: String,
    pub base_schedule_sha256: String,
    pub randomization_commitment_sha256: String,
    pub cohort_id: String,
    pub blocks: Vec<ParticipantBlockAssignment>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrivateParticipantSequence {
    pub block_id: String,
    pub participant_token: String,
    pub key: FrozenTrialKey,
    pub sequence_index: u8,
    pub arm_order: Vec<CognitivePolicyArm>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParticipantScheduleAudit {
    pub schedule_version: String,
    pub manifest_sha256: String,
    pub base_schedule_sha256: String,
    pub cohort_id: String,
    pub sequences: Vec<PrivateParticipantSequence>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParticipantScheduleIssue {
    InvalidManifest,
    InvalidBaseSchedule,
    ManifestSerializationFailed,
    BaseScheduleSerializationFailed,
    ManifestDigestMismatch,
    BaseScheduleDigestMismatch,
    SecretCommitmentMismatch,
    WrongVersion {
        found: String,
    },
    EmptyCohortId,
    TooFewParticipants {
        found: usize,
        required: usize,
    },
    ParticipantCountNotDivisibleByFour {
        found: usize,
    },
    EmptyParticipantToken {
        index: usize,
    },
    DuplicateParticipantToken {
        participant_token: String,
    },
    DuplicateBlockId {
        block_id: String,
    },
    MissingParticipantFixture {
        participant_token: String,
        key: FrozenTrialKey,
    },
    UnexpectedParticipantFixture {
        participant_token: String,
        key: FrozenTrialKey,
    },
    DuplicateParticipantFixture {
        participant_token: String,
        key: FrozenTrialKey,
    },
    IncompleteBlock {
        block_id: String,
        found: usize,
    },
    DuplicatePresentationInBlock {
        block_id: String,
        presentation_id: String,
    },
    UnknownPresentation {
        block_id: String,
        presentation_id: String,
    },
    PresentationFixtureMismatch {
        block_id: String,
        presentation_id: String,
    },
    PositionImbalance {
        key: FrozenTrialKey,
        presentation_id: String,
        counts: Vec<usize>,
    },
    CarryoverImbalance {
        key: FrozenTrialKey,
        left_presentation_id: String,
        right_presentation_id: String,
        found: usize,
        expected: usize,
    },
    AuditMismatch {
        block_id: String,
    },
}

pub fn build_participant_schedule(
    manifest: &FrozenStudyManifest,
    base_schedule: &BlindedSchedule,
    codebook: &BlindingCodebook,
    cohort: &ParticipantCohortSpec,
    secret_key: [u8; 32],
) -> Result<(ParticipantScheduleBook, ParticipantScheduleAudit), Vec<ParticipantScheduleIssue>> {
    let mut issues = validate_cohort(manifest, cohort);
    if !validate_blinded_schedule(manifest, base_schedule, Some(codebook)).is_empty() {
        issues.push(ParticipantScheduleIssue::InvalidBaseSchedule);
    }
    if sha256_hex(&secret_key) != manifest.randomization_commitment_sha256 {
        issues.push(ParticipantScheduleIssue::SecretCommitmentMismatch);
    }
    let manifest_sha256 = match canonical_json_sha256(manifest) {
        Ok(value) => value,
        Err(_) => {
            issues.push(ParticipantScheduleIssue::ManifestSerializationFailed);
            String::new()
        }
    };
    let base_schedule_sha256 = match canonical_json_sha256(base_schedule) {
        Ok(value) => value,
        Err(_) => {
            issues.push(ParticipantScheduleIssue::BaseScheduleSerializationFailed);
            String::new()
        }
    };
    if base_schedule.manifest_sha256 != manifest_sha256 {
        issues.push(ParticipantScheduleIssue::ManifestDigestMismatch);
    }
    if !issues.is_empty() {
        return Err(issues);
    }

    let codebook_by_fixture_arm: BTreeMap<_, _> = codebook
        .entries
        .iter()
        .map(|entry| {
            (
                (entry.key.clone(), entry.arm),
                entry.presentation_id.as_str(),
            )
        })
        .collect();
    let mut participants = cohort.participant_tokens.clone();
    participants.sort();
    let mut fixtures: Vec<_> = manifest
        .fixtures
        .iter()
        .filter(|fixture| fixture.split == StudySplit::Confirmatory)
        .collect();
    fixtures.sort_by(|left, right| left.key.cmp(&right.key));

    let mut rng = StdRng::from_seed(secret_key);
    let global_offset = rng.gen_range(0..WILLIAMS_FOUR.len());
    let mut blocks = Vec::with_capacity(participants.len() * fixtures.len());
    let mut sequences = Vec::with_capacity(participants.len() * fixtures.len());

    for fixture in fixtures {
        let fixture_offset = rng.gen_range(0..WILLIAMS_FOUR.len());
        for (participant_index, participant_token) in participants.iter().enumerate() {
            let sequence_index =
                (participant_index + fixture_offset + global_offset) % WILLIAMS_FOUR.len();
            let arm_order: Vec<_> = WILLIAMS_FOUR[sequence_index]
                .iter()
                .map(|index| CognitivePolicyArm::ALL[*index])
                .collect();
            let ordered_presentation_ids: Vec<_> = arm_order
                .iter()
                .map(|arm| {
                    codebook_by_fixture_arm
                        .get(&(fixture.key.clone(), *arm))
                        .expect("validated V8 codebook contains all fixture-arm bindings")
                        .to_string()
                })
                .collect();
            let block_id = format!(
                "{}-{}-{:016x}",
                cohort.cohort_id,
                participant_token,
                stable_hash(&format!(
                    "{}|{}|{}",
                    participant_token, fixture.key.fixture_id, fixture.key.seed
                ))
            );
            blocks.push(ParticipantBlockAssignment {
                block_id: block_id.clone(),
                participant_token: participant_token.clone(),
                key: fixture.key.clone(),
                ordered_presentation_ids,
            });
            sequences.push(PrivateParticipantSequence {
                block_id,
                participant_token: participant_token.clone(),
                key: fixture.key.clone(),
                sequence_index: sequence_index as u8,
                arm_order,
            });
        }
    }

    let schedule = ParticipantScheduleBook {
        schedule_version: PARTICIPANT_SCHEDULE_VERSION.into(),
        manifest_sha256: manifest_sha256.clone(),
        base_schedule_sha256: base_schedule_sha256.clone(),
        randomization_commitment_sha256: manifest.randomization_commitment_sha256.clone(),
        cohort_id: cohort.cohort_id.clone(),
        blocks,
    };
    let audit = ParticipantScheduleAudit {
        schedule_version: PARTICIPANT_SCHEDULE_VERSION.into(),
        manifest_sha256,
        base_schedule_sha256,
        cohort_id: cohort.cohort_id.clone(),
        sequences,
    };
    let mut validation =
        validate_participant_schedule(manifest, base_schedule, cohort, &schedule, Some(&audit));
    validation.extend(validate_participant_audit(codebook, &schedule, &audit));
    if validation.is_empty() {
        Ok((schedule, audit))
    } else {
        Err(validation)
    }
}

pub fn validate_participant_schedule(
    manifest: &FrozenStudyManifest,
    base_schedule: &BlindedSchedule,
    cohort: &ParticipantCohortSpec,
    schedule: &ParticipantScheduleBook,
    audit: Option<&ParticipantScheduleAudit>,
) -> Vec<ParticipantScheduleIssue> {
    let mut issues = validate_cohort(manifest, cohort);
    if schedule.schedule_version != PARTICIPANT_SCHEDULE_VERSION {
        issues.push(ParticipantScheduleIssue::WrongVersion {
            found: schedule.schedule_version.clone(),
        });
    }
    match canonical_json_sha256(manifest) {
        Ok(value) if value == schedule.manifest_sha256 => {}
        Ok(_) => issues.push(ParticipantScheduleIssue::ManifestDigestMismatch),
        Err(_) => issues.push(ParticipantScheduleIssue::ManifestSerializationFailed),
    }
    match canonical_json_sha256(base_schedule) {
        Ok(value) if value == schedule.base_schedule_sha256 => {}
        Ok(_) => issues.push(ParticipantScheduleIssue::BaseScheduleDigestMismatch),
        Err(_) => issues.push(ParticipantScheduleIssue::BaseScheduleSerializationFailed),
    }
    if schedule.randomization_commitment_sha256 != manifest.randomization_commitment_sha256 {
        issues.push(ParticipantScheduleIssue::SecretCommitmentMismatch);
    }
    if schedule.cohort_id != cohort.cohort_id || schedule.cohort_id.trim().is_empty() {
        issues.push(ParticipantScheduleIssue::EmptyCohortId);
    }

    let presentations: BTreeMap<_, _> = base_schedule
        .presentations
        .iter()
        .map(|presentation| (presentation.presentation_id.as_str(), presentation))
        .collect();
    let confirmatory_keys: BTreeSet<_> = manifest
        .fixtures
        .iter()
        .filter(|fixture| fixture.split == StudySplit::Confirmatory)
        .map(|fixture| fixture.key.clone())
        .collect();
    let participant_tokens: BTreeSet<_> = cohort.participant_tokens.iter().cloned().collect();
    let mut block_ids = BTreeSet::new();
    let mut participant_fixtures = BTreeSet::new();
    let mut by_fixture: BTreeMap<FrozenTrialKey, Vec<&ParticipantBlockAssignment>> =
        BTreeMap::new();

    for block in &schedule.blocks {
        if !block_ids.insert(block.block_id.clone()) {
            issues.push(ParticipantScheduleIssue::DuplicateBlockId {
                block_id: block.block_id.clone(),
            });
        }
        if !participant_tokens.contains(&block.participant_token)
            || !confirmatory_keys.contains(&block.key)
        {
            issues.push(ParticipantScheduleIssue::UnexpectedParticipantFixture {
                participant_token: block.participant_token.clone(),
                key: block.key.clone(),
            });
        }
        if !participant_fixtures.insert((block.participant_token.clone(), block.key.clone())) {
            issues.push(ParticipantScheduleIssue::DuplicateParticipantFixture {
                participant_token: block.participant_token.clone(),
                key: block.key.clone(),
            });
        }
        if block.ordered_presentation_ids.len() != 4 {
            issues.push(ParticipantScheduleIssue::IncompleteBlock {
                block_id: block.block_id.clone(),
                found: block.ordered_presentation_ids.len(),
            });
        }
        let mut seen = BTreeSet::new();
        for presentation_id in &block.ordered_presentation_ids {
            if !seen.insert(presentation_id.clone()) {
                issues.push(ParticipantScheduleIssue::DuplicatePresentationInBlock {
                    block_id: block.block_id.clone(),
                    presentation_id: presentation_id.clone(),
                });
            }
            match presentations.get(presentation_id.as_str()) {
                None => issues.push(ParticipantScheduleIssue::UnknownPresentation {
                    block_id: block.block_id.clone(),
                    presentation_id: presentation_id.clone(),
                }),
                Some(presentation) if presentation.key != block.key => {
                    issues.push(ParticipantScheduleIssue::PresentationFixtureMismatch {
                        block_id: block.block_id.clone(),
                        presentation_id: presentation_id.clone(),
                    });
                }
                Some(_) => {}
            }
        }
        by_fixture.entry(block.key.clone()).or_default().push(block);
    }
    for participant_token in &cohort.participant_tokens {
        for key in &confirmatory_keys {
            if !participant_fixtures.contains(&(participant_token.clone(), key.clone())) {
                issues.push(ParticipantScheduleIssue::MissingParticipantFixture {
                    participant_token: participant_token.clone(),
                    key: key.clone(),
                });
            }
        }
    }

    for (key, blocks) in by_fixture {
        let expected_position_count = blocks.len() / 4;
        let expected_carryover_count = blocks.len() / 4;
        let fixture_presentations: BTreeSet<_> = blocks
            .iter()
            .flat_map(|block| block.ordered_presentation_ids.iter().cloned())
            .collect();
        for presentation_id in &fixture_presentations {
            let counts: Vec<_> = (0..4)
                .map(|position| {
                    blocks
                        .iter()
                        .filter(|block| {
                            block
                                .ordered_presentation_ids
                                .get(position)
                                .is_some_and(|value| value == presentation_id)
                        })
                        .count()
                })
                .collect();
            if counts.iter().any(|count| *count != expected_position_count) {
                issues.push(ParticipantScheduleIssue::PositionImbalance {
                    key: key.clone(),
                    presentation_id: presentation_id.clone(),
                    counts,
                });
            }
        }
        for left in &fixture_presentations {
            for right in &fixture_presentations {
                if left == right {
                    continue;
                }
                let found = blocks
                    .iter()
                    .flat_map(|block| block.ordered_presentation_ids.windows(2))
                    .filter(|pair| pair[0] == *left && pair[1] == *right)
                    .count();
                if found != expected_carryover_count {
                    issues.push(ParticipantScheduleIssue::CarryoverImbalance {
                        key: key.clone(),
                        left_presentation_id: left.clone(),
                        right_presentation_id: right.clone(),
                        found,
                        expected: expected_carryover_count,
                    });
                }
            }
        }
    }

    if let Some(audit) = audit {
        let audit_map: BTreeMap<_, _> = audit
            .sequences
            .iter()
            .map(|sequence| (sequence.block_id.as_str(), sequence))
            .collect();
        for block in &schedule.blocks {
            let Some(sequence) = audit_map.get(block.block_id.as_str()) else {
                issues.push(ParticipantScheduleIssue::AuditMismatch {
                    block_id: block.block_id.clone(),
                });
                continue;
            };
            if sequence.participant_token != block.participant_token
                || sequence.key != block.key
                || sequence.arm_order.len() != block.ordered_presentation_ids.len()
            {
                issues.push(ParticipantScheduleIssue::AuditMismatch {
                    block_id: block.block_id.clone(),
                });
            }
        }
    }
    issues
}

pub fn validate_participant_audit(
    codebook: &BlindingCodebook,
    schedule: &ParticipantScheduleBook,
    audit: &ParticipantScheduleAudit,
) -> Vec<ParticipantScheduleIssue> {
    let mut issues = Vec::new();
    let arm_by_presentation: BTreeMap<_, _> = codebook
        .entries
        .iter()
        .map(|entry| (entry.presentation_id.as_str(), entry.arm))
        .collect();
    let audit_map: BTreeMap<_, _> = audit
        .sequences
        .iter()
        .map(|sequence| (sequence.block_id.as_str(), sequence))
        .collect();
    for block in &schedule.blocks {
        let Some(sequence) = audit_map.get(block.block_id.as_str()) else {
            issues.push(ParticipantScheduleIssue::AuditMismatch {
                block_id: block.block_id.clone(),
            });
            continue;
        };
        let derived: Option<Vec<_>> = block
            .ordered_presentation_ids
            .iter()
            .map(|presentation_id| arm_by_presentation.get(presentation_id.as_str()).copied())
            .collect();
        if sequence.participant_token != block.participant_token
            || sequence.key != block.key
            || sequence.sequence_index as usize >= WILLIAMS_FOUR.len()
            || derived.as_ref() != Some(&sequence.arm_order)
        {
            issues.push(ParticipantScheduleIssue::AuditMismatch {
                block_id: block.block_id.clone(),
            });
        }
    }
    if audit.sequences.len() != schedule.blocks.len() {
        issues.push(ParticipantScheduleIssue::AuditMismatch {
            block_id: "<audit-cardinality>".into(),
        });
    }
    issues
}

fn validate_cohort(
    manifest: &FrozenStudyManifest,
    cohort: &ParticipantCohortSpec,
) -> Vec<ParticipantScheduleIssue> {
    let mut issues = Vec::new();
    if !manifest.validate().is_empty() {
        issues.push(ParticipantScheduleIssue::InvalidManifest);
    }
    if cohort.cohort_id.trim().is_empty() {
        issues.push(ParticipantScheduleIssue::EmptyCohortId);
    }
    if cohort.participant_tokens.len() < MIN_COUNTERBALANCED_PARTICIPANTS {
        issues.push(ParticipantScheduleIssue::TooFewParticipants {
            found: cohort.participant_tokens.len(),
            required: MIN_COUNTERBALANCED_PARTICIPANTS,
        });
    }
    if !cohort.participant_tokens.len().is_multiple_of(4) {
        issues.push(
            ParticipantScheduleIssue::ParticipantCountNotDivisibleByFour {
                found: cohort.participant_tokens.len(),
            },
        );
    }
    let mut tokens = BTreeSet::new();
    for (index, token) in cohort.participant_tokens.iter().enumerate() {
        if token.trim().is_empty() {
            issues.push(ParticipantScheduleIssue::EmptyParticipantToken { index });
        }
        if !tokens.insert(token.clone()) {
            issues.push(ParticipantScheduleIssue::DuplicateParticipantToken {
                participant_token: token.clone(),
            });
        }
    }
    issues
}

fn stable_hash(value: &str) -> u64 {
    value.bytes().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blinded_study::{ArmArtifactBinding, build_blinded_schedule};
    use crate::evidence_digest::sha256_hex;
    use crate::experiment_manifest::{
        ConfirmatoryEndpoint, FrozenStudyFixture, MIN_CONFIRMATORY_FIXTURES, MIN_PILOT_FIXTURES,
        STUDY_MANIFEST_VERSION,
    };

    const DIGEST: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SECRET: [u8; 32] = [0x42; 32];

    fn manifest() -> FrozenStudyManifest {
        let fixtures = (0..MIN_PILOT_FIXTURES + MIN_CONFIRMATORY_FIXTURES)
            .map(|index| FrozenStudyFixture {
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
            })
            .collect();
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

    fn artifacts(manifest: &FrozenStudyManifest) -> Vec<ArmArtifactBinding> {
        manifest
            .fixtures
            .iter()
            .flat_map(|fixture| {
                CognitivePolicyArm::ALL.map(|arm| ArmArtifactBinding {
                    key: fixture.key.clone(),
                    arm,
                    audio_sha256: DIGEST.into(),
                    recipe_sha256: DIGEST.into(),
                })
            })
            .collect()
    }

    #[test]
    fn participant_orders_balance_positions_and_carryover() {
        let manifest = manifest();
        let (base, codebook) =
            build_blinded_schedule(&manifest, &artifacts(&manifest), SECRET).unwrap();
        let cohort = ParticipantCohortSpec {
            cohort_id: "cohort-a".into(),
            participant_tokens: (0..12).map(|index| format!("P{index:03}")).collect(),
        };
        let (schedule, audit) =
            build_participant_schedule(&manifest, &base, &codebook, &cohort, SECRET).unwrap();
        assert!(
            validate_participant_schedule(&manifest, &base, &cohort, &schedule, Some(&audit))
                .is_empty()
        );
    }

    #[test]
    fn private_arm_audit_detects_tampering() {
        let manifest = manifest();
        let (base, codebook) =
            build_blinded_schedule(&manifest, &artifacts(&manifest), SECRET).unwrap();
        let cohort = ParticipantCohortSpec {
            cohort_id: "cohort-a".into(),
            participant_tokens: (0..12).map(|index| format!("P{index:03}")).collect(),
        };
        let (schedule, mut audit) =
            build_participant_schedule(&manifest, &base, &codebook, &cohort, SECRET).unwrap();
        audit.sequences[0].arm_order.swap(0, 1);
        assert!(
            validate_participant_audit(&codebook, &schedule, &audit)
                .iter()
                .any(|issue| matches!(issue, ParticipantScheduleIssue::AuditMismatch { .. }))
        );
    }

    #[test]
    fn incomplete_williams_cycle_is_rejected() {
        let manifest = manifest();
        let cohort = ParticipantCohortSpec {
            cohort_id: "cohort-a".into(),
            participant_tokens: (0..13).map(|index| format!("P{index:03}")).collect(),
        };
        assert!(
            validate_cohort(&manifest, &cohort)
                .iter()
                .any(|issue| matches!(
                    issue,
                    ParticipantScheduleIssue::ParticipantCountNotDivisibleByFour { .. }
                ))
        );
    }
}
