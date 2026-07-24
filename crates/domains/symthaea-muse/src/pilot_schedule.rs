// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Participant-specific Williams counterbalancing for the pilot split.
//!
//! Pilot assignment uses an independent secret from confirmatory assignment so
//! closing and revealing the pilot cannot expose the confirmatory sequence.

use crate::blinded_study::{BlindedSchedule, BlindingCodebook, validate_blinded_schedule};
use crate::cognitive_experiment::{CognitivePolicyArm, FrozenTrialKey};
use crate::evidence_digest::{canonical_json_sha256, sha256_hex};
use crate::experiment_manifest::{FrozenStudyManifest, StudySplit};
use crate::participant_schedule::ParticipantBlockAssignment;
use crate::pilot_protocol::{FrozenPilotProtocol, PilotProtocolIssue};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const PILOT_PARTICIPANT_SCHEDULE_VERSION: &str = "symthaea-muse-pilot-williams-schedule-v1";

const WILLIAMS_FOUR: [[usize; 4]; 4] = [[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1]];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PilotCohortSpec {
    pub cohort_id: String,
    pub wave_id: String,
    pub participant_tokens: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PilotParticipantScheduleBook {
    pub schedule_version: String,
    pub manifest_sha256: String,
    pub base_schedule_sha256: String,
    pub pilot_protocol_sha256: String,
    pub pilot_randomization_commitment_sha256: String,
    pub cohort_id: String,
    pub wave_id: String,
    pub blocks: Vec<ParticipantBlockAssignment>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PilotPrivateSequence {
    pub block_id: String,
    pub participant_token: String,
    pub key: FrozenTrialKey,
    pub sequence_index: u8,
    pub arm_order: Vec<CognitivePolicyArm>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PilotScheduleAudit {
    pub schedule_version: String,
    pub manifest_sha256: String,
    pub base_schedule_sha256: String,
    pub pilot_protocol_sha256: String,
    pub cohort_id: String,
    pub wave_id: String,
    pub sequences: Vec<PilotPrivateSequence>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PilotScheduleIssue {
    InvalidManifest,
    InvalidBaseSchedule,
    PilotProtocol {
        issue: PilotProtocolIssue,
    },
    SerializationFailed {
        field: String,
    },
    DigestMismatch {
        field: String,
    },
    SecretCommitmentMismatch,
    ConfirmatoryCommitmentReused,
    WrongVersion {
        found: String,
    },
    EmptyCohortField {
        field: String,
    },
    EmptyParticipantRegistry,
    ParticipantCountNotDivisibleByFour {
        found: usize,
    },
    TooManyParticipantsInWave {
        found: usize,
        maximum: usize,
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
    DuplicatePresentation {
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

pub fn build_pilot_participant_schedule(
    manifest: &FrozenStudyManifest,
    base_schedule: &BlindedSchedule,
    codebook: &BlindingCodebook,
    protocol: &FrozenPilotProtocol,
    cohort: &PilotCohortSpec,
    pilot_secret_key: [u8; 32],
) -> Result<(PilotParticipantScheduleBook, PilotScheduleAudit), Vec<PilotScheduleIssue>> {
    let mut issues = validate_cohort(cohort, protocol);
    if protocol.protocol_version != crate::pilot_protocol::PILOT_PROTOCOL_VERSION {
        issues.push(PilotScheduleIssue::PilotProtocol {
            issue: PilotProtocolIssue::WrongProtocolVersion {
                found: protocol.protocol_version.clone(),
            },
        });
    }
    if !validate_blinded_schedule(manifest, base_schedule, Some(codebook)).is_empty() {
        issues.push(PilotScheduleIssue::InvalidBaseSchedule);
    }
    if sha256_hex(&pilot_secret_key) != protocol.pilot_randomization_commitment_sha256 {
        issues.push(PilotScheduleIssue::SecretCommitmentMismatch);
    }
    if protocol.pilot_randomization_commitment_sha256 == manifest.randomization_commitment_sha256 {
        issues.push(PilotScheduleIssue::ConfirmatoryCommitmentReused);
    }
    let manifest_sha256 = digest_or_issue(manifest, "manifest", &mut issues);
    let base_schedule_sha256 = digest_or_issue(base_schedule, "base_schedule", &mut issues);
    let pilot_protocol_sha256 = digest_or_issue(protocol, "pilot_protocol", &mut issues);
    if base_schedule.manifest_sha256 != manifest_sha256 {
        issues.push(PilotScheduleIssue::DigestMismatch {
            field: "base_schedule.manifest_sha256".into(),
        });
    }
    if protocol.manifest_sha256 != manifest_sha256 {
        issues.push(PilotScheduleIssue::DigestMismatch {
            field: "pilot_protocol.manifest_sha256".into(),
        });
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
        .filter(|fixture| fixture.split == StudySplit::Pilot)
        .collect();
    fixtures.sort_by(|left, right| left.key.cmp(&right.key));

    let mut rng = StdRng::from_seed(pilot_secret_key);
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
                        .expect("validated codebook contains each pilot fixture-arm")
                        .to_string()
                })
                .collect();
            let block_id = format!(
                "pilot-{}-{}-{}-{:016x}",
                cohort.cohort_id,
                cohort.wave_id,
                participant_token,
                stable_hash(&format!(
                    "{}|{}|{}|{}",
                    cohort.wave_id, participant_token, fixture.key.fixture_id, fixture.key.seed
                ))
            );
            blocks.push(ParticipantBlockAssignment {
                block_id: block_id.clone(),
                participant_token: participant_token.clone(),
                key: fixture.key.clone(),
                ordered_presentation_ids,
            });
            sequences.push(PilotPrivateSequence {
                block_id,
                participant_token: participant_token.clone(),
                key: fixture.key.clone(),
                sequence_index: sequence_index as u8,
                arm_order,
            });
        }
    }
    let schedule = PilotParticipantScheduleBook {
        schedule_version: PILOT_PARTICIPANT_SCHEDULE_VERSION.into(),
        manifest_sha256: manifest_sha256.clone(),
        base_schedule_sha256: base_schedule_sha256.clone(),
        pilot_protocol_sha256: pilot_protocol_sha256.clone(),
        pilot_randomization_commitment_sha256: protocol
            .pilot_randomization_commitment_sha256
            .clone(),
        cohort_id: cohort.cohort_id.clone(),
        wave_id: cohort.wave_id.clone(),
        blocks,
    };
    let audit = PilotScheduleAudit {
        schedule_version: PILOT_PARTICIPANT_SCHEDULE_VERSION.into(),
        manifest_sha256,
        base_schedule_sha256,
        pilot_protocol_sha256,
        cohort_id: cohort.cohort_id.clone(),
        wave_id: cohort.wave_id.clone(),
        sequences,
    };
    let validation = validate_pilot_participant_schedule(
        manifest,
        base_schedule,
        codebook,
        protocol,
        cohort,
        &schedule,
        Some(&audit),
    );
    if validation.is_empty() {
        Ok((schedule, audit))
    } else {
        Err(validation)
    }
}

pub fn validate_pilot_participant_schedule(
    manifest: &FrozenStudyManifest,
    base_schedule: &BlindedSchedule,
    codebook: &BlindingCodebook,
    protocol: &FrozenPilotProtocol,
    cohort: &PilotCohortSpec,
    schedule: &PilotParticipantScheduleBook,
    audit: Option<&PilotScheduleAudit>,
) -> Vec<PilotScheduleIssue> {
    let mut issues = validate_cohort(cohort, protocol);
    if schedule.schedule_version != PILOT_PARTICIPANT_SCHEDULE_VERSION {
        issues.push(PilotScheduleIssue::WrongVersion {
            found: schedule.schedule_version.clone(),
        });
    }
    verify_digest(manifest, &schedule.manifest_sha256, "manifest", &mut issues);
    verify_digest(
        base_schedule,
        &schedule.base_schedule_sha256,
        "base_schedule",
        &mut issues,
    );
    verify_digest(
        protocol,
        &schedule.pilot_protocol_sha256,
        "pilot_protocol",
        &mut issues,
    );
    if schedule.pilot_randomization_commitment_sha256
        != protocol.pilot_randomization_commitment_sha256
    {
        issues.push(PilotScheduleIssue::SecretCommitmentMismatch);
    }
    if schedule.cohort_id != cohort.cohort_id {
        issues.push(PilotScheduleIssue::EmptyCohortField {
            field: "cohort_id_mismatch".into(),
        });
    }
    if schedule.wave_id != cohort.wave_id {
        issues.push(PilotScheduleIssue::EmptyCohortField {
            field: "wave_id_mismatch".into(),
        });
    }

    let pilot_keys: BTreeSet<_> = manifest
        .fixtures
        .iter()
        .filter(|fixture| fixture.split == StudySplit::Pilot)
        .map(|fixture| fixture.key.clone())
        .collect();
    let participant_tokens: BTreeSet<_> = cohort.participant_tokens.iter().cloned().collect();
    let presentations: BTreeMap<_, _> = base_schedule
        .presentations
        .iter()
        .filter(|presentation| presentation.split == StudySplit::Pilot)
        .map(|presentation| (presentation.presentation_id.as_str(), presentation))
        .collect();
    let mut block_ids = BTreeSet::new();
    let mut participant_fixtures = BTreeSet::new();
    let mut by_fixture: BTreeMap<FrozenTrialKey, Vec<&ParticipantBlockAssignment>> =
        BTreeMap::new();
    for block in &schedule.blocks {
        if !block_ids.insert(block.block_id.clone()) {
            issues.push(PilotScheduleIssue::DuplicateBlockId {
                block_id: block.block_id.clone(),
            });
        }
        if !participant_tokens.contains(&block.participant_token)
            || !pilot_keys.contains(&block.key)
        {
            issues.push(PilotScheduleIssue::UnexpectedParticipantFixture {
                participant_token: block.participant_token.clone(),
                key: block.key.clone(),
            });
        }
        if !participant_fixtures.insert((block.participant_token.clone(), block.key.clone())) {
            issues.push(PilotScheduleIssue::DuplicateParticipantFixture {
                participant_token: block.participant_token.clone(),
                key: block.key.clone(),
            });
        }
        if block.ordered_presentation_ids.len() != 4 {
            issues.push(PilotScheduleIssue::IncompleteBlock {
                block_id: block.block_id.clone(),
                found: block.ordered_presentation_ids.len(),
            });
        }
        let mut seen = BTreeSet::new();
        for presentation_id in &block.ordered_presentation_ids {
            if !seen.insert(presentation_id) {
                issues.push(PilotScheduleIssue::DuplicatePresentation {
                    block_id: block.block_id.clone(),
                    presentation_id: presentation_id.clone(),
                });
            }
            match presentations.get(presentation_id.as_str()) {
                None => issues.push(PilotScheduleIssue::UnknownPresentation {
                    block_id: block.block_id.clone(),
                    presentation_id: presentation_id.clone(),
                }),
                Some(presentation) if presentation.key != block.key => {
                    issues.push(PilotScheduleIssue::PresentationFixtureMismatch {
                        block_id: block.block_id.clone(),
                        presentation_id: presentation_id.clone(),
                    });
                }
                Some(_) => {}
            }
        }
        by_fixture.entry(block.key.clone()).or_default().push(block);
    }
    for participant_token in &participant_tokens {
        for key in &pilot_keys {
            if !participant_fixtures.contains(&(participant_token.clone(), key.clone())) {
                issues.push(PilotScheduleIssue::MissingParticipantFixture {
                    participant_token: participant_token.clone(),
                    key: key.clone(),
                });
            }
        }
    }
    validate_balance(&by_fixture, &mut issues);
    if let Some(audit) = audit {
        validate_audit(codebook, schedule, audit, &mut issues);
    }
    issues
}

fn validate_cohort(
    cohort: &PilotCohortSpec,
    protocol: &FrozenPilotProtocol,
) -> Vec<PilotScheduleIssue> {
    let mut issues = Vec::new();
    for (field, value) in [
        ("cohort_id", cohort.cohort_id.as_str()),
        ("wave_id", cohort.wave_id.as_str()),
    ] {
        if value.trim().is_empty() {
            issues.push(PilotScheduleIssue::EmptyCohortField {
                field: field.into(),
            });
        }
    }
    if cohort.participant_tokens.is_empty() {
        issues.push(PilotScheduleIssue::EmptyParticipantRegistry);
    }
    if cohort.participant_tokens.len() % 4 != 0 {
        issues.push(PilotScheduleIssue::ParticipantCountNotDivisibleByFour {
            found: cohort.participant_tokens.len(),
        });
    }
    if cohort.participant_tokens.len() > protocol.thresholds.cohort_wave_size {
        issues.push(PilotScheduleIssue::TooManyParticipantsInWave {
            found: cohort.participant_tokens.len(),
            maximum: protocol.thresholds.cohort_wave_size,
        });
    }
    let mut seen = BTreeSet::new();
    for (index, token) in cohort.participant_tokens.iter().enumerate() {
        if token.trim().is_empty() {
            issues.push(PilotScheduleIssue::EmptyParticipantToken { index });
        } else if !seen.insert(token.clone()) {
            issues.push(PilotScheduleIssue::DuplicateParticipantToken {
                participant_token: token.clone(),
            });
        }
    }
    issues
}

fn validate_balance(
    by_fixture: &BTreeMap<FrozenTrialKey, Vec<&ParticipantBlockAssignment>>,
    issues: &mut Vec<PilotScheduleIssue>,
) {
    for (key, blocks) in by_fixture {
        if blocks.is_empty() {
            continue;
        }
        let expected_position = blocks.len() / 4;
        let expected_carryover = blocks.len() / 4;
        let all_ids: BTreeSet<_> = blocks
            .iter()
            .flat_map(|block| block.ordered_presentation_ids.iter().cloned())
            .collect();
        for presentation_id in &all_ids {
            let mut counts = vec![0usize; 4];
            for block in blocks {
                if let Some(position) = block
                    .ordered_presentation_ids
                    .iter()
                    .position(|value| value == presentation_id)
                {
                    counts[position] += 1;
                }
            }
            if counts.iter().any(|count| *count != expected_position) {
                issues.push(PilotScheduleIssue::PositionImbalance {
                    key: key.clone(),
                    presentation_id: presentation_id.clone(),
                    counts,
                });
            }
        }
        for left in &all_ids {
            for right in &all_ids {
                if left == right {
                    continue;
                }
                let found = blocks
                    .iter()
                    .map(|block| {
                        block
                            .ordered_presentation_ids
                            .windows(2)
                            .filter(|pair| pair[0] == *left && pair[1] == *right)
                            .count()
                    })
                    .sum();
                if found != expected_carryover {
                    issues.push(PilotScheduleIssue::CarryoverImbalance {
                        key: key.clone(),
                        left_presentation_id: left.clone(),
                        right_presentation_id: right.clone(),
                        found,
                        expected: expected_carryover,
                    });
                }
            }
        }
    }
}

fn validate_audit(
    codebook: &BlindingCodebook,
    schedule: &PilotParticipantScheduleBook,
    audit: &PilotScheduleAudit,
    issues: &mut Vec<PilotScheduleIssue>,
) {
    for (field, left, right) in [
        (
            "schedule_version",
            audit.schedule_version.as_str(),
            schedule.schedule_version.as_str(),
        ),
        (
            "manifest_sha256",
            audit.manifest_sha256.as_str(),
            schedule.manifest_sha256.as_str(),
        ),
        (
            "base_schedule_sha256",
            audit.base_schedule_sha256.as_str(),
            schedule.base_schedule_sha256.as_str(),
        ),
        (
            "pilot_protocol_sha256",
            audit.pilot_protocol_sha256.as_str(),
            schedule.pilot_protocol_sha256.as_str(),
        ),
        (
            "cohort_id",
            audit.cohort_id.as_str(),
            schedule.cohort_id.as_str(),
        ),
        ("wave_id", audit.wave_id.as_str(), schedule.wave_id.as_str()),
    ] {
        if left != right {
            issues.push(PilotScheduleIssue::DigestMismatch {
                field: format!("audit.{field}"),
            });
        }
    }
    let arm_by_id: BTreeMap<_, _> = codebook
        .entries
        .iter()
        .map(|entry| (entry.presentation_id.as_str(), entry.arm))
        .collect();
    let audit_by_block: BTreeMap<_, _> = audit
        .sequences
        .iter()
        .map(|sequence| (sequence.block_id.as_str(), sequence))
        .collect();
    for block in &schedule.blocks {
        let Some(sequence) = audit_by_block.get(block.block_id.as_str()) else {
            issues.push(PilotScheduleIssue::AuditMismatch {
                block_id: block.block_id.clone(),
            });
            continue;
        };
        let actual: Vec<_> = block
            .ordered_presentation_ids
            .iter()
            .filter_map(|id| arm_by_id.get(id.as_str()).copied())
            .collect();
        if actual != sequence.arm_order
            || sequence.participant_token != block.participant_token
            || sequence.key != block.key
        {
            issues.push(PilotScheduleIssue::AuditMismatch {
                block_id: block.block_id.clone(),
            });
        }
    }
}

fn verify_digest<T: Serialize>(
    value: &T,
    expected: &str,
    field: &str,
    issues: &mut Vec<PilotScheduleIssue>,
) {
    match canonical_json_sha256(value) {
        Ok(found) if found == expected => {}
        Ok(_) => issues.push(PilotScheduleIssue::DigestMismatch {
            field: field.into(),
        }),
        Err(_) => issues.push(PilotScheduleIssue::SerializationFailed {
            field: field.into(),
        }),
    }
}

fn digest_or_issue<T: Serialize>(
    value: &T,
    field: &str,
    issues: &mut Vec<PilotScheduleIssue>,
) -> String {
    match canonical_json_sha256(value) {
        Ok(value) => value,
        Err(_) => {
            issues.push(PilotScheduleIssue::SerializationFailed {
                field: field.into(),
            });
            String::new()
        }
    }
}

fn stable_hash(value: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in value.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn williams_square_balances_order_and_carryover() {
        let mut positions = [[0usize; 4]; 4];
        let mut carryover = [[0usize; 4]; 4];
        for sequence in WILLIAMS_FOUR {
            for (position, treatment) in sequence.into_iter().enumerate() {
                positions[treatment][position] += 1;
            }
            for pair in sequence.windows(2) {
                carryover[pair[0]][pair[1]] += 1;
            }
        }
        assert!(positions.into_iter().flatten().all(|count| count == 1));
        for (left, row) in carryover.into_iter().enumerate() {
            for (right, count) in row.into_iter().enumerate() {
                assert_eq!(count, usize::from(left != right));
            }
        }
    }
}
